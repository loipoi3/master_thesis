import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Dict, Optional, List
import numpy as np
from tqdm import tqdm

from vae import VAE
from env import VAELatentEnv
from config import VAEConfig, RLConfig, TrainingConfig


class RLAgentWrapper:
    def __init__(self, env: VAELatentEnv, rl_config: RLConfig, device: str = "cpu"):
        self.env = env
        self.rl_config = rl_config
        self.device = device
        self.model = None

    def setup_model(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        def make_env():
            return self.env

        vec_env = DummyVecEnv([make_env])

        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.rl_config.learning_rate,
            n_steps=self.rl_config.n_steps_per_update,
            batch_size=self.rl_config.batch_size,
            n_epochs=self.rl_config.n_epochs,
            gamma=self.rl_config.gamma,
            gae_lambda=self.rl_config.gae_lambda,
            clip_range=self.rl_config.clip_range,
            verbose=0,
            device=self.device
        )

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        if self.model is None:
            action = np.zeros(self.env.action_space.shape, dtype=np.float32)
            return action, None
        return self.model.predict(obs, deterministic=deterministic)

    def learn(self, total_timesteps: int):
        if self.model is not None:
            self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: str):
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, env=self.env)


class Trainer:
    def __init__(
            self,
            vae_config: VAEConfig,
            rl_config: RLConfig,
            training_config: TrainingConfig,
            enable_rl: bool = True
    ):
        self.vae_config = vae_config
        self.rl_config = rl_config
        self.training_config = training_config
        self.device = training_config.device
        self.enable_rl = enable_rl

        self.vae = VAE(vae_config).to(self.device)
        self.vae_optimizer = torch.optim.Adam(
            self.vae.parameters(),
            lr=vae_config.learning_rate
        )

        self.env = VAELatentEnv(
            self.vae,
            vae_config,
            rl_config,
            self.device
        )

        self.rl_agent = RLAgentWrapper(self.env, rl_config, "cpu")
        if enable_rl:
            self.rl_agent.setup_model()

        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.vae_losses: List[float] = []
        self.rl_rewards: List[float] = []

    def setup_data(self) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])

        if self.training_config.dataset == "mnist":
            train_dataset = datasets.MNIST(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                root="./data", train=False, download=True, transform=transform
            )
        elif self.training_config.dataset == "fashion_mnist":
            train_dataset = datasets.FashionMNIST(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.FashionMNIST(
                root="./data", train=False, download=True, transform=transform
            )
        else:
            raise ValueError(f"Unknown dataset: {self.training_config.dataset}")

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.training_config.batch_size,
            shuffle=True, num_workers=0, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.training_config.batch_size,
            shuffle=False, num_workers=0, pin_memory=True
        )

    def train_vae_phase(self, epoch: int) -> Dict[str, float]:
        assert self.train_loader is not None
        self.vae.train()
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"VAE Epoch {epoch}")

        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(self.device)

            # FIXED: Get shifts (delta_mu, delta_log_var) from the RL agent
            delta_mu = None
            delta_log_var = None

            if self.enable_rl:
                with torch.no_grad():
                    mu, log_var = self.vae.encode(data)
                    mu_mean, mu_std = mu.mean(dim=0), mu.std(dim=0)
                    log_var_mean, log_var_std = log_var.mean(dim=0), log_var.std(dim=0)
                    obs = torch.cat([mu_mean, mu_std, log_var_mean, log_var_std], dim=0)
                    obs_np = obs.cpu().numpy().astype(np.float32)

                    action, _ = self.rl_agent.predict(obs_np, deterministic=True)

                    latent_dim = self.vae_config.latent_dim
                    delta_mu = torch.tensor(action[:latent_dim], device=self.device, dtype=torch.float32)
                    delta_log_var = torch.tensor(action[latent_dim:], device=self.device, dtype=torch.float32)

            self.vae_optimizer.zero_grad()

            # FIXED: Pass the shifts into the VAE forward pass
            x_recon, mu_out, log_var_out, z = self.vae(
                data, delta_mu=delta_mu, delta_log_var=delta_log_var
            )

            # beta remains static (as defined in config)
            loss, recon_loss, kl_loss = self.vae.compute_loss(
                data, x_recon, mu_out, log_var_out, beta=self.vae_config.beta_kl
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
            self.vae_optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}"
            })

        avg_loss = total_loss / num_batches
        self.vae_losses.append(avg_loss)

        return {
            "loss": avg_loss,
            "recon": total_recon / num_batches,
            "kl": total_kl / num_batches
        }

    def train_rl_phase(self, phase: int) -> Dict[str, float]:
        assert self.train_loader is not None
        self.vae.eval()

        total_reward = 0.0
        num_updates = 0
        n_episodes = min(self.rl_config.n_steps_per_update, 32)

        for episode in range(n_episodes):
            data_iter = iter(self.train_loader)
            try:
                batch, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch, labels = next(data_iter)

            batch = batch.to(self.device)
            self.env.set_batch(batch, labels)

            obs, _ = self.env.reset()

            # Train PPO (agent interacts with the environment sequentially for max_steps)
            total_timesteps = self.rl_config.n_epochs * 32
            self.rl_agent.learn(total_timesteps=total_timesteps)
            num_updates += 1

            if episode % 10 == 0:
                print(f"  RL Episode {episode + 1}/{n_episodes}")

        return {"reward": total_reward / max(num_updates, 1)}

    def train(self) -> Dict[str, List]:
        os.makedirs(self.training_config.save_dir, exist_ok=True)

        for cycle in range(self.training_config.alternating_cycles):
            print(f"\n{'=' * 60}")
            print(f"Training Cycle {cycle + 1}/{self.training_config.alternating_cycles}")
            print(f"{'=' * 60}")

            print(f"\n--- Phase A: Training VAE ---")
            for epoch in range(self.training_config.vae_epochs):
                vae_metrics = self.train_vae_phase(epoch + 1)
                print(f"Epoch {epoch + 1}: Loss={vae_metrics['loss']:.4f}, "
                      f"Recon={vae_metrics['recon']:.4f}, KL={vae_metrics['kl']:.4f}")

            print(f"\n--- Phase B: Training RL Agent ---")
            if self.enable_rl:
                _ = self.train_rl_phase(cycle + 1)
                print(f"RL Phase Complete")
            else:
                print(f"RL Phase Skipped (disabled)")

            checkpoint_path = os.path.join(
                self.training_config.save_dir, f"vae_cycle_{cycle + 1}.pt"
            )
            torch.save({
                'vae_state_dict': self.vae.state_dict(),
                'optimizer_state_dict': self.vae_optimizer.state_dict(),
                'cycle': cycle + 1,
                'vae_losses': self.vae_losses
            }, checkpoint_path)

            rl_path = os.path.join(
                self.training_config.save_dir, f"rl_agent_cycle_{cycle + 1}.zip"
            )
            self.rl_agent.save(rl_path)

        return {
            "vae_losses": self.vae_losses,
            "rl_rewards": self.rl_rewards
        }

    def evaluate(self) -> Dict[str, float]:
        assert self.test_loader is not None
        self.vae.eval()
        total_loss, total_recon, total_kl, num_batches = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                x_recon, mu, log_var, z = self.vae(data)
                loss, recon_loss, kl_loss = self.vae.compute_loss(data, x_recon, mu, log_var)

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                num_batches += 1

        return {
            "test_loss": total_loss / num_batches,
            "test_recon": total_recon / num_batches,
            "test_kl": total_kl / num_batches
        }

    def get_latent_representations(
            self,
            loader: Optional[DataLoader] = None,
            use_rl: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.vae.eval()
        if loader is None:
            assert self.test_loader is not None
            loader = self.test_loader

        all_z = []
        all_labels = []

        with torch.no_grad():
            for data, labels in loader:
                data = data.to(self.device)
                mu, log_var = self.vae.encode(data)

                # FIXED: Apply RL shift during representation extraction/visualization
                if use_rl and self.enable_rl:
                    mu_mean, mu_std = mu.mean(dim=0), mu.std(dim=0)
                    log_var_mean, log_var_std = log_var.mean(dim=0), log_var.std(dim=0)
                    obs = torch.cat([mu_mean, mu_std, log_var_mean, log_var_std], dim=0)
                    obs_np = obs.cpu().numpy().astype(np.float32)

                    action, _ = self.rl_agent.predict(obs_np, deterministic=True)
                    latent_dim = self.vae_config.latent_dim

                    delta_mu = torch.tensor(action[:latent_dim], device=self.device, dtype=torch.float32)
                    delta_log_var = torch.tensor(action[latent_dim:], device=self.device, dtype=torch.float32)

                    mu = mu + delta_mu
                    log_var = torch.clamp(log_var + delta_log_var, min=-10, max=10)

                z = self.vae.reparameterize(mu, log_var)
                all_z.append(z.cpu().numpy())
                all_labels.append(labels.numpy())

        return np.concatenate(all_z, axis=0), np.concatenate(all_labels, axis=0)

    def save_final(self, path: str) -> None:
        torch.save({
            'vae_state_dict': self.vae.state_dict(),
            'vae_config': self.vae_config,
            'rl_config': self.rl_config,
            'vae_losses': self.vae_losses,
            'rl_rewards': self.rl_rewards
        }, path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.vae_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'vae_losses' in checkpoint:
            self.vae_losses = checkpoint['vae_losses']