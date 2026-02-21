import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
from gymnasium import spaces

from vae import VAE
from config import VAEConfig, RLConfig


class VAELatentEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(
            self,
            vae: VAE,
            vae_config: VAEConfig,
            rl_config: RLConfig,
            device: str = "cpu"
    ):
        super().__init__()

        self.vae = vae
        self.vae_config = vae_config
        self.rl_config = rl_config
        self.device = device

        self.vae.eval()

        latent_dim = vae_config.latent_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(latent_dim * 4,),
            dtype=np.float32
        )

        # FIXED: Action is now a vector of shifts (delta_mu and delta_log_var) for each axis.
        self.action_space = spaces.Box(
            low=-1.0,  # Bound the shift per step to prevent exploding spaces
            high=1.0,
            shape=(latent_dim * 2,),
            dtype=np.float32
        )

        self.current_batch: Optional[torch.Tensor] = None
        self.current_mu: Optional[torch.Tensor] = None
        self.current_log_var: Optional[torch.Tensor] = None

        # FIXED: Added tracking of the current shifted state within a multi-step episode
        self.current_shifted_mu: Optional[torch.Tensor] = None
        self.current_shifted_log_var: Optional[torch.Tensor] = None

        self.current_labels: Optional[torch.Tensor] = None
        self.current_step: int = 0
        # FIXED: Allow the agent to take multiple steps (iteratively move the space)
        self.max_steps: int = 5

    def set_batch(
            self,
            batch: torch.Tensor,
            labels: Optional[torch.Tensor] = None
    ) -> None:
        self.current_batch = batch.to(self.device)
        self.current_labels = labels
        with torch.no_grad():
            self.current_mu, self.current_log_var = self.vae.encode(self.current_batch)

    def _get_obs(self) -> np.ndarray:
        if self.current_shifted_mu is None or self.current_shifted_log_var is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # Calculate observations based on the ALREADY SHIFTED distributions
        mu_mean = self.current_shifted_mu.mean(dim=0)
        mu_std = self.current_shifted_mu.std(dim=0)
        log_var_mean = self.current_shifted_log_var.mean(dim=0)
        log_var_std = self.current_shifted_log_var.std(dim=0)

        obs = torch.cat([mu_mean, mu_std, log_var_mean, log_var_std], dim=0)
        return obs.cpu().numpy().astype(np.float32)

    def _compute_reward(
            self,
            mu: torch.Tensor,
            log_var: torch.Tensor,
            z: torch.Tensor,
            x_recon: torch.Tensor,
            beta: float = 1.0
    ) -> Tuple[float, float, float]:
        w_recon, w_density = self.rl_config.reward_weights

        if self.current_batch is None:
            return 0.0, 0.0, 0.0

        recon_loss = F.mse_loss(x_recon, self.current_batch, reduction='mean')
        r_recon = -recon_loss.item()

        r_density = self._compute_density_reward(z)

        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        r_kl = beta * kl_loss.item()

        total_reward = w_recon * r_recon + w_density * r_density - 0.1 * r_kl

        return total_reward, r_recon, r_density

    def _compute_density_reward(self, z: torch.Tensor) -> float:
        if z.size(0) < 4:
            return 0.0

        z_np = z.detach().cpu().numpy()
        grid_size = 5

        # FIXED: Use global coordinates [-3, 3] instead of local batch limits.
        # This prevents the agent from getting rewarded for collapsing all points into a single dense cluster.
        z_normalized = np.clip((z_np + 3.0) / 6.0, 0.0, 1.0)

        grid_coords = (z_normalized * grid_size).astype(int)
        grid_coords = np.clip(grid_coords, 0, grid_size - 1)

        if self.vae_config.latent_dim == 2:
            occupied = set((coord[0], coord[1]) for coord in grid_coords)
            num_occupied = len(occupied)
            total_cells = grid_size * grid_size
        else:
            occupied = set(tuple(coord) for coord in grid_coords)
            num_occupied = len(occupied)
            total_cells = grid_size ** self.vae_config.latent_dim

        coverage = num_occupied / total_cells

        if z_np.shape[0] > 10:
            center = np.full(self.vae_config.latent_dim, 0.5)
            distances_to_center = np.sqrt(((z_normalized - center) ** 2).sum(axis=1))
            coverage_bonus = 1.0 / (1.0 + distances_to_center.mean())
            coverage = 0.7 * coverage + 0.3 * coverage_bonus

        return coverage

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_shifted_mu is None or self.current_batch is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {"error": "No batch"}

        # FIXED: Split the action into delta_mu and delta_log_var
        latent_dim = self.vae_config.latent_dim
        delta_mu_np = action[:latent_dim]
        delta_log_var_np = action[latent_dim:]

        delta_mu = torch.tensor(delta_mu_np, device=self.device, dtype=torch.float32)
        delta_log_var = torch.tensor(delta_log_var_np, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            # Apply shift to the current batch distribution incrementally
            self.current_shifted_mu = self.current_shifted_mu + delta_mu
            self.current_shifted_log_var = self.current_shifted_log_var + delta_log_var
            self.current_shifted_log_var = torch.clamp(self.current_shifted_log_var, min=-10, max=10)

            z = self.vae.reparameterize(self.current_shifted_mu, self.current_shifted_log_var)
            x_recon = self.vae.decode(z)

        # Calculate reward for the newly shifted state
        total_reward, r_recon, r_density = self._compute_reward(
            self.current_shifted_mu, self.current_shifted_log_var, z, x_recon, self.vae_config.beta_kl
        )

        self.current_step += 1
        # FIXED: Episode only terminates after max_steps to allow for sequential improvements
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "reward_recon": r_recon,
            "reward_density": r_density,
            "reward_total": total_reward
        }

        obs = self._get_obs()
        return obs, total_reward, terminated, truncated, info

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0

        if self.current_batch is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        with torch.no_grad():
            self.current_mu, self.current_log_var = self.vae.encode(self.current_batch)
            # FIXED: Reset shifted variables back to original values at the start of the episode
            self.current_shifted_mu = self.current_mu.clone()
            self.current_shifted_log_var = self.current_log_var.clone()

        obs = self._get_obs()
        return obs, {}

    def render(self, mode: str = "console"):
        if mode == "console" and self.current_shifted_mu is not None:
            print(f"Step {self.current_step}: mu mean = {self.current_shifted_mu.mean(dim=0).cpu().numpy()}")


class VAELatentVecEnv:
    def __init__(
            self,
            vae: VAE,
            vae_config: VAEConfig,
            rl_config: RLConfig,
            device: str = "cpu",
            num_envs: int = 1
    ):
        self.env = VAELatentEnv(vae, vae_config, rl_config, device)
        self.num_envs = num_envs
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def set_batch(self, batch: torch.Tensor, labels: Optional[torch.Tensor] = None):
        self.env.set_batch(batch, labels)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode: str = "console"):
        return self.env.render(mode)