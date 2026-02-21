from dataclasses import dataclass, field
from typing import List, Tuple
import torch


@dataclass
class VAEConfig:
    latent_dim: int = 2
    encoder_hidden: List[int] = field(default_factory=lambda: [512, 256])
    decoder_hidden: List[int] = field(default_factory=lambda: [256, 512])
    input_dim: int = 784
    learning_rate: float = 1e-3
    beta_kl: float = 1.0


@dataclass
class RLConfig:
    reward_weights: Tuple[float, float] = (1.0, 2.0)
    n_steps_per_update: int = 64
    n_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2


@dataclass
class TrainingConfig:
    dataset: str = "mnist"
    batch_size: int = 128
    vae_epochs: int = 10
    rl_episodes_per_phase: int = 5
    alternating_cycles: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
