import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Tuple, Optional, List, Dict
import torch
from torch.utils.data import DataLoader

from vae import VAE
from train import Trainer
from config import VAEConfig, RLConfig, TrainingConfig


def plot_latent_space(
    z: np.ndarray,
    labels: np.ndarray,
    title: str = "Latent Space",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            z[mask, 0], 
            z[mask, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.6,
            s=5
        )
    
    ax.set_xlabel('z₁', fontsize=12)
    ax.set_ylabel('z₂', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_latent_comparison(
    z_vanilla: np.ndarray,
    labels_vanilla: np.ndarray,
    z_rl: np.ndarray,
    labels_rl: np.ndarray,
    title: str = "Vanilla VAE vs RL-Optimized VAE",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 7)
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    unique_labels = np.unique(labels_vanilla)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    ax = axes[0]
    for i, label in enumerate(unique_labels):
        mask = labels_vanilla == label
        ax.scatter(
            z_vanilla[mask, 0], 
            z_vanilla[mask, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.6,
            s=3
        )
    ax.set_xlabel('z₁', fontsize=12)
    ax.set_ylabel('z₂', fontsize=12)
    ax.set_title('Vanilla VAE', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for i, label in enumerate(unique_labels):
        mask = labels_rl == label
        ax.scatter(
            z_rl[mask, 0], 
            z_rl[mask, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.6,
            s=3
        )
    ax.set_xlabel('z₁', fontsize=12)
    ax.set_ylabel('z₂', fontsize=12)
    ax.set_title('RL-Optimized VAE', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def compute_hole_metric(z: np.ndarray, grid_size: int = 20) -> float:
    x_min, x_max = z[:, 0].min(), z[:, 0].max()
    y_min, y_max = z[:, 1].min(), z[:, 1].max()
    
    x_bins = np.linspace(x_min, x_max, grid_size + 1)
    y_bins = np.linspace(y_min, y_max, grid_size + 1)
    
    hist, _, _ = np.histogram2d(z[:, 0], z[:, 1], bins=[x_bins, y_bins])
    
    empty_cells = np.sum(hist == 0)
    total_cells = grid_size * grid_size
    
    hole_ratio = empty_cells / total_cells
    return hole_ratio


def compute_coverage_metric(z: np.ndarray, grid_size: int = 10) -> float:
    z_min = z.min(axis=0)
    z_max = z.max(axis=0)
    z_range = z_max - z_min + 1e-8
    z_normalized = (z - z_min) / z_range
    
    grid_coords = (z_normalized * grid_size).astype(int)
    grid_coords = np.clip(grid_coords, 0, grid_size - 1)
    
    occupied = set()
    for coord in grid_coords:
        occupied.add((coord[0], coord[1]))
    
    coverage = len(occupied) / (grid_size * grid_size)
    return coverage


def compute_distribution_metrics(z: np.ndarray) -> Dict[str, float]:
    center = z.mean(axis=0)
    distances_to_center = np.sqrt(((z - center) ** 2).sum(axis=1))
    
    std_per_dim = z.std(axis=0)
    
    return {
        "mean_dist_to_center": distances_to_center.mean(),
        "std_dist_to_center": distances_to_center.std(),
        "std_z1": std_per_dim[0],
        "std_z2": std_per_dim[1] if len(std_per_dim) > 1 else 0.0,
        "spread": np.linalg.norm(std_per_dim)
    }


def plot_training_curves(
    vae_losses: List[float],
    rl_rewards: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    ax.plot(vae_losses, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('VAE Training Loss', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    if rl_rewards and len(rl_rewards) > 0:
        ax.plot(rl_rewards, linewidth=2, color='orange')
        ax.set_xlabel('Update', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('RL Agent Reward', fontsize=14)
    else:
        ax.text(0.5, 0.5, 'No RL rewards recorded', ha='center', va='center', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_reconstruction(
    vae: VAE,
    data_loader: DataLoader,
    device: str = "cpu",
    num_samples: int = 10,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4)
) -> None:
    vae.eval()
    
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images = images[:num_samples].to(device)
    
    with torch.no_grad():
        reconstructions, _, _, _ = vae(images)
    
    images_np = images.cpu().numpy()
    recon_np = reconstructions.cpu().numpy()
    
    fig, axes = plt.subplots(2, num_samples, figsize=figsize)
    
    for i in range(num_samples):
        ax = axes[0, i]
        ax.imshow(images_np[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title('Original', fontsize=10)
        
        ax = axes[1, i]
        ax.imshow(recon_np[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_samples_from_prior(
    vae: VAE,
    device: str = "cpu",
    num_samples: int = 25,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10)
) -> None:
    vae.eval()
    
    with torch.no_grad():
        samples = vae.sample(num_samples, device)
    
    samples_np = samples.cpu().numpy()
    
    n_cols = int(np.sqrt(num_samples))
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(samples_np[i].reshape(28, 28), cmap='gray')
        axes[i].axis('off')
    
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Samples from Prior', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def run_comparison_experiment(
    vae_config: VAEConfig,
    rl_config: RLConfig,
    training_config: TrainingConfig,
    save_dir: str = "results"
) -> Tuple[Trainer, Trainer]:
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Training Vanilla VAE (baseline)")
    print("=" * 60)
    
    vanilla_config = TrainingConfig(
        dataset=training_config.dataset,
        batch_size=training_config.batch_size,
        vae_epochs=training_config.vae_epochs * training_config.alternating_cycles,
        rl_episodes_per_phase=0,
        alternating_cycles=1,
        device=training_config.device,
        seed=training_config.seed,
        save_dir=os.path.join(save_dir, "vanilla")
    )
    
    vanilla_trainer = Trainer(vae_config, rl_config, vanilla_config, enable_rl=False)
    vanilla_trainer.setup_data()
    vanilla_trainer.train()
    
    print("\n" + "=" * 60)
    print("Training RL-Optimized VAE")
    print("=" * 60)
    
    rl_trainer = Trainer(vae_config, rl_config, training_config, enable_rl=True)
    rl_trainer.setup_data()
    rl_trainer.train()
    
    z_vanilla, labels_vanilla = vanilla_trainer.get_latent_representations(use_rl=False)
    z_rl, labels_rl = rl_trainer.get_latent_representations(use_rl=True)
    
    hole_vanilla = compute_hole_metric(z_vanilla)
    hole_rl = compute_hole_metric(z_rl)
    
    coverage_vanilla = compute_coverage_metric(z_vanilla)
    coverage_rl = compute_coverage_metric(z_rl)
    
    dist_vanilla = compute_distribution_metrics(z_vanilla)
    dist_rl = compute_distribution_metrics(z_rl)
    
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"\nHole Metric (lower is better):")
    print(f"  Vanilla VAE:       {hole_vanilla:.4f}")
    print(f"  RL-Optimized VAE:  {hole_rl:.4f}")
    print(f"  Improvement:       {(hole_vanilla - hole_rl) / hole_vanilla * 100:.2f}%")
    
    print(f"\nCoverage Metric (higher is better):")
    print(f"  Vanilla VAE:       {coverage_vanilla:.4f}")
    print(f"  RL-Optimized VAE:  {coverage_rl:.4f}")
    print(f"  Improvement:       {(coverage_rl - coverage_vanilla) / coverage_vanilla * 100:.2f}%")
    
    print(f"\nDistribution Metrics:")
    print(f"  Spread (norm of std):")
    print(f"    Vanilla:       {dist_vanilla['spread']:.4f}")
    print(f"    RL-Optimized:  {dist_rl['spread']:.4f}")
    print(f"  Mean distance to center:")
    print(f"    Vanilla:       {dist_vanilla['mean_dist_to_center']:.4f}")
    print(f"    RL-Optimized:  {dist_rl['mean_dist_to_center']:.4f}")
    
    comparison_path = os.path.join(save_dir, "latent_comparison.png")
    plot_latent_comparison(
        z_vanilla, labels_vanilla,
        z_rl, labels_rl,
        save_path=comparison_path
    )
    
    return vanilla_trainer, rl_trainer
