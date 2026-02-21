import os
import argparse
import torch
import random
import numpy as np

from config import VAEConfig, RLConfig, TrainingConfig
from train import Trainer
from visualize import (
    plot_latent_space,
    plot_latent_comparison,
    plot_training_curves,
    plot_reconstruction,
    plot_samples_from_prior,
    compute_hole_metric,
    run_comparison_experiment
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="RL-Optimized VAE for Latent Space Hole Filling")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "compare", "visualize"],
                        help="Running mode")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"],
                        help="Dataset to use")
    parser.add_argument("--vae_epochs", type=int, default=5,
                        help="Number of VAE epochs per cycle")
    parser.add_argument("--cycles", type=int, default=5,
                        help="Number of alternating training cycles")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--latent_dim", type=int, default=2,
                        help="Latent dimension (use 2 for visualization)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to load")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    set_seed(args.seed)
    
    vae_config = VAEConfig(
        latent_dim=args.latent_dim,
        encoder_hidden=[512, 256],
        decoder_hidden=[256, 512],
        input_dim=784,
        learning_rate=1e-3,
        beta_kl=1.0
    )
    
    rl_config = RLConfig(
        reward_weights=(1.0, 2.0),
        n_steps_per_update=64,
        n_epochs=5,
        batch_size=32,
        learning_rate=3e-4
    )
    
    training_config = TrainingConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        vae_epochs=args.vae_epochs,
        rl_episodes_per_phase=5,
        alternating_cycles=args.cycles,
        device=device,
        seed=args.seed,
        save_dir=os.path.join(args.save_dir, "checkpoints")
    )
    
    print("=" * 60)
    print("RL-Optimized VAE for Latent Space Hole Filling")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"VAE epochs per cycle: {args.vae_epochs}")
    print(f"Training cycles: {args.cycles}")
    print("=" * 60)
    
    if args.mode == "train":
        trainer = Trainer(vae_config, rl_config, training_config)
        trainer.setup_data()
        
        if args.checkpoint:
            print(f"Loading checkpoint from {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)
        
        history = trainer.train()
        
        eval_metrics = trainer.evaluate()
        print(f"\nTest Metrics:")
        print(f"  Loss: {eval_metrics['test_loss']:.4f}")
        print(f"  Recon: {eval_metrics['test_recon']:.4f}")
        print(f"  KL: {eval_metrics['test_kl']:.4f}")
        
        z_vanilla, labels = trainer.get_latent_representations(use_rl=False)
        z_rl, _ = trainer.get_latent_representations(use_rl=True)
        
        hole_vanilla = compute_hole_metric(z_vanilla)
        hole_rl = compute_hole_metric(z_rl)
        
        print(f"\nHole Metric (lower is better):")
        print(f"  Without RL shift: {hole_vanilla:.4f}")
        print(f"  With RL shift: {hole_rl:.4f}")
        print(f"  Improvement: {(hole_vanilla - hole_rl) / hole_vanilla * 100:.2f}%")
        
        os.makedirs(args.save_dir, exist_ok=True)
        
        plot_latent_comparison(
            z_vanilla, labels, z_rl, labels,
            save_path=os.path.join(args.save_dir, "latent_comparison.png")
        )
        
        plot_training_curves(
            history["vae_losses"],
            history.get("rl_rewards", []),
            save_path=os.path.join(args.save_dir, "training_curves.png")
        )
        
        plot_reconstruction(
            trainer.vae,
            trainer.test_loader,
            device=device,
            save_path=os.path.join(args.save_dir, "reconstructions.png")
        )
        
        plot_samples_from_prior(
            trainer.vae,
            device=device,
            save_path=os.path.join(args.save_dir, "prior_samples.png")
        )
        
        final_path = os.path.join(args.save_dir, "final_model.pt")
        trainer.save_final(final_path)
        print(f"\nModel saved to {final_path}")
    
    elif args.mode == "compare":
        vanilla_trainer, rl_trainer = run_comparison_experiment(
            vae_config, rl_config, training_config,
            save_dir=args.save_dir
        )
        
        vanilla_trainer.save_final(
            os.path.join(args.save_dir, "vanilla_vae.pt")
        )
        rl_trainer.save_final(
            os.path.join(args.save_dir, "rl_optimized_vae.pt")
        )
    
    elif args.mode == "visualize":
        if not args.checkpoint:
            print("Error: --checkpoint required for visualize mode")
            return
        
        trainer = Trainer(vae_config, rl_config, training_config)
        trainer.setup_data()
        trainer.load_checkpoint(args.checkpoint)
        
        z, labels = trainer.get_latent_representations(use_rl=True)
        plot_latent_space(z, labels, title="RL-Optimized VAE Latent Space")
        
        plot_reconstruction(
            trainer.vae,
            trainer.test_loader,
            device=device
        )


if __name__ == "__main__":
    main()
