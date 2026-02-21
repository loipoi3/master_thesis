# RL-Optimized VAE for Latent Space Hole Filling

## Project Overview

This project implements a Variational Autoencoder (VAE) optimized by a Reinforcement Learning (RL) agent to address the "hole" problem in the latent space. The RL agent dynamically adjusts the latent distribution during training to fill low-density regions while maintaining reconstruction quality.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python main.py --mode train --dataset mnist --vae_epochs 5 --cycles 5
```

### Comparison Experiment (Vanilla VAE vs RL-Optimized VAE)

```bash
python main.py --mode compare --dataset mnist --vae_epochs 5 --cycles 5
```

### Visualize from Checkpoint

```bash
python main.py --mode visualize --checkpoint results/final_model.pt
```

## Command Line Arguments

- `--mode`: Running mode (`train`, `compare`, `visualize`)
- `--dataset`: Dataset to use (`mnist`, `fashion_mnist`)
- `--vae_epochs`: Number of VAE epochs per cycle
- `--cycles`: Number of alternating training cycles
- `--batch_size`: Batch size for training
- `--latent_dim`: Latent dimension (use 2 for visualization)
- `--seed`: Random seed
- `--device`: Device to use (`auto`, `cpu`, `cuda`)
- `--checkpoint`: Path to checkpoint to load
- `--save_dir`: Directory to save results

## Project Structure

```
.
├── config.py       # Configuration dataclasses
├── vae.py          # VAE model (Encoder, Decoder, VAE)
├── env.py          # Custom Gymnasium environment
├── train.py        # Training loop with alternating phases
├── visualize.py    # Visualization functions
├── main.py         # Entry point
└── requirements.txt
```

## Architecture

### VAE
- Encoder: MLP with LayerNorm and LeakyReLU
- Decoder: MLP with LayerNorm and LeakyReLU
- Latent dimension: 2 (for visualization)

### RL Agent
- Algorithm: PPO from Stable Baselines3
- Action: delta_mu and delta_log_var shifts
- Reward: Weighted combination of reconstruction quality and density uniformity

### Training
Alternating between:
1. **Phase A**: Train VAE with RL agent inference
2. **Phase B**: Train RL agent with frozen VAE

## Linting

Run linting and type checking:
```bash
ruff check .
mypy . --ignore-missing-imports
```
