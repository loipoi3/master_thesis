import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from config import VAEConfig


class Encoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        layers = []
        prev_dim = config.input_dim
        for hidden_dim in config.encoder_hidden:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            prev_dim = hidden_dim
        
        self.encoder_net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, config.latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, config.latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        log_var = torch.clamp(log_var, min=-10, max=10)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        layers = []
        prev_dim = config.latent_dim
        for hidden_dim in config.decoder_hidden:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, config.input_dim))
        self.decoder_net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(z)


class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(
        self, 
        x: torch.Tensor, 
        delta_mu: Optional[torch.Tensor] = None,
        delta_log_var: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        
        mu_shifted = mu
        log_var_shifted = log_var
        
        if delta_mu is not None:
            mu_shifted = mu + delta_mu
        if delta_log_var is not None:
            log_var_shifted = log_var + delta_log_var
            log_var_shifted = torch.clamp(log_var_shifted, min=-10, max=10)
        
        z = self.reparameterize(mu_shifted, log_var_shifted)
        x_recon = self.decode(z)
        
        return x_recon, mu_shifted, log_var_shifted, z
    
    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kl_loss = kl_loss.mean()
        
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss
    
    def sample(self, num_samples: int, device: str) -> torch.Tensor:
        z = torch.randn(num_samples, self.config.latent_dim, device=device)
        return self.decode(z)
