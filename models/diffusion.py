import torch
import torch.nn as nn
from typing import Optional

class AdaptiveNoiseScheduler:
    def __init__(self, num_timesteps: int, min_beta: float = 1e-4, max_beta: float = 0.02):
        self.num_timesteps = num_timesteps
        self.min_beta = min_beta
        self.max_beta = max_beta
        
    def get_schedule(self, uncertainty: torch.Tensor) -> torch.Tensor:
        base_schedule = torch.linspace(self.min_beta, self.max_beta, self.num_timesteps)
        uncertainty_scale = uncertainty.mean().clamp(0.5, 2.0)
        return base_schedule * uncertainty_scale

class ActionDiffusion(nn.Module):
    def __init__(
        self,
        action_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        num_timesteps: int = 20
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.scheduler = AdaptiveNoiseScheduler(num_timesteps)
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*4,
                dropout=dropout_rate,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.final = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        t: int,
        uncertainty: Optional[torch.Tensor] = None,
        temp: float = 0.5
    ) -> torch.Tensor:
        betas = self.scheduler.get_schedule(uncertainty) if uncertainty is not None else None
        
        t_emb = self.time_embed(torch.tensor([t/self.num_timesteps]).to(x.device))
        
        h = self.action_proj(x)
        
        h = h + t_emb
        
        for layer in self.layers:
            h = layer(h)
            
        noise = self.final(h)
        
        alpha = (1 - betas[t]) if betas is not None else 0.99
        noise_scale = temp * torch.sqrt(1 - alpha)
        
        return x + noise * noise_scale