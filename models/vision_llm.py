import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, ViTModel
from typing import Tuple, List, Optional
from .diffusion import ActionDiffusion

class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, x.shape[-1])  # (batch, seq, dim)
        context = context.view(batch_size, -1, context.shape[-1])
        
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        q = q.view(batch_size, -1, self.heads, q.shape[-1] // self.heads).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, k.shape[-1] // self.heads).transpose(1, 2) 
        v = v.view(batch_size, -1, self.heads, v.shape[-1] // self.heads).transpose(1, 2)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(batch_size, -1, out.shape[-1] * self.heads)
        return self.to_out(out)
    
class ProgressiveVisionEncoder(nn.Module):
    def __init__(self, base_model: nn.Module, stages: List[int] = [224, 384, 512]):
        super().__init__()
        self.stages = stages
        self.base_model = base_model
        self.adaptation = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.GELU()
            ) for _ in range(len(stages)-1)
        ])
        
    def forward(self, x: torch.Tensor, stage_idx: int) -> torch.Tensor:
        features = self.base_model(x).last_hidden_state
        for i in range(stage_idx):
            features = self.adaptation[i](features)
        return features

class RoboticVisionLLM(nn.Module):
    def __init__(
        self,
        llm_path: str = "llm",
        action_dim: int = 17,
        vision_dim: int = 768,
        llm_dim: int = 2048,
        dropout_rate: float = 0.1,
        num_ensembles: int = 3,
        num_diffusion_steps: int = 20
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        base_vision = ViTModel.from_pretrained('vit')
        self.vision_encoder = ProgressiveVisionEncoder(base_vision)
        
        self.cross_attn = CrossAttention(llm_dim)
        
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path)
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, llm_dim),
            nn.Dropout(dropout_rate)
        )
        
        self.action_diffusion = ActionDiffusion(
            action_dim=action_dim,
            hidden_dim=512,
            num_timesteps=num_diffusion_steps
        )
        
        self.contrastive_head = nn.Linear(llm_dim, 128)
        self.num_diffusion_steps = num_diffusion_steps
        self.action_dimension = action_dim
        
    def forward(
        self, 
        images: torch.Tensor,
        stage_idx: int = 0,
        temp: float = 1.0,
        num_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = images.shape[0]
        vision_features = self.vision_encoder(images, stage_idx)
        vision_features = vision_features.view(batch_size, -1, vision_features.shape[-1])
        
        vision_embeds = self.vision_proj(vision_features)
    
        llm_features = self.llm(
            inputs_embeds=vision_embeds,
            output_hidden_states=True
        ).hidden_states[-1]
        
        attended = self.cross_attn(llm_features, vision_embeds)
        
        actions = torch.randn(batch_size, self.action_dimension).to(self.device)
        uncertainty = torch.zeros(batch_size, self.action_dimension).to(self.device)
        
        steps = num_steps or self.num_diffusion_steps
        
        for t in reversed(range(steps)):
            actions = self.action_diffusion(
                actions,
                t,
                uncertainty=uncertainty,
                temp=temp
            )
            uncertainty = torch.abs(actions - actions.mean(dim=0))
        
        contrastive = self.contrastive_head(attended.mean(dim=1))
        
        return actions, uncertainty, F.normalize(contrastive, dim=-1)