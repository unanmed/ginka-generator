import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import DoubleFCModule

class ConditionEncoder(nn.Module):
    def __init__(self, tag_dim, val_dim, hidden_dim, out_dim):
        super().__init__()
        self.tag_embed = DoubleFCModule(tag_dim, hidden_dim, hidden_dim)
        self.val_embed = DoubleFCModule(val_dim, hidden_dim, hidden_dim)
        self.stage_embed = DoubleFCModule(1, hidden_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4,
                batch_first=True
            ),
            num_layers=4
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, tag, val, stage):
        # tag = self.tag_embed(tag)
        val = self.val_embed(val)
        stage = self.stage_embed(stage)
        feat = torch.stack([val, stage], dim=1)
        feat = self.encoder(feat)
        feat = torch.mean(feat, dim=1)
        feat = self.fusion(feat)
        return feat

class ConditionInjector(nn.Module):
    def __init__(self, cond_dim, out_dim):
        super().__init__()
        self.gamma_layer = nn.Sequential(
            nn.Linear(cond_dim, out_dim)
        )
        self.beta_layer = nn.Sequential(
            nn.Linear(cond_dim, out_dim)
        )

    def forward(self, x, cond):
        gamma = self.gamma_layer(cond).unsqueeze(2).unsqueeze(3)
        beta = self.beta_layer(cond).unsqueeze(2).unsqueeze(3)
        return x * gamma + beta
