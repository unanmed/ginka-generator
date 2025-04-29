import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionEncoder(nn.Module):
    def __init__(self, tag_dim, val_dim, hidden_dim, out_dim):
        super().__init__()
        self.tag_embed = nn.Linear(tag_dim, hidden_dim)
        self.val_embed = nn.Linear(val_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim*2),
            nn.ELU(),
            
            nn.Linear(hidden_dim*2, hidden_dim*4),
            nn.LayerNorm(hidden_dim*4),
            nn.ELU(),
            
            nn.Linear(hidden_dim*4, out_dim)
        )
        
    def forward(self, tag, val):
        tag = self.tag_embed(tag)
        val = self.val_embed(val)
        feat = torch.cat([tag, val], dim=1)
        feat = self.fusion(feat)
        return feat

class ConditionInjector(nn.Module):
    def __init__(self, cond_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, cond_dim*2),
            nn.LayerNorm(cond_dim*2),
            nn.ELU(),
            
            nn.Linear(cond_dim*2, out_dim)
        )
        
    def forward(self, x, cond):
        cond = self.fc(cond)
        B, D = cond.shape
        cond = cond.view(B, D, 1, 1)
        return x + cond
