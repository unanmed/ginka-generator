import torch
import torch.nn as nn

class GinkaInput(nn.Module):
    def __init__(self, feat_dim=1024, out_ch=1, size=(32, 32)):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, size[0] * size[1] * out_ch),
            nn.Unflatten(1, (out_ch, *size))
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x
    
class FeatureEncoder(nn.Module):
    def __init__(self, feat_dim, size, mid_ch, out_ch):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(feat_dim, mid_ch * size * size),
            nn.Unflatten(1, (mid_ch, size, size)),
            nn.Conv2d(mid_ch, out_ch, 1)
        )
        
    def forward(self, x):
        x = self.encode(x)
        return x

class GinkaFeatureInput(nn.Module):
    def __init__(self, feat_dim=1024, mid_ch=1, out_ch=64):
        super().__init__()
        self.encode1 = FeatureEncoder(feat_dim, 32, mid_ch, out_ch)
        self.encode2 = FeatureEncoder(feat_dim, 16, mid_ch * 2, out_ch * 2)
        self.encode3 = FeatureEncoder(feat_dim, 8, mid_ch * 4, out_ch * 4)
        self.encode4 = FeatureEncoder(feat_dim, 4, mid_ch * 8, out_ch * 8)
        self.encode5 = FeatureEncoder(feat_dim, 2, mid_ch * 16, out_ch * 16)
        
    def forward(self, x):
        x1 = self.encode1(x)
        x2 = self.encode2(x)
        x3 = self.encode3(x)
        x4 = self.encode4(x)
        x5 = self.encode5(x)
        return x1, x2, x3, x4, x5
