import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import VAEEncoder
from .decoder import VAEDecoder

class GinkaVAE(nn.Module):
    def __init__(self, device, tile_classes=32, latent_dim=32):
        super().__init__()
        self.encoder = VAEEncoder(tile_classes, latent_dim)
        self.decoder = VAEDecoder(device)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, target_map: torch.Tensor, use_self_probility=0):
        target = F.one_hot(target_map, num_classes=32).float().permute(0, 3, 1, 2)
        mu, logvar = self.encoder(target)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, target_map, use_self_probility)
        return logits, mu, logvar