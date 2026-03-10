import torch
import torch.nn.functional as F

class VAELoss:
    def __init__(self):
        self.num_classes = 32
    
    def vae_loss(self, logits, target, mu, logvar, beta=0.1):
        # logits: [B, 169, 16]
        # target: [B, 169]
        B, L = target.shape
        end_token = torch.tensor([15], dtype=torch.long).to(logits.device).repeat(B, 1)
        target = torch.cat([target, end_token], dim=1)
        recon_loss = F.cross_entropy(logits.permute(0, 2, 1), target)

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        return recon_loss + beta * kl_loss, recon_loss, kl_loss
