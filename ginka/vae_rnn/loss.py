import torch
import torch.nn.functional as F

class VAELoss:
    def __init__(self):
        self.num_classes = 32
    
    def vae_loss(self, logits, target, mu, logvar, beta=0.1):
        # target: [B, 13, 13]
        target = F.one_hot(target, num_classes=self.num_classes).float().permute(0, 3, 1, 2)
        recon_loss = F.cross_entropy(logits, target)

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        return recon_loss + beta * kl_loss, recon_loss, kl_loss
