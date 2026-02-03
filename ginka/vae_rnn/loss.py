import torch
import torch.nn.functional as F

class VAELoss:
    def __init__(self):
        self.num_classes = 32
    
    def vae_loss(self, logits, target):
        # target: [B, 13, 13]
        target = F.one_hot(target, num_classes=self.num_classes).float().permute(0, 3, 1, 2)
        recon_loss = F.cross_entropy(logits, target)

        return recon_loss
