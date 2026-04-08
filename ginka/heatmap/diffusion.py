import math
import torch

class Diffusion:
    def __init__(self, device, T=100, min_beta=0.0001, max_beta=0.01):
        self.T = T
        self.device = device

        betas = torch.linspace(min_beta, max_beta, T).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = T
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def q_sample(self, x0, t, noise):
        """
        前向加噪
        """
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        res = noise * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x0
        return res
        
    def sample(self, model, cond: torch.Tensor):
        x = torch.randn_like(cond).to(self.device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, cond, model)
        return x

    def sample_backward_step(self, x_t, t, cond, model):
        B = x_t.shape[0]
        t_tensor = torch.tensor([t] * B, dtype=torch.long).to(self.device)
        eps = model(x_t, cond, t_tensor)

        if t == 0:
            noise = 0
        else:
            var = (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        mean = (x_t -
                (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t

if __name__ == '__main__':
    diff = Diffusion("cpu")
    print(diff.alphas)
    print(diff.alpha_bars)
