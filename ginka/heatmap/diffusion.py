import math
import torch

class Diffusion:
    def __init__(self, device, T=100, noise_scale=0.5):
        self.T = T
        self.device = device
        self.noise_scale = noise_scale

        # cosine schedule（推荐）
        steps = torch.arange(T + 1, dtype=torch.float32)
        s = 0.1
        f = torch.cos(((steps / (T + 1)) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = f / f[0]

        self.alpha_bar = alpha_bar.to(device)
        self.sqrt_ab = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_ab = torch.sqrt(1 - self.alpha_bar)

    def q_sample(self, x0, t, noise):
        """
        前向加噪：x_t = sqrt(αbar_t) * x0 + sqrt(1-αbar_t) * noise_scale * ε
        noise_scale 降低噪声功率，使信号不被淹没
        """
        return (
            self.sqrt_ab[t][:, None, None, None] * x0
            + self.sqrt_one_minus_ab[t][:, None, None, None] * noise * self.noise_scale
        )
        
    def sample(self, model, cond: torch.Tensor, steps=20):
        """
        DDIM 风格逆向采样，模型预测 x_0
        x_{t-1} = sqrt(αbar_{t-1}) * x0_pred
                + sqrt(1-αbar_{t-1}) / sqrt(1-αbar_t) * (x_t - sqrt(αbar_t) * x0_pred)
        """
        B = cond.shape[0]
        # 初始噪声与前向过程保持一致的噪声功率
        x = torch.randn_like(cond).to(cond.device) * self.noise_scale

        step_size = self.T // steps

        for i in reversed(range(0, self.T, step_size)):
            t = torch.full((B,), i, device=cond.device)

            # 模型直接预测 x_0
            x0_pred = model(x, cond, t)

            alpha = self.alpha_bar[i]
            alpha_prev = self.alpha_bar[max(i - step_size, 0)]

            # DDIM x0-prediction 更新
            direction = (
                torch.sqrt(1 - alpha_prev) / torch.sqrt(1 - alpha)
            ) * (x - torch.sqrt(alpha) * x0_pred)

            x = torch.sqrt(alpha_prev) * x0_pred + direction

        return x

if __name__ == '__main__':
    diff = Diffusion("cpu")
    print(diff.sqrt_one_minus_ab)
    print(diff.sqrt_ab)
