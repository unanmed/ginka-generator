import torch

class VAEScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(
        self, optimizer, mode="min", factor=0.1, patience=10, threshold=0.0001,
        threshold_mode="rel", cooldown=0, min_lr=0, eps=1e-8, verbose="deprecated",
        max_lr=1e-2, increase_factor=2, start_prob=0
    ):
        super().__init__(
            optimizer, mode, factor, patience, threshold,
            threshold_mode, cooldown, min_lr, eps, verbose
        )
        self.max_lr = max_lr
        self.increase_factor = increase_factor
        self.last_prob = start_prob
        
        if isinstance(max_lr, (list, tuple)):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} max_lrs, got {len(max_lr)}"
                )
            self.default_max_lr = None
            self.max_lrs = list(max_lr)
        else:
            self.default_max_lr = max_lr
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
        
    def step(self, metrics, prob: float, epoch=None):
        if prob > self.last_prob:
            self.best = metrics
            self.num_bad_epochs = 0
            self.last_prob = prob
            self._increase_lr()
            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        else:
            return super().step(metrics, epoch)
    
    def _increase_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = min(old_lr * self.increase_factor, self.max_lrs[i])
            if new_lr - old_lr > self.eps:
                param_group["lr"] = new_lr
