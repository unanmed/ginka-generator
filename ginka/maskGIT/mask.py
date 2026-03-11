import random
import torch
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

class MapMask:
    def __init__(self, probs: list[float] = [0.5, 0.5]):
        # 掩码方案
        # 0: 纯随机掩码
        # 1: 分块随机掩码
        self.probs = [sum(probs[0:i+1]) for i in range(len(probs))]
        
    def _sample_mask_ratio(self, alpha=2, beta=2, min_ratio=0.05, max_ratio=1):
        r = np.random.beta(alpha, beta)
        r = min_ratio + (max_ratio - min_ratio) * r
        return r

    def mask(self, h: int, w: int):
        test = random.random()
        mask = None
        if test < self.probs[0]:
            mask = self.mask_random(h, w)
        elif test < self.probs[1]:
            mask = self.block_mask(h, w)
            
        mask = self.random_morphology(mask)
        return mask.reshape(h * w)

    def mask_random(self, h: int, w: int):
        # 纯随机掩码
        ratio = self._sample_mask_ratio()
        total = h * w
        num = int(total * ratio)

        idx = np.random.choice(total, num, replace=False)

        mask = np.zeros(total, dtype=bool)
        mask[idx] = True

        return mask.reshape(h, w)

    def block_mask(self, h: int, w: int, min_block=2, max_block=None):
        # 分块随机掩码
        ratio = self._sample_mask_ratio()
        if max_block is None:
            max_block = min(h, w) // 2

        target = int(h * w * ratio)
        mask = np.zeros((h, w), dtype=bool)

        while mask.sum() < target:

            bw = np.random.randint(min_block, max_block + 1)
            bh = np.random.randint(min_block, max_block + 1)

            x = np.random.randint(0, h - bh + 1)
            y = np.random.randint(0, w - bw + 1)

            mask[x:x + bh, y:y + bw] = True

        return mask

    def random_morphology(self, mask, max_iter=2):
        op = np.random.choice(["none", "dilate", "erode"])

        if op == "none":
            return mask

        it = np.random.randint(1, max_iter + 1)

        if op == "dilate":
            return binary_dilation(mask, iterations=it)

        if op == "erode":
            return binary_erosion(mask, iterations=it)
    