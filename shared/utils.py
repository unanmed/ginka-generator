import torch

def random_smooth_onehot(onehot_map, min_main=0.75, max_main=1.0, epsilon=0.25):
    """
    生成随机平滑的 one-hot 编码，使主类别概率不再固定，而是随机波动
    """
    C, H, W = onehot_map.shape
    # 生成主类别的随机概率 (min_main, max_main)
    main_prob = torch.rand(H, W) * (max_main - min_main) + min_main  
    
    # 计算剩余概率并随机分配到其他类别
    noise = torch.rand(C, H, W) * epsilon  # 随机噪声
    noise = noise / noise.sum(dim=1, keepdim=True)  # 归一化到总和为 epsilon
    
    # 计算最终平滑 one-hot 结果
    smooth_onehot = onehot_map * main_prob + (1 - onehot_map) * noise
    return smooth_onehot