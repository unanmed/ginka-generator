import torch

def print_memory(device, tag=""):
    if torch.cuda.is_available():
        print(f"{tag} | 当前显存: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB, 最大显存: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
    else:
        print("当前设备不支持 cuda.")