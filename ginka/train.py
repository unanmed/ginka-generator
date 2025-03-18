import os
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from .model.model import GinkaModel
from .model.loss import GinkaLoss
from .dataset import GinkaDataset
from minamo.model.model import MinamoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("result", exist_ok=True)

epochs = 70

def update_tau(epoch):
    start_tau = 1.0
    min_tau = 0.1
    decay_rate = 0.95
    return max(min_tau, start_tau * (decay_rate ** epoch))

def train():
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} to train model.")
    model = GinkaModel()
    model.to(device)
    minamo = MinamoModel(32)
    minamo.to(device)
    minamo.eval()

    # 准备数据集
    dataset = GinkaDataset("dataset.json", minamo)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )
    
    # 设定优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = GinkaLoss(minamo)
    
    # 开始训练
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        model.softmax.tau = update_tau(epoch)
        
        for batch in dataloader:
            # 数据迁移到设备
            target = batch["target"].to(device)
            feat_vec = batch["feat_vec"].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output, output_softmax = model(feat_vec)
            
            # 计算损失
            loss = criterion(output, output_softmax, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: {epoch} | loss: {total_loss:.6f} | lr: {(optimizer.param_groups[0]['lr']):.6f}")
        
        # 学习率调整
        scheduler.step()
        
    print("Train ended.")
    
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, f"result/ginka.pth")

if __name__ == "__main__":
    torch.set_num_threads(8)
    train()
