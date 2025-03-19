import os
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model.model import GinkaModel
from .model.loss import GinkaLoss
from .dataset import GinkaDataset
from minamo.model.model import MinamoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("result", exist_ok=True)
os.makedirs("result/ginka_checkpoint", exist_ok=True)

epochs = 150

def update_tau(epoch):
    start_tau = 1.0
    min_tau = 0.1
    decay_rate = 0.95
    return max(min_tau, start_tau * (decay_rate ** epoch))

# 在生成器输出后添加梯度检查钩子
def grad_hook(module, grad_input, grad_output):
    print(f"Generator output grad norm: {grad_output[0].norm().item()}")

def train():
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} to train model.")
    model = GinkaModel()
    model.to(device)
    minamo = MinamoModel(32)
    minamo.load_state_dict(torch.load("result/minamo.pth", map_location=device)["model_state"])
    minamo.to(device)
    minamo.eval()
    
    # for param in minamo.parameters():
    #     param.requires_grad = False

    # 准备数据集
    dataset = GinkaDataset("ginka-dataset.json", device, minamo)
    dataset_val = GinkaDataset("ginka-eval.json", device, minamo)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=32,
        shuffle=True
    )
    
    # 设定优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = GinkaLoss(minamo, weight=[1, 0, 0, 0, 0, 0, 0, 0])
    
    model.register_full_backward_hook(grad_hook)
    # converter.register_full_backward_hook(grad_hook)
    criterion.register_full_backward_hook(grad_hook)
    
    # 开始训练
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            # 数据迁移到设备
            target = batch["target"].to(device)
            target_vision_feat = batch["target_vision_feat"].to(device)
            target_topo_feat = batch["target_topo_feat"].to(device)
            feat_vec = torch.cat([target_vision_feat, target_topo_feat], dim=-1).to(device)
            # 前向传播
            optimizer.zero_grad()
            output, output_softmax = model(feat_vec)
            
            # 计算损失
            loss = criterion(output, output_softmax, target, target_vision_feat, target_topo_feat)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: {epoch + 1} | loss: {avg_loss:.6f} | lr: {(optimizer.param_groups[0]['lr']):.6f}")
        
        # total_norm = 0
        # for p in model.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.detach().data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # tqdm.write(f"Gradient Norm: {total_norm:.4f}")  # 正常应保持在1~100之间
        
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: grad_mean={param.grad.abs().mean():.3e}, max={param.grad.abs().max():.3e}")
        
        # 学习率调整
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            loss_val = 0
            model.eval()
            with torch.no_grad():
                for batch in dataloader_val:
                    # 数据迁移到设备
                    target = batch["target"].to(device)
                    target_vision_feat = batch["target_vision_feat"].to(device)
                    target_topo_feat = batch["target_topo_feat"].to(device)
                    feat_vec = torch.cat([target_vision_feat, target_topo_feat], dim=-1).to(device)
                    
                    # 前向传播
                    output, output_softmax = model(feat_vec)
                    print(output_softmax[0])
                    
                    # 计算损失
                    loss = criterion(output, output_softmax, target, target_vision_feat, target_topo_feat)
                    loss_val += loss.item()
            
            avg_val_loss = loss_val / len(dataloader_val)
            tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation::loss: {avg_val_loss:.6f}")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, f"result/ginka_checkpoint/{epoch + 1}.pth")
                
        
    print("Train ended.")
    
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, f"result/ginka.pth")

if __name__ == "__main__":
    torch.set_num_threads(8)
    train()
