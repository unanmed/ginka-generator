import os
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from .model.model import MinamoModel
from .model.loss import MinamoLoss
from .dataset import MinamoDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("result", exist_ok=True)
os.makedirs("result/minamo_checkpoint", exist_ok=True)

epochs = 150

def collate_fn(batch):
    """动态处理不同尺寸地图的批处理"""
    map1_batch = [item[0] for item in batch]
    map2_batch = [item[1] for item in batch]
    vis_sim = torch.cat([item[2] for item in batch])
    topo_sim = torch.cat([item[3] for item in batch])
    
    # 保持批次内地图尺寸一致（根据问题描述）
    assert all(m.shape == map1_batch[0].shape for m in map1_batch), \
           "对比地图必须尺寸相同"
    
    return (
        torch.stack(map1_batch),  # (B, H, W)
        torch.stack(map2_batch),  # (B, H, W)
        vis_sim,
        topo_sim
    )

def train():
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} to train model.")
    model = MinamoModel(32)
    model.to(device)

    # 准备数据集
    dataset = MinamoDataset("minamo-dataset.json")
    val_dataset = MinamoDataset("minamo-eval.json")
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=True
    )
    
    # 设定优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = MinamoLoss()
    
    # 开始训练
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            # 数据迁移到设备
            map1, map2, vision_simi, topo_simi, graph1, graph2 = batch
            map1 = map1.to(device) # 转为 [B, C, H, W]
            map2 = map2.to(device)
            topo_simi = topo_simi.to(device)
            vision_simi = vision_simi.to(device)
            graph1 = graph1.to(device)
            graph2 = graph2.to(device)
            
            # print(map1.shape, map2.shape)
            
            # 前向传播
            optimizer.zero_grad()
            vision_feat1, topo_feat1 = model(map1, graph1)
            vision_feat2, topo_feat2 = model(map2, graph2)
            
            vision_pred = F.cosine_similarity(vision_feat1, vision_feat2, -1).unsqueeze(-1)
            topo_pred = F.cosine_similarity(topo_feat1, topo_feat2, -1).unsqueeze(-1)
            
            # 计算损失
            loss = criterion(vision_pred, topo_pred, vision_simi, topo_simi)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        ave_loss = total_loss / len(dataloader)
        tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: {epoch + 1} | loss: {ave_loss:.6f} | lr: {(optimizer.param_groups[0]['lr']):.6f}")
        
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
        
        # 每十轮推理一次验证集
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    map1_val, map2_val, vision_simi_val, topo_simi_val, graph1, graph2  = val_batch
                    map1_val = map1_val.to(device)
                    map2_val = map2_val.to(device)
                    vision_simi_val = vision_simi_val.to(device)
                    topo_simi_val = topo_simi_val.to(device)
                    graph1 = graph1.to(device)
                    graph2 = graph2.to(device)
                    
                    vision_feat1, vision_feat2, topo_feat1, topo_feat2 = model(map1_val, map2_val, graph1, graph2)
            
                    vision_pred = F.cosine_similarity(vision_feat1, vision_feat2, -1).unsqueeze(-1)
                    topo_pred = F.cosine_similarity(topo_feat1, topo_feat2, -1).unsqueeze(-1)
                    
                    # 计算损失
                    loss_val = criterion(vision_pred, topo_pred, vision_simi, topo_simi)
                    val_loss += loss_val.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation::loss: {avg_val_loss:.6f}")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, f"result/minamo_checkpoint/{epoch + 1}.pth")
        
    print("Train ended.")
    
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, "result/minamo.pth")

if __name__ == "__main__":
    torch.set_num_threads(2)
    train()
