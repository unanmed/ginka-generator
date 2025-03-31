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
from shared.args import parse_arguments

BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("result", exist_ok=True)
os.makedirs("result/ginka_checkpoint", exist_ok=True)

def train():
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} to train model.")
    
    args = parse_arguments("result/ginka.pth", "ginka-dataset.json", 'ginka-eval.json')
    
    model = GinkaModel()
    model.to(device)
    minamo = MinamoModel(32)
    minamo.load_state_dict(torch.load("result/minamo.pth", map_location=device)["model_state"])
    minamo.to(device)
    minamo.eval()

    # 准备数据集
    dataset = GinkaDataset(args.train, device, minamo)
    dataset_val = GinkaDataset(args.validate, device, minamo)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # 设定优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = GinkaLoss(minamo)
    
    if args.resume:
        data = torch.load(args.from_state, map_location=device)
        model.load_state_dict(data["model_state"], strict=False)
        if args.load_optim:
            optimizer.load_state_dict(data["optimizer_state"])
        print("Train from loaded state.")
        
    else:
        # 从头开始训练的话，初始时先把 minamo 损失值权重改为 0
        criterion.weight[0] = 0.0
    
    # 开始训练
    for epoch in tqdm(range(args.epochs)):
        model.train()
        total_loss = 0
        
        # 从头开始训练的，在第 10 个 epoch 将 minamo 损失值权重改回来
        if not args.resume and epoch == 10:
            criterion.weight[0] = 0.5
        
        for batch in dataloader:
            # 数据迁移到设备
            target = batch["target"].to(device)
            target_vision_feat = batch["target_vision_feat"].to(device)
            target_topo_feat = batch["target_topo_feat"].to(device)
            feat_vec = torch.cat([target_vision_feat, target_topo_feat], dim=-1).to(device).squeeze(1)
            # 前向传播
            optimizer.zero_grad()
            noise = torch.randn((target.shape[0], 1, 32, 32)).to(device)
            _, output_softmax = model(noise, feat_vec)
            
            # 计算损失
            scaled_losses, losses = criterion(output_softmax, target, target_vision_feat, target_topo_feat)
            
            # 反向传播
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
            
        avg_loss = total_loss / len(dataloader)
        tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: {epoch + 1} | loss: {avg_loss:.6f} | lr: {(optimizer.param_groups[0]['lr']):.6f}")
        
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
                    feat_vec = torch.cat([target_vision_feat, target_topo_feat], dim=-1).to(device).squeeze(1)
                    
                    # 前向传播
                    noise = torch.randn((target.shape[0], 1, 32, 32)).to(device)
                    output, output_softmax = model(noise, feat_vec)
                    print(torch.argmax(output, dim=1)[0])
                    
                    # 计算损失
                    scaled_losses, losses = criterion(output_softmax, target, target_vision_feat, target_topo_feat)
                    loss_val += losses.item()
            
            avg_val_loss = loss_val / len(dataloader_val)
            tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation::loss: {avg_val_loss:.6f}")
            torch.save({
                "model_state": model.state_dict(),
                # "optimizer_state": optimizer.state_dict(),
            }, f"result/ginka_checkpoint/{epoch + 1}.pth")
                
        
    print("Train ended.")
    
    torch.save({
        "model_state": model.state_dict(),
        # "optimizer_state": optimizer.state_dict(),
    }, f"result/ginka.pth")

if __name__ == "__main__":
    torch.set_num_threads(4)
    train()
