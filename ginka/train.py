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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("result", exist_ok=True)

epochs = 100

def collate_fn(batch):
    # 动态填充噪声到最大尺寸
    max_h = max([b["noise"].shape[0] for b in batch])
    max_w = max([b["noise"].shape[1] for b in batch])
    
    padded_batch = {}
    for key in ["noise", "target"]:
        padded = []
        for b in batch:
            tensor = b[key]
            pad_h = max_h - tensor.shape[0]
            pad_w = max_w - tensor.shape[1]
            padded.append(F.pad(tensor, (0, pad_w, 0, pad_h), value=-100 if key=="target" else 0))
        padded_batch[key] = torch.stack(padded)
    
    # 其他字段直接堆叠
    for key in ["input_ids", "attention_mask", "map_size"]:
        padded_batch[key] = torch.stack([b[key] for b in batch])
    
    return padded_batch

def train():
    print(f"Using {"cuda" if torch.cuda.is_available() else "cpu"} to train model.")
    model = GinkaModel()
    model.to(device)

    # 准备数据集
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
    dataset = GinkaDataset("F:/github-ai/ginka-generator/dataset.json", tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 设定优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = GinkaLoss()
    
    # 开始训练
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        
        # 温度退火
        model.gumbel.tau = max(0.1, 1.0 - 0.9 * epoch / epochs)
        
        for batch in dataloader:
            # 数据迁移到设备
            noise = batch["noise"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            map_size = batch["map_size"].to(device)
            target = batch["target"].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(noise, input_ids, attention_mask, map_size)
            
            print(torch.argmax(torch.softmax(outputs, dim=1), dim=1))
            # print(sampled[0, :, :, 1])
            
            # 构建拓扑图
            # with torch.no_grad():
            #     pred_graphs = build_topology_graph(outputs.argmax(1))
            #     ref_graphs = build_topology_graph(target)
            
            # 计算损失
            loss = criterion(
                outputs,  # 调整为 [BS, C, H, W]
                target
            )
            
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
