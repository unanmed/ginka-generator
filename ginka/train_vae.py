import argparse
import os
import sys
import random
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from .vae_rnn.vae import GinkaVAE
from .vae_rnn.loss import VAELoss
from .dataset import GinkaRNNDataset
from shared.image import matrix_to_image_cv

# 手工标注标签定义（暂时不用）：
# 0. 蓝海, 1. 红海, 2: 室内, 3. 野外, 4. 左右对称, 5. 上下对称, 6. 伪对称, 7. 咸鱼层,
# 8. 剧情层, 9. 水层, 10. 爽塔, 11. Boss层, 12. 纯Boss层, 13. 多房间, 14. 多走廊, 15. 道具风
# 16. 区域入口, 17. 区域连接, 18. 有机关门, 19. 道具层, 20. 斜向对称, 21. 左右通道, 22. 上下通道, 23. 多机关门
# 24. 中心对称, 25. 部分对称, 26. 鱼骨

# 自动标注标签定义（暂时不用）：
# 0. 左右对称, 1. 上下对称, 2. 中心对称, 3. 斜向对称, 4. 伪对称, 5. 多房间, 6. 多走廊
# 32. 平面塔, 33. 转换塔, 34. 道具塔

# 标量值定义：
# 0. 整体密度，非空白图块/地图面积，空白图块还包括装饰图块
# 1. 墙体密度，墙壁/地图面积
# 2. 装饰密度，装饰数量/地图面积
# 3. 门密度，门数量/地图面积
# 4. 怪物密度，怪物数量/地图面积
# 5. 资源密度，资源数量/地图面积
# 6. 宝石密度，宝石数量/地图面积
# 7. 血瓶密度，血瓶数量/地图面积
# 8. 钥匙密度，钥匙数量/地图面积
# 9. 道具密度，道具数量/地图面积
# 10. 入口数量
# 11. 机关门数量
# 12. 咸鱼门数量（多层咸鱼门只算一个）

# 图块定义：
# 0. 空地, 1. 墙壁, 2. 装饰（用于野外装饰，视为空地）, 
# 3. 黄门, 4. 蓝门, 5. 红门, 6. 机关门, 其余种类的门如绿门都视为红门
# 7-9. 黄蓝红门钥匙，机关门不使用钥匙开启
# 10-12. 三种等级的红宝石
# 13-15. 三种等级的蓝宝石
# 16-18. 三种等级的绿宝石
# 19-22. 四种等级的血瓶
# 23-25. 三种等级的道具
# 26-28. 三种等级的怪物
# 29. 入口，不区分楼梯和箭头

BATCH_SIZE = 128
LATENT_DIM = 48
KL_BETA = 0.01

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.makedirs("result", exist_ok=True)
os.makedirs("result/vae", exist_ok=True)
os.makedirs("result/ginka_vae_img", exist_ok=True)

disable_tqdm = not sys.stdout.isatty()

def gt_prob(epoch: int, max_epoch: int) -> float:
    progress = epoch / max_epoch
    if progress < 0.2:
        return 1
    elif progress < 0.8:
        return 1 - (progress - 0.2) / 0.6
    else:
        return 0

def parse_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--state_ginka", type=str, default="result/vae/ginka-100.pth")
    parser.add_argument("--train", type=str, default="ginka-dataset.json")
    parser.add_argument("--validate", type=str, default="ginka-eval.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint", type=int, default=5)
    parser.add_argument("--load_optim", type=bool, default=True)
    args = parser.parse_args()
    return args

def train():
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} to train model.")
    
    args = parse_arguments()
    
    vae = GinkaVAE(device, latent_dim=LATENT_DIM).to(device)
    
    dataset = GinkaRNNDataset(args.train, device)
    dataset_val = GinkaRNNDataset(args.validate, device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE // 64, shuffle=True)
    
    optimizer_ginka = optim.AdamW(vae.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler_ginka = optim.lr_scheduler.CosineAnnealingLR(optimizer_ginka, T_max=800, eta_min=1e-6)

    criterion = VAELoss()

    # 用于生成图片
    tile_dict = dict()
    for file in os.listdir('tiles'):
        name = os.path.splitext(file)[0]
        tile_dict[name] = cv2.imread(f"tiles/{file}", cv2.IMREAD_UNCHANGED)
        
    if args.resume:
        data_ginka = torch.load(args.state_ginka, map_location=device)

        vae.load_state_dict(data_ginka["model_state"], strict=False)
        
        if args.load_optim:
            if data_ginka.get("optim_state") is not None:
                optimizer_ginka.load_state_dict(data_ginka["optim_state"])
            
        print("Train from loaded state.")
        
    for epoch in tqdm(range(args.epochs), desc="VAE Training", disable=disable_tqdm):
        loss_total = torch.Tensor([0]).to(device)
        
        for batch in tqdm(dataloader, leave=False, desc="Epoch Progress", disable=disable_tqdm):
            target_map = batch["target_map"].to(device)
            
            fake_logits, mu, logvar = vae(target_map, 1 - gt_prob(epoch, args.epochs))
            
            loss = criterion.vae_loss(fake_logits, target_map, mu, logvar, KL_BETA)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer_ginka.step()
            loss_total += loss.detach()
                
        avg_loss = loss_total.item() / len(dataloader)
        tqdm.write(
            f"[Epoch {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " +
            f"E: {epoch + 1} | Loss: {avg_loss:.6f} | " +
            f"LR: {optimizer_ginka.param_groups[0]['lr']:.6f}"
        )
        
        scheduler_ginka.step()

        # 每若干轮输出一次图片，并保存检查点
        if (epoch + 1) % args.checkpoint == 0:
            # 保存检查点
            torch.save({
                "model_state": vae.state_dict(),
                "optim_state": optimizer_ginka.state_dict(),
            }, f"result/rnn/ginka-{epoch + 1}.pth")
        
            val_loss_total = torch.Tensor([0]).to(device)
            reco_loss_total = torch.Tensor([0]).to(device)
            with torch.no_grad():
                idx = 0
                gap = 5
                color = (255, 255, 255)  # 白色
                vline = np.full((416, gap, 3), color, dtype=np.uint8)  # 垂直分割线
                # 地图重建展示
                for batch in tqdm(dataloader_val, desc="Validating generator.", leave=False, disable=disable_tqdm):
                    target_map = batch["target_map"].to(device)
                    
                    fake_logits, z = vae(target_map, 1 - gt_prob(epoch, args.epochs))

                    loss = criterion.vae_loss(fake_logits, target_map, z, KL_BETA)
                    val_loss_total += loss.detach()
                    
                    fake_map = torch.argmax(fake_logits, dim=1).cpu().numpy()
                    fake_img = matrix_to_image_cv(fake_map[0], tile_dict)
                    real_map = target_map.cpu().numpy()
                    real_img = matrix_to_image_cv(real_map[0], tile_dict)
                    img = np.block([[real_img], [vline], [fake_img]])
                    cv2.imwrite(f"result/ginka_vae_img/{idx}.png", img)
                    
                    idx += 1
                
                # 随机采样
                for i in range(0, 8):
                    z = torch.randn(1, LATENT_DIM).to(device)
                    
                    fake_logits = vae.decoder(z, torch.zeros(1, 13, 13).to(device), 1)
                    fake_map = torch.argmax(fake_logits, dim=1).cpu().numpy()
                    fake_img = matrix_to_image_cv(fake_map[0], tile_dict)
                    
                    cv2.imwrite(f"result/ginka_vae_img/{i}_rand.png", fake_img)
                
                # 插值
                val_length = len(dataset_val.data)
                index1 = random.randint(0, val_length - 1)
                index2 = random.randint(0, val_length - 1)
                map1 = torch.LongTensor(dataset_val.data[index1]["map"]).to(device).reshape(1, 13, 13)
                map2 = torch.LongTensor(dataset_val.data[index2]["map"]).to(device).reshape(1, 13, 13)
                z1 = vae.encoder(map1)
                z2 = vae.encoder(map2)
                real_img1 = matrix_to_image_cv(map1[0].cpu().numpy(), tile_dict)
                real_img2 = matrix_to_image_cv(map2[0].cpu().numpy(), tile_dict)
                i = 0
                for t in torch.linspace(0, 1, 8):
                    z = z1 * (1 - t / 8) + z2 * t / 8
                    fake_logits = vae.decoder(z, torch.zeros(1, 13, 13).to(device), 1)
                    fake_map = torch.argmax(fake_logits, dim=1).cpu().numpy()
                    fake_img = matrix_to_image_cv(fake_map[0], tile_dict)
                    img = np.block([[real_img1], [vline], [fake_img], [vline], [real_img2]])
                    
                    cv2.imwrite(f"result/ginka_vae_img/{i}_linspace.png", img)
                    i += 1
                    
            avg_loss_val = val_loss_total.item() / len(dataloader_val)
            tqdm.write(
                f"[Validate {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] E: {epoch + 1} | " +
                f"Loss: {avg_loss_val:.6f}"
            )
            
    print("Train ended.")
    torch.save({
        "model_state": vae.state_dict(),
    }, f"result/ginka_rnn.pth")
    

if __name__ == "__main__":
    torch.set_num_threads(4)
    train()
