import argparse
import os
import sys
from datetime import datetime
import torch
import torch.optim as optim
import cv2
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from .common.cond import ConditionEncoder
from .generator.rnn import GinkaRNN
from .dataset import GinkaRNNDataset
from .generator.loss import RNNGinkaLoss
from shared.image import matrix_to_image_cv

# 手工标注标签定义：
# 0. 蓝海, 1. 红海, 2: 室内, 3. 野外, 4. 左右对称, 5. 上下对称, 6. 伪对称, 7. 咸鱼层,
# 8. 剧情层, 9. 水层, 10. 爽塔, 11. Boss层, 12. 纯Boss层, 13. 多房间, 14. 多走廊, 15. 道具风
# 16. 区域入口, 17. 区域连接, 18. 有机关门, 19. 道具层, 20. 斜向对称, 21. 左右通道, 22. 上下通道, 23. 多机关门
# 24. 中心对称, 25. 部分对称, 26. 鱼骨

# 自动标注标签定义：
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

BATCH_SIZE = 8

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.makedirs("result", exist_ok=True)
os.makedirs("result/rnn", exist_ok=True)
os.makedirs("result/ginka_rnn_img", exist_ok=True)

disable_tqdm = not sys.stdout.isatty()

def parse_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--state_ginka", type=str, default="result/rnn/ginka-100.pth")
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
    
    cond_inj = ConditionEncoder().to(device)
    ginka_rnn = GinkaRNN().to(device)
    
    dataset = GinkaRNNDataset(args.train, device)
    dataset_val = GinkaRNNDataset(args.validate, device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE)
    
    optimizer_ginka = optim.Adam(list(ginka_rnn.parameters()) + list(cond_inj.parameters()), lr=1e-3, betas=(0.0, 0.9))
    scheduler_ginka = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ginka, T_0=10, T_mult=2)

    criterion = RNNGinkaLoss()

    # 用于生成图片
    tile_dict = dict()
    for file in os.listdir('tiles'):
        name = os.path.splitext(file)[0]
        tile_dict[name] = cv2.imread(f"tiles/{file}", cv2.IMREAD_UNCHANGED)
        
    if args.resume:
        data_ginka = torch.load(args.state_ginka, map_location=device)

        ginka_rnn.load_state_dict(data_ginka["model_state"], strict=False)
            
        if args.load_optim:
            if data_ginka.get("optim_state") is not None:
                optimizer_ginka.load_state_dict(data_ginka["optim_state"])
            
        print("Train from loaded state.")
        
    for epoch in tqdm(range(args.epochs), desc="RNN Training", disable=disable_tqdm):
        loss_total_ginka = torch.Tensor([0]).to(device)
        
        iters = 0
        
        for batch in tqdm(dataloader, leave=False, desc="Epoch Progress", disable=disable_tqdm):
            tag_cond = batch["tag_cond"].to(device)
            val_cond = batch["val_cond"].to(device)
            target_map = batch["target_map"].to(device)
            
            B, D = val_cond.shape
            stage = torch.Tensor([0]).expand(B, 1).to(device)
            cond_vec = cond_inj(tag_cond, val_cond, stage)
            fake = ginka_rnn(target_map, cond_vec)
            
            loss = criterion.rnn_loss(fake, target_map)
            
            loss.backward()
            optimizer_ginka.step()
            loss_total_ginka += loss.detach()
            
            iters += 1
            
            # if iters % 100 == 0:
            #     avg_loss_ginka = loss_total_ginka.item() / iters
                
            #     tqdm.write(
            #         f"[Iters {iters} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " +
            #         f"E: {epoch + 1} | Loss: {avg_loss_ginka:.6f} | " +
            #         f"LR: {optimizer_ginka.param_groups[0]['lr']:.6f}"
            #     )
                
        avg_loss_ginka = loss_total_ginka.item() / iters
        tqdm.write(
            f"[Epoch {epoch} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " +
            f"E: {epoch + 1} | Loss: {avg_loss_ginka:.6f} | " +
            f"LR: {optimizer_ginka.param_groups[0]['lr']:.6f}"
        )
        
        scheduler_ginka.step()

        # 每若干轮输出一次图片，并保存检查点
        if (epoch + 1) % args.checkpoint == 0:
            # 保存检查点
            torch.save({
                "model_state": ginka_rnn.state_dict(),
                "optim_state": optimizer_ginka.state_dict(),
            }, f"result/rnn/ginka-{epoch + 1}.pth")
        
            val_loss_total = torch.Tensor([0]).to(device)
            with torch.no_grad():
                idx = 0
                for batch in tqdm(dataloader_val, desc="Validating generator.", leave=False, disable=disable_tqdm):
                    tag_cond = batch["tag_cond"].to(device)
                    val_cond = batch["val_cond"].to(device)
                    target_map = batch["target_map"].to(device)
                    
                    B, T = val_cond.shape
                    stage = torch.Tensor([0]).expand(B, 1).to(device)
                    cond_vec = cond_inj(tag_cond, val_cond, stage)
                    fake = ginka_rnn(target_map, cond_vec)
                    
                    val_loss_total += criterion.rnn_loss(fake, target_map).detach()
                    
                    B, T, C = fake.shape
                    fake_map = torch.argmax(fake, dim=-1).reshape(B, 13, 13).cpu().numpy()
                    fake_img = matrix_to_image_cv(fake_map[0], tile_dict)
                    cv2.imwrite(f"result/ginka_rnn_img/{idx}.png", fake_img)
                    
                    idx += 1
                
    print("Train ended.")
    torch.save({
        "model_state": ginka_rnn.state_dict(),
    }, f"result/ginka_rnn.pth")
    

if __name__ == "__main__":
    torch.set_num_threads(4)
    train()
