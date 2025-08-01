import argparse
import os
import sys
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from .generator.model import GinkaModel
from .dataset import GinkaWGANDataset
from .generator.loss import WGANGinkaLoss
from .critic.model import MinamoModel2
from shared.image import matrix_to_image_cv

# 标签定义：
# 0. 蓝海, 1. 红海, 2: 室内, 3. 野外, 4. 左右对称, 5. 上下对称, 6. 伪对称, 7. 咸鱼层,
# 8. 剧情层, 9. 水层, 10. 爽塔, 11. Boss层, 12. 纯Boss层, 13. 多房间, 14. 多走廊, 15. 道具风
# 16. 区域入口, 17. 区域连接, 18. 有机关门, 19. 道具层, 20. 斜向对称, 21. 左右通道, 22. 上下通道, 23. 多机关门
# 24. 中心对称, 25. 部分对称, 26. 鱼骨

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
# 9. 道具数量
# 10. 入口数量

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
# 29. 楼梯入口
# 30. 箭头入口

BATCH_SIZE = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("result", exist_ok=True)
os.makedirs("result/wgan", exist_ok=True)

disable_tqdm = not sys.stdout.isatty()

def parse_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--state_ginka", type=str, default="result/wgan/ginka-100.pth")
    parser.add_argument("--state_minamo", type=str, default="result/wgan/minamo-100.pth")
    parser.add_argument("--train", type=str, default="ginka-dataset.json")
    parser.add_argument("--validate", type=str, default="ginka-eval.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint", type=int, default=5)
    parser.add_argument("--load_optim", type=bool, default=True)
    parser.add_argument("--curr_epoch", type=int, default=20) # 课程学习至少多少 epoch
    parser.add_argument("--tuning", type=bool, default=False)
    args = parser.parse_args()
    return args

def gen_curriculum(gen, masked1, masked2, masked3, tag, val, detach=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fake1 = gen(masked1, 1, tag, val)
    fake2 = gen(masked2, 2, tag, val)
    fake3 = gen(masked3, 3, tag, val)
    if detach:
        return fake1.detach(), fake2.detach(), fake3.detach()
    else:
        return fake1, fake2, fake3
    
def gen_total(gen, input, tag, val, progress_detach=True, result_detach=False, random=False) -> torch.Tensor:
    if random:
        fake0 = gen(input, 0, tag, val)
        x_in = F.softmax(fake0, dim=1)
    else:
        fake0 = input
        x_in = input
    if progress_detach:
        fake1 = gen(x_in.detach(), 1, tag, val)
        fake2 = gen(F.softmax(fake1.detach(), dim=1), 2, tag, val)
        fake3 = gen(F.softmax(fake2.detach(), dim=1), 3, tag, val)
    else:
        fake1 = gen(x_in, 1, tag, val)
        fake2 = gen(F.softmax(fake1, dim=1), 2, tag, val)
        fake3 = gen(F.softmax(fake2, dim=1), 3, tag, val)
    if result_detach:
        return fake1.detach(), fake2.detach(), fake3.detach(), fake0.detach()
    else:
        return fake1, fake2, fake3, fake0

def train():
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} to train model.")
    
    args = parse_arguments()
    
    c_steps = 2
    g_steps = 1
    # 训练阶段
    train_stage = 1
    mask_ratio = 0.2 # 蒙版区域大小
    stage_epoch = 0 # 记录当前阶段的 epoch 数，用于控制训练过程
    total_epoch = 0
    
    ginka = GinkaModel().to(device)
    minamo = MinamoModel2().to(device)
    
    dataset = GinkaWGANDataset(args.train, device)
    dataset_val = GinkaWGANDataset(args.validate, device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE)
    
    optimizer_ginka = optim.Adam(ginka.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_minamo = optim.Adam(minamo.parameters(), lr=1e-4, betas=(0.0, 0.9))
    
    scheduler_ginka = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ginka, T_0=100, T_mult=1)
    scheduler_minamo = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_minamo, T_0=100, T_mult=1)
    
    criterion = WGANGinkaLoss()
    
    # 用于生成图片
    tile_dict = dict()
    for file in os.listdir('tiles'):
        name = os.path.splitext(file)[0]
        tile_dict[name] = cv2.imread(f"tiles/{file}", cv2.IMREAD_UNCHANGED)
        
    if args.resume:
        data_ginka = torch.load(args.state_ginka, map_location=device)
        data_minamo = torch.load(args.state_minamo, map_location=device)
        
        ginka.load_state_dict(data_ginka["model_state"], strict=False)
        minamo.load_state_dict(data_minamo["model_state"], strict=False)
        
        # if data_ginka.get("c_steps") is not None and data_ginka.get("g_steps") is not None:
        #     c_steps = data_ginka["c_steps"]
        #     g_steps = data_ginka["g_steps"]
            
        if data_ginka.get("mask_ratio") is not None:
            mask_ratio = data_ginka["mask_ratio"]
            
        if data_ginka.get("stage_epoch") is not None:
            stage_epoch = data_ginka["stage_epoch"]
            
        if data_ginka.get("stage") is not None:
            train_stage = data_ginka["stage"]
            
        if data_ginka.get("total_epoch") is not None:
            total_epoch = data_ginka["data_ginka"]
            
        if args.load_optim:
            if data_ginka.get("optim_state") is not None:
                optimizer_ginka.load_state_dict(data_ginka["optim_state"])
            if data_minamo.get("optim_state") is not None:
                optimizer_minamo.load_state_dict(data_minamo["optim_state"])
            
        print("Train from loaded state.")
        
    curr_epoch = args.curr_epoch
    first_curr = curr_epoch * 3
        
    if args.tuning:
        train_stage = 1
        curr_epoch = curr_epoch // 4
        first_curr = first_curr // 4
        stage_epoch = 0
        mask_ratio = 0.2
        
    dataset.train_stage = train_stage
    dataset.mask_ratio1 = mask_ratio
    dataset.mask_ratio2 = mask_ratio
    dataset.mask_ratio3 = mask_ratio
    
    dataset_val.train_stage = train_stage
    dataset_val.mask_ratio1 = mask_ratio
    dataset_val.mask_ratio2 = mask_ratio
    dataset_val.mask_ratio3 = mask_ratio
        
    for epoch in tqdm(range(args.epochs), desc="WGAN Training", disable=disable_tqdm):
        loss_total_minamo = torch.Tensor([0]).to(device)
        loss_total_ginka = torch.Tensor([0]).to(device)
        dis_total = torch.Tensor([0]).to(device)
        loss_ce_total = torch.Tensor([0]).to(device)
        
        for batch in tqdm(dataloader, leave=False, desc="Epoch Progress", disable=disable_tqdm):
            rand = batch["rand"].to(device)
            real0 = batch["real0"].to(device)
            real1 = batch["real1"].to(device)
            masked1 = batch["masked1"].to(device)
            real2 = batch["real2"].to(device)
            masked2 = batch["masked2"].to(device)
            real3 = batch["real3"].to(device)
            masked3 = batch["masked3"].to(device)
            tag_cond = batch["tag_cond"].to(device)
            val_cond = batch["val_cond"].to(device)
            
            # ---------- 训练判别器
            for _ in range(c_steps):
                # 生成假样本
                optimizer_minamo.zero_grad()
                optimizer_ginka.zero_grad()
                
                with torch.no_grad():
                    if train_stage == 1 or train_stage == 2:
                        fake1, fake2, fake3 = gen_curriculum(ginka, masked1, masked2, masked3, tag_cond, val_cond, True)
                    elif train_stage == 3 or train_stage == 4:
                        fake1, fake2, fake3, fake0 = gen_total(ginka, masked1, tag_cond, val_cond, True, True, train_stage == 4)
                    
                    if train_stage < 4:
                        fake0 = ginka(rand, 0, tag_cond, val_cond)
                    
                loss_d0, dis0 = criterion.discriminator_loss(minamo, 0, real0, fake0, tag_cond, val_cond)
                loss_d1, dis1 = criterion.discriminator_loss(minamo, 1, real1, fake1, tag_cond, val_cond)
                loss_d2, dis2 = criterion.discriminator_loss(minamo, 2, real2, fake2, tag_cond, val_cond)
                loss_d3, dis3 = criterion.discriminator_loss(minamo, 3, real3, fake3, tag_cond, val_cond)
                
                dis = [dis0, dis1, dis2, dis3]
                loss_d = [loss_d0, loss_d1, loss_d2, loss_d3]
                    
                dis_avg = sum(dis) / len(dis)
                loss_d_avg = sum(loss_d) / len(loss_d)

                # 反向传播
                loss_d_avg.backward()
                
                optimizer_minamo.step()
                
                loss_total_minamo += loss_d_avg.detach()
                dis_total += dis_avg.detach()
            
            # ---------- 训练生成器
            
            for _ in range(g_steps):
                optimizer_minamo.zero_grad()
                optimizer_ginka.zero_grad()
                if train_stage == 1 or train_stage == 2:
                    fake1, fake2, fake3 = gen_curriculum(ginka, masked1, masked2, masked3, tag_cond, val_cond, False)
                    
                    loss_g1, loss_ce_g1 = criterion.generator_loss(minamo, 1, mask_ratio, real1, fake1, masked1, tag_cond, val_cond)
                    loss_g2, loss_ce_g2 = criterion.generator_loss(minamo, 2, mask_ratio, real2, fake2, masked2, tag_cond, val_cond)
                    loss_g3, loss_ce_g3 = criterion.generator_loss(minamo, 3, mask_ratio, real3, fake3, masked3, tag_cond, val_cond)
                    
                    loss_g = (loss_g1 * 3.0 + loss_g2 + loss_g3) / 5.0
                    loss_ce = max(loss_ce_g1, loss_ce_g2, loss_ce_g3)
                    
                    loss_ce_total += loss_ce.detach()
                    
                elif train_stage == 3 or train_stage == 4:
                    fake1, fake2, fake3, fake0 = gen_total(ginka, masked1, tag_cond, val_cond, True, False, train_stage == 4)
                    if train_stage == 4:
                        fake0 = F.softmax(fake0, dim=1)

                    loss_g1 = criterion.generator_loss_total_with_input(minamo, 1, fake1, fake0, tag_cond, val_cond)
                    loss_g2 = criterion.generator_loss_total_with_input(minamo, 2, fake2, fake1, tag_cond, val_cond)
                    loss_g3 = criterion.generator_loss_total_with_input(minamo, 3, fake3, fake2, tag_cond, val_cond)
                    
                    loss_g = (loss_g1 * 3.0 + loss_g2 + loss_g3) / 5.0
                    
                if train_stage < 4:
                    fake0 = F.softmax(ginka(rand, 0, tag_cond, val_cond), dim=1)
                    
                loss_g0 = criterion.generator_input_head_loss(minamo, fake0, tag_cond, val_cond)
                loss_g += loss_g0

                loss_g.backward()
                optimizer_ginka.step()
                loss_total_ginka += loss_g.detach()
            
        avg_loss_ginka = loss_total_ginka.item() / len(dataloader) / g_steps
        avg_loss_minamo = loss_total_minamo.item() / len(dataloader) / c_steps
        avg_loss_ce = loss_ce_total.item() / len(dataloader) / g_steps
        avg_dis = dis_total.item() / len(dataloader) / c_steps
        tqdm.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " +
            f"E: {epoch + 1} | S: {train_stage} | W: {avg_dis:.6f} | " +
            f"G: {avg_loss_ginka:.6f} | D: {avg_loss_minamo:.6f} | " +
            f"CE: {avg_loss_ce:.6f} | M: {mask_ratio:.1f} | " +
            f"LR: {optimizer_ginka.param_groups[0]['lr']:.6f}"
        )
        
        # 每若干轮输出一次图片，并保存检查点
        if (epoch + 1) % args.checkpoint == 0:
            # 保存检查点
            torch.save({
                "model_state": ginka.state_dict(),
                "optim_state": optimizer_ginka.state_dict(),
                "c_steps": c_steps,
                "g_steps": g_steps,
                "stage": train_stage,
                "mask_ratio": mask_ratio,
                "stage_epoch": stage_epoch,
            }, f"result/wgan/ginka-{epoch + 1}.pth")
            torch.save({
                "model_state": minamo.state_dict(),
                "optim_state": optimizer_minamo.state_dict()
            }, f"result/wgan/minamo-{epoch + 1}.pth")
            
            idx = 0
            gap = 5
            color = (255, 255, 255)  # 白色
            with torch.no_grad():
                for batch in tqdm(dataloader_val, desc="Validating generator.", leave=False, disable=disable_tqdm):
                    real1 = batch["real1"].to(device)
                    masked1 = batch["masked1"].to(device)
                    real2 = batch["real2"].to(device)
                    masked2 = batch["masked2"].to(device)
                    real3 = batch["real3"].to(device)
                    masked3 = batch["masked3"].to(device)
                    tag_cond = batch["tag_cond"].to(device)
                    val_cond = batch["val_cond"].to(device)
                    
                    if train_stage == 1 or train_stage == 2:
                        fake1, fake2, fake3 = gen_curriculum(ginka, masked1, masked2, masked3, tag_cond, val_cond, True)
                            
                    elif train_stage == 3 or train_stage == 4:
                        fake1, fake2, fake3, fake0 = gen_total(ginka, masked1, tag_cond, val_cond, True, True, train_stage == 4)
                        fake0 = torch.argmax(fake0, dim=1).cpu().numpy()
                        
                    fake1 = torch.argmax(fake1, dim=1).cpu().numpy()
                    fake2 = torch.argmax(fake2, dim=1).cpu().numpy()
                    fake3 = torch.argmax(fake3, dim=1).cpu().numpy()
                    masked1 = torch.argmax(masked1, dim=1).cpu().numpy()
                    masked2 = torch.argmax(masked2, dim=1).cpu().numpy()
                    masked3 = torch.argmax(masked3, dim=1).cpu().numpy()

                    for i in range(fake1.shape[0]):
                        fake1_img = matrix_to_image_cv(fake1[i], tile_dict)
                        fake2_img = matrix_to_image_cv(fake2[i], tile_dict)
                        fake3_img = matrix_to_image_cv(fake3[i], tile_dict)
                        if train_stage == 1 or train_stage == 2:
                            vline = np.full((416, gap, 3), color, dtype=np.uint8)  # 垂直分割线
                            hline = np.full((gap, 3 * 416 + gap * 2, 3), color, dtype=np.uint8)  # 水平分割线
                            in1_img = matrix_to_image_cv(masked1[i], tile_dict)
                            in2_img = matrix_to_image_cv(masked2[i], tile_dict)
                            in3_img = matrix_to_image_cv(masked3[i], tile_dict)
                            img = np.block([
                                [[in1_img], [vline], [in2_img], [vline], [in3_img]],
                                [[hline]],
                                [[fake1_img], [vline], [fake2_img], [vline], [fake3_img]]
                            ])
                        elif train_stage == 3 or train_stage == 4:
                            vline = np.full((416, gap, 3), color, dtype=np.uint8)  # 垂直分割线
                            hline = np.full((gap, 2 * 416 + gap, 3), color, dtype=np.uint8)  # 水平分割线
                            in_img = matrix_to_image_cv(fake0[i], tile_dict)
                            img = np.block([
                                [[in_img], [vline], [fake1_img]],
                                [[hline]],
                                [[fake2_img], [vline], [fake3_img]]
                            ])
                        
                        cv2.imwrite(f"result/ginka_img/{idx}.png", img)

                        idx += 1
                        
        # 训练流程控制
        
        if train_stage >= 2:
            # train_stage = 4
            if (epoch + 1) % 100 == 5:
                train_stage = 3
            elif (epoch + 1) % 100 == 20:
                train_stage = 4
            elif (epoch + 1) % 100 == 0:
                train_stage = 2
        
        if train_stage == 1:
            if (mask_ratio < 0.3 and stage_epoch >= first_curr) or \
                (mask_ratio > 0.3 and stage_epoch >= curr_epoch):
                mask_ratio += 0.2
                mask_ratio = min(mask_ratio, 0.8)

                stage_epoch = 0
                if mask_ratio >= 0.8:
                    train_stage = 2
        
        stage_epoch += 1
        total_epoch += 1

        dataset.train_stage = train_stage
        dataset_val.train_stage = train_stage
        dataset.mask_ratio1 = dataset.mask_ratio2 = dataset.mask_ratio3 = mask_ratio
        dataset_val.mask_ratio1 = dataset_val.mask_ratio2 = dataset_val.mask_ratio3 = mask_ratio
        
        scheduler_ginka.step()
        scheduler_minamo.step()
    
    print("Train ended.")
    torch.save({
        "model_state": ginka.state_dict(),
    }, f"result/ginka.pth")
    torch.save({
        "model_state": minamo.state_dict(),
    }, f"result/minamo.pth")

if __name__ == "__main__":
    torch.set_num_threads(4)
    train()
