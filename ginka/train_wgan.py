import argparse
import os
import sys
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
import cv2
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from .model.model import GinkaModel
from .dataset import GinkaWGANDataset
from .model.loss import WGANGinkaLoss
from .model.input import RandomInputHead
from minamo.model.model import MinamoScoreModule
from shared.image import matrix_to_image_cv

BATCH_SIZE = 16

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
    args = parser.parse_args()
    return args

def gen_curriculum(gen, masked1, masked2, masked3, detach=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fake1: torch.Tensor = gen(masked1, 1)
    fake2: torch.Tensor = gen(masked2, 2)
    fake3: torch.Tensor = gen(masked3, 3)
    if detach:
        return fake1.detach(), fake2.detach(), fake3.detach()
    else:
        return fake1, fake2, fake3
    
def gen_total(gen, input, progress_detach=True, result_detach=False) -> torch.Tensor:
    if progress_detach:
        fake1 = gen(input.detach(), 1)
        fake2 = gen(fake1.detach(), 2)
        fake3 = gen(fake2.detach(), 3)
    else:
        fake1 = gen(input, 1)
        fake2 = gen(fake1, 2)
        fake3 = gen(fake2, 3)
    if result_detach:
        return fake1.detach(), fake2.detach(), fake3.detach()
    else:
        return fake1, fake2, fake3

def train():
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} to train model.")
    
    args = parse_arguments()
    
    c_steps = 5
    g_steps = 1
    # 训练阶段
    train_stage = 1
    last_stage = False
    mask_ratio = 0.2 # 蒙版区域大小，每次增加 0.1，到达 0.9 之后进入阶段 2 的训练
    stage_epoch = 0 # 记录当前阶段的 epoch 数，用于控制训练过程
    
    ginka = GinkaModel().to(device)
    ginka_head = RandomInputHead().to(device)
    minamo = MinamoScoreModule().to(device)
    
    dataset = GinkaWGANDataset(args.train, device)
    dataset_val = GinkaWGANDataset(args.validate, device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer_ginka = optim.Adam(ginka.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_head = optim.Adam(ginka_head.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_minamo = optim.Adam(minamo.parameters(), lr=1e-5, betas=(0.0, 0.9))
    
    # scheduler_ginka = optim.lr_scheduler.CosineAnnealingLR(optimizer_ginka, T_max=args.epochs)
    # scheduler_minamo = optim.lr_scheduler.CosineAnnealingLR(optimizer_minamo, T_max=args.epochs)
    
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
        
        if data_ginka.get("c_steps") is not None and data_ginka.get("g_steps") is not None:
            c_steps = data_ginka["c_steps"]
            g_steps = data_ginka["g_steps"]
            
        if data_ginka.get("mask_ratio") is not None:
            mask_ratio = data_ginka["mask_ratio"]
            
        if data_ginka.get("stage_epoch") is not None:
            stage_epoch = data_ginka["stage_epoch"]
            
        if data_ginka.get("stage") is not None:
            train_stage = data_ginka["stage"]
            
        if data_ginka.get("last_stage") is not None:
            last_stage = data_ginka["last_stage"]
            
        if args.load_optim:
            if data_ginka.get("optim_state") is not None:
                optimizer_ginka.load_state_dict(data_ginka["optim_state"])
            if data_minamo.get("optim_state") is not None:
                optimizer_minamo.load_state_dict(data_minamo["optim_state"])
                
        dataset.train_stage = train_stage
        dataset.mask_ratio1 = mask_ratio
        dataset.mask_ratio2 = mask_ratio
        dataset.mask_ratio3 = mask_ratio
        
        dataset_val.train_stage = train_stage
        dataset_val.mask_ratio1 = mask_ratio
        dataset_val.mask_ratio2 = mask_ratio
        dataset_val.mask_ratio3 = mask_ratio
            
        print("Train from loaded state.")
        
    low_loss_epochs = 0
        
    for epoch in tqdm(range(args.epochs), desc="WGAN Training", disable=disable_tqdm):
        loss_total_minamo = torch.Tensor([0]).to(device)
        loss_total_ginka = torch.Tensor([0]).to(device)
        dis_total = torch.Tensor([0]).to(device)
        loss_ce_total = torch.Tensor([0]).to(device)
        
        for batch in tqdm(dataloader, leave=False, desc="Epoch Progress", disable=disable_tqdm):
            real1, masked1, real2, masked2, real3, masked3 = [item.to(device) for item in batch]
            
            if train_stage == 4:
                # 最后一个阶段训练输入头
                count = 5 if stage_epoch <= 20 else 2
                for _ in range(count):
                    optimizer_head.zero_grad()
                    output = F.softmax(ginka_head(masked1), dim=1)
                    loss_head = criterion.generator_input_head_loss(output)
                    loss_head.backward()
                    optimizer_head.step()
            
            # ---------- 训练判别器
            for _ in range(c_steps):
                # 生成假样本
                optimizer_minamo.zero_grad()
                optimizer_ginka.zero_grad()
                optimizer_head.zero_grad()
                
                with torch.no_grad():
                    if train_stage == 1 or train_stage == 2:
                        fake1, fake2, fake3 = gen_curriculum(ginka, masked1, masked2, masked3, True)
                        
                    elif train_stage == 3:
                        fake1, fake2, fake3 = gen_total(ginka, masked1, True, True)
                    
                    elif train_stage == 4:
                        input = F.softmax(ginka_head(masked1), dim=1)
                        fake1, fake2, fake3 = gen_total(ginka, input, True, True)
                        
                
                loss_d1, dis1 = criterion.discriminator_loss(minamo, 1, real1, fake1)
                loss_d2, dis2 = criterion.discriminator_loss(minamo, 2, real2, fake2)
                loss_d3, dis3 = criterion.discriminator_loss(minamo, 3, real3, fake3)
                
                dis_avg = (dis1 + dis2 + dis3) / 3.0
                loss_d_avg = (loss_d1 + loss_d2 + loss_d3) / 3.0

                # 反向传播
                loss_d_avg.backward()
                
                optimizer_minamo.step()
                
                loss_total_minamo += loss_d_avg.detach()
                dis_total += dis_avg.detach()
            
            # ---------- 训练生成器
            
            for _ in range(g_steps):
                optimizer_minamo.zero_grad()
                optimizer_ginka.zero_grad()
                optimizer_head.zero_grad()
                if train_stage == 1 or train_stage == 2:
                    fake1, fake2, fake3 = gen_curriculum(ginka, masked1, masked2, masked3, False)
                    
                    loss_g1, _, loss_ce_g1, _ = criterion.generator_loss(minamo, 1, mask_ratio, real1, fake1, masked1)
                    loss_g2, _, loss_ce_g2, _ = criterion.generator_loss(minamo, 2, mask_ratio, real2, fake2, masked2)
                    loss_g3, _, loss_ce_g3, _ = criterion.generator_loss(minamo, 3, mask_ratio, real3, fake3, masked3)
                    
                    loss_g = (loss_g1 + loss_g2 + loss_g3) / 3.0
                    loss_ce = max(loss_ce_g1, loss_ce_g2, loss_ce_g3)
                    
                    loss_g.backward()
                    optimizer_ginka.step()
                    loss_total_ginka += loss_g.detach()
                    loss_ce_total += loss_ce.detach()
                    
                elif train_stage == 3 or train_stage == 4:
                    input = masked1 if train_stage == 3 else F.softmax(ginka_head(masked1), dim=1)

                    fake1, fake2, fake3 = gen_total(ginka, input, True, False)
                    
                    if train_stage == 3:
                        loss_g1 = criterion.generator_loss_total_with_input(minamo, 1, fake1, input)
                    else:
                        loss_g1 = criterion.generator_loss_total(minamo, 1, fake1)
                    loss_g2 = criterion.generator_loss_total_with_input(minamo, 2, fake2, fake1)
                    loss_g3 = criterion.generator_loss_total_with_input(minamo, 3, fake3, fake2)
                    
                    loss_g = (loss_g1 + loss_g2 + loss_g3) / 3.0
                    loss_g.backward()
                    optimizer_ginka.step()
                    loss_total_ginka += loss_g.detach()
            
        avg_loss_ginka = loss_total_ginka.item() / len(dataloader) / g_steps
        avg_loss_minamo = loss_total_minamo.item() / len(dataloader) / c_steps
        avg_loss_ce = loss_ce_total.item() / len(dataloader) / g_steps
        avg_dis = dis_total.item() / len(dataloader) / c_steps
        tqdm.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " +
            f"Epoch: {epoch + 1} | S: {train_stage} | W: {avg_dis:.6f} | " +
            f"G: {avg_loss_ginka:.6f} | D: {avg_loss_minamo:.6f} | " +
            f"CE: {avg_loss_ce:.6f} | M: {mask_ratio:.1f}"
        )
        
        if avg_loss_ce < 1.0:
            low_loss_epochs += 1
        else:
            low_loss_epochs = 0
        
        if low_loss_epochs >= 3 and train_stage == 1:
            if mask_ratio >= 0.9:
                train_stage = 2
                stage_epoch = 0
            mask_ratio += 0.2
            mask_ratio = min(mask_ratio, 0.9)
            low_loss_epochs = 0
            
        if train_stage == 3 or train_stage == 2:
            if stage_epoch >= 25:
                train_stage += 1
                stage_epoch = 0
                
        if train_stage >= 3:
            # 第三阶段后交叉熵损失不再应该生效
            mask_ratio = 1.0
            
        if last_stage:
            if train_stage == 2 and stage_epoch % 5 == 0:
                train_stage = 4
            
            if train_stage == 4 and stage_epoch % 5 == 1:
                train_stage = 2
        
        stage_epoch += 1
            
        dataset.train_stage = train_stage
        dataset_val.train_stage = train_stage
        dataset.mask_ratio1 = dataset.mask_ratio2 = dataset.mask_ratio3 = mask_ratio
        dataset_val.mask_ratio1 = dataset_val.mask_ratio2 = dataset_val.mask_ratio3 = mask_ratio
        
        # scheduler_ginka.step()
        # scheduler_minamo.step()
        
        if avg_dis < 0:
            g_steps = max(int(-avg_dis * 5), 1)
        else:
            g_steps = 1
            
        if avg_loss_minamo > 0:
            c_steps = int(min(5 + avg_loss_minamo * 5, 15))
        else:
            c_steps = 5
        
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
                "last_stage": last_stage
            }, f"result/wgan/ginka-{epoch + 1}.pth")
            torch.save({
                "model_state": minamo.state_dict(),
                "optim_state": optimizer_minamo.state_dict()
            }, f"result/wgan/minamo-{epoch + 1}.pth")
            
            idx = 0
            with torch.no_grad():
                for batch in tqdm(dataloader_val, desc="Validating generator.", leave=False, disable=disable_tqdm):
                    real1, masked1, real2, masked2, real3, masked3 = [item.to(device) for item in batch]
                    if train_stage == 1 or train_stage == 2:
                        fake1, fake2, fake3 = gen_curriculum(ginka, masked1, masked2, masked3, True)
                            
                    elif train_stage == 3 or train_stage == 4:
                        input = masked1 if train_stage == 3 else F.softmax(ginka_head(masked1), dim=1)
                        fake1, fake2, fake3 = gen_total(ginka, input, True, True)
                        
                    fake1 = torch.argmax(fake1, dim=1).cpu().numpy()
                    fake2 = torch.argmax(fake2, dim=1).cpu().numpy()
                    fake3 = torch.argmax(fake3, dim=1).cpu().numpy()
                        
                    for i in range(fake1.shape[0]):
                        for key, one in enumerate([fake1, fake2, fake3]):
                            map_matrix = one[i]
                            image = matrix_to_image_cv(map_matrix, tile_dict)
                            cv2.imwrite(f"result/ginka_img/{idx}_{key}.png", image)

                        idx += 1
    
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
