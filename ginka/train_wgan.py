import argparse
import os
import sys
from datetime import datetime
import torch
import torch.optim as optim
import cv2
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from .model.model import GinkaModel
from .dataset import GinkaWGANDataset
from .model.loss import WGANGinkaLoss
from minamo.model.model import MinamoScoreModule
from minamo.model.similarity import MinamoSimilarityModel
from shared.graph import batch_convert_soft_map_to_graph
from shared.image import matrix_to_image_cv
from shared.constant import VISION_WEIGHT, TOPO_WEIGHT

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
    # 1 代表课程学习阶段，2 代表课程学习后，逐渐转为联合学习的阶段
    # 3 代表课程学习后的联合遮挡学习阶段，4 代表最后随机输入的联合学习阶段
    train_stage = 1
    mask_ratio = 0.1 # 蒙版区域大小，每次增加 0.1，到达 0.9 之后进入阶段 2 的训练
    random_ratio = 0
    stage3_epoch = 0 # 第三阶段 epoch 数，100 轮后进入第四阶段
    
    ginka = GinkaModel()
    minamo = MinamoScoreModule()
    ginka.to(device)
    minamo.to(device)
    
    dataset = GinkaWGANDataset(args.train, device)
    dataset_val = GinkaWGANDataset(args.validate, device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer_ginka = optim.Adam(ginka.parameters(), lr=1e-4, betas=(0.0, 0.9))
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
            
        if data_ginka.get("random_ratio") is not None:
            random_ratio = data_ginka["random_ratio"]
            
        if data_ginka.get("stage_epoch3") is not None:
            stage3_epoch = data_ginka["stage_epoch3"]
            
        if data_ginka.get("stage") is not None:
            train_stage = data_ginka["stage"]
            
        if args.load_optim:
            if data_ginka.get("optim_state") is not None:
                optimizer_ginka.load_state_dict(data_ginka["optim_state"])
            if data_minamo.get("optim_state") is not None:
                optimizer_minamo.load_state_dict(data_minamo["optim_state"])
                
        dataset.train_stage = train_stage
        dataset.mask_ratio1 = mask_ratio
        dataset.mask_ratio2 = mask_ratio
        dataset.mask_ratio3 = mask_ratio
        dataset.random_ratio = random_ratio
        
        dataset_val.train_stage = train_stage
        dataset_val.mask_ratio1 = mask_ratio
        dataset_val.mask_ratio2 = mask_ratio
        dataset_val.mask_ratio3 = mask_ratio
        dataset_val.random_ratio = random_ratio
            
        print("Train from loaded state.")
        
    low_loss_epochs = 0
        
    for epoch in tqdm(range(args.epochs), desc="WGAN Training", disable=disable_tqdm):
        loss_total_minamo = torch.Tensor([0]).to(device)
        loss_total_ginka = torch.Tensor([0]).to(device)
        dis_total = torch.Tensor([0]).to(device)
        loss_ce_total = torch.Tensor([0]).to(device)
        
        for batch in tqdm(dataloader, leave=False, desc="Epoch Progress", disable=disable_tqdm):
            real1, masked1, real2, masked2, real3, masked3 = [item.to(device) for item in batch]
            
            # ---------- 训练判别器
            for _ in range(c_steps):
                # 生成假样本
                optimizer_minamo.zero_grad()
                optimizer_ginka.zero_grad()
                if train_stage == 1 or train_stage == 2:
                    fake1, fake2, fake3 = gen_curriculum(ginka, masked1, masked2, masked3, True)
                    
                elif train_stage == 3 or train_stage == 4:
                    fake1, fake2, fake3 = gen_total(ginka, masked1, True, True)
                
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
                    fake1, fake2, fake3 = gen_total(ginka, masked1, True, False)
                    
                    loss_g1 = criterion.generator_loss_total(minamo, 1, fake1)
                    loss_g2 = criterion.generator_loss_total(minamo, 2, fake2)
                    loss_g3 = criterion.generator_loss_total(minamo, 3, fake3)
                    
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
            f"Epoch: {epoch + 1} | W: {avg_dis:.8f} | " +
            f"G: {avg_loss_ginka:.8f} | D: {avg_loss_minamo:.8f} | " +
            f"CE: {avg_loss_ce:.8f} | Mask: {mask_ratio:.2f}"
        )
        
        if avg_loss_ce < 0.1:
            low_loss_epochs += 1
        else:
            low_loss_epochs = 0
            
        if low_loss_epochs >= 5 and train_stage == 2:
            if random_ratio >= 0.5:
                train_stage = 3
            random_ratio += 0.1
            random_ratio = min(random_ratio, 0.5)
            low_loss_epochs = 0
        
        if low_loss_epochs >= 5 and train_stage == 1:
            if mask_ratio >= 0.9:
                train_stage = 2
            mask_ratio += 0.1
            mask_ratio = min(mask_ratio, 0.9)
            low_loss_epochs = 0
            
        if train_stage == 3:
            stage3_epoch += 1
            if stage3_epoch >= 100:
                train_stage = 4
                stage3_epoch = 0
                
        if train_stage >= 2:
            # 第二阶段后 L1 损失不再应该生效
            mask_ratio = 1.0
            
        dataset.train_stage = 2
        dataset_val.train_stage = 2
        dataset.random_ratio = random_ratio
        dataset_val.random_ratio = random_ratio
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
                "random_ratio": random_ratio,
                "stage3_epoch": stage3_epoch,
            }, f"result/wgan/ginka-{epoch + 1}.pth")
            torch.save({
                "model_state": minamo.state_dict(),
                "optim_state": optimizer_minamo.state_dict()
            }, f"result/wgan/minamo-{epoch + 1}.pth")
            
            idx = 0
            with torch.no_grad():
                for batch in tqdm(dataloader_val, desc="Validating generator.", leave=False, disable=disable_tqdm):
                    real1, masked1, real2, masked2, real3, masked3 = [item.to(device) for item in batch]
                    if train_stage == 1:
                        fake1, fake2, fake3 = gen_curriculum(ginka, masked1, masked2, masked3, True)
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
