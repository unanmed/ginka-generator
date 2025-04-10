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

BATCH_SIZE = 32

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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint", type=int, default=5)
    parser.add_argument("--load_optim", type=bool, default=True)
    args = parser.parse_args()
    return args

def clip_weights(model, clip_value=0.01):
    for param in model.parameters():
        param.data = torch.clamp(param.data, -clip_value, clip_value)

def train():
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} to train model.")
    
    args = parse_arguments()
    
    # c_steps = 1 if args.resume else 5
    # g_steps = 5 if args.resume else 1
    c_steps = 5
    g_steps = 1
    
    ginka = GinkaModel()
    minamo = MinamoScoreModule()
    minamo_sim = MinamoSimilarityModel()
    ginka.to(device)
    minamo.to(device)
    minamo_sim.to(device)
    
    dataset = GinkaWGANDataset(args.train, device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    optimizer_ginka = optim.Adam(ginka.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_minamo = optim.Adam(minamo.parameters(), lr=1e-5, betas=(0.0, 0.9))
    optimizer_minamo_sim = optim.Adam(minamo_sim.parameters(), lr=1e-4)
    
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
        
        if data_ginka["c_steps"] is not None and data_ginka["g_steps"] is not None:
            c_steps = data_ginka["c_steps"]
            g_steps = data_ginka["g_steps"]
            
        if args.load_optim:
            if data_ginka["optim_state"] is not None:
                optimizer_ginka.load_state_dict(data_ginka["optim_state"])
            if data_minamo["optim_state"] is not None:
                optimizer_minamo.load_state_dict(data_minamo["optim_state"])
            if data_minamo["optim_state_sim"] is not None:
                optimizer_minamo_sim.load_state_dict(data_minamo["optim_state_sim"])
            
        print("Train from loaded state.")
        
    for epoch in tqdm(range(args.epochs), desc="WGAN Training", disable=disable_tqdm):
        loss_total_minamo = torch.Tensor([0]).to(device)
        loss_total_minamo_sim = torch.Tensor([0]).to(device)
        loss_total_ginka = torch.Tensor([0]).to(device)
        dis_total = torch.Tensor([0]).to(device)
        
        for real_data in tqdm(dataloader, leave=False, desc="Epoch Progress", disable=disable_tqdm):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            real_graph = batch_convert_soft_map_to_graph(real_data)
            
            # ---------- 训练判别器
            for _ in range(c_steps):
                # 生成假样本
                optimizer_minamo.zero_grad()
                z = torch.rand(batch_size, 1024, device=device)
                fake_data = ginka(z)
                fake_data = fake_data.detach()
                
                # 计算判别器输出
                # 反向传播
                dis, loss_d = criterion.discriminator_loss(minamo, real_data, real_graph, fake_data)
                loss_d.backward()
                # torch.nn.utils.clip_grad_norm_(minamo.parameters(), max_norm=2.0)
                # total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in minamo.topo_model.parameters()]), 2)
                # print("Critic 梯度范数:", total_norm.item())
                # print("Critic 输入范围:", fake_data.min().item(), fake_data.max().item(), real_data.min().item(), real_data.max().item())
                # print("Critic 输出范围:", d_real.min().item(), d_real.max().item())
                optimizer_minamo.step()
                
                loss_total_minamo += loss_d.detach()
                dis_total += dis.detach()
            
            # ---------- 训练生成器
            
            for _ in range(g_steps):
                optimizer_ginka.zero_grad()
                # optimizer_minamo_sim.zero_grad()
                
                z1 = torch.randn(batch_size, 1024, device=device)
                z2 = torch.randn(batch_size, 1024, device=device)
                fake_softmax1, fake_softmax2 = ginka(z1), ginka(z2)
                
                # 先训练辅助判别器
                # loss_c_assist = criterion.discriminator_loss_assist2(minamo_sim, real_data, fake_softmax1, fake_softmax2)
                # loss_c_assist.backward(retain_graph=True)
                # optimizer_minamo_sim.step()
                
                loss_g = criterion.generator_loss(minamo, minamo_sim, fake_softmax1, fake_softmax2)
                loss_g.backward()
                optimizer_ginka.step()
                
                loss_total_ginka += loss_g
                # loss_total_minamo_sim += loss_c_assist.detach()
            # tqdm.write(f"{dis.item():.12f}, {loss_d.item():.12f}, {loss_g.item():.12f}")
            
        avg_loss_ginka = loss_total_ginka.item() / len(dataloader) / g_steps
        avg_loss_minamo = loss_total_minamo.item() / len(dataloader) / c_steps
        avg_loss_minamo_sim = loss_total_minamo_sim.item() / len(dataloader) / g_steps
        avg_dis = dis_total.item() / len(dataloader) / c_steps
        tqdm.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " +\
            f"Epoch: {epoch + 1} | W Loss: {avg_dis:.8f} | " +\
            f"G Loss: {avg_loss_ginka:.8f} | D Loss: {avg_loss_minamo:.8f} | " +\
            f"lr G: {(optimizer_ginka.param_groups[0]['lr']):.8f}"
        )
        
        # scheduler_ginka.step()
        # scheduler_minamo.step()
        
        if avg_dis < 0:
            g_steps = max(int(-avg_dis * 5), 1)
        else:
            g_steps = 1
            
        if avg_loss_ginka > 0 or avg_loss_minamo > 0:
            c_steps = min(5 + (avg_loss_ginka + avg_loss_minamo) * 5, 15)
        else:
            c_steps = 5
        
        # 每若干轮输出一次图片，并保存检查点
        if (epoch + 1) % args.checkpoint == 0:
            # 输出 20 张图片，每批次 4 张，一共五批
            idx = 0
            with torch.no_grad():
                for _ in range(5):
                    z = torch.randn(4, 1024, device=device)
                    output = ginka(z)
                    
                    map_matrix = torch.argmax(output, dim=1).cpu().numpy()
                    for matrix in map_matrix:
                        image = matrix_to_image_cv(matrix, tile_dict)
                        cv2.imwrite(f"result/ginka_img/{idx}.png", image)
                        idx += 1
            
            # 保存检查点
            torch.save({
                "model_state": ginka.state_dict(),
                "optim_state": optimizer_ginka.state_dict(),
                "c_steps": c_steps,
                "g_steps": g_steps
            }, f"result/wgan/ginka-{epoch + 1}.pth")
            torch.save({
                "model_state": minamo.state_dict(),
                "model_state_sim": minamo_sim.state_dict(),
                "optim_state": optimizer_minamo.state_dict(),
                "optim_state_sim": optimizer_minamo_sim.state_dict()
            }, f"result/wgan/minamo-{epoch + 1}.pth")
    
    print("Train ended.")
    torch.save({
        "model_state": ginka.state_dict(),
    }, f"result/ginka.pth")
    torch.save({
        "model_state": minamo.state_dict(),
        "model_state_sim": minamo_sim.state_dict(),
    }, f"result/minamo.pth")

if __name__ == "__main__":
    torch.set_num_threads(4)
    train()
