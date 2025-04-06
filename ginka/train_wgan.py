import argparse
import os
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
from shared.graph import batch_convert_soft_map_to_graph
from shared.image import matrix_to_image_cv
from shared.constant import VISION_WEIGHT, TOPO_WEIGHT

BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("result", exist_ok=True)
os.makedirs("result/wgan", exist_ok=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--state_ginka", type=str, default="result/ginka.pth")
    parser.add_argument("--state_minamo", type=str, default="result/minamo.pth")
    parser.add_argument("--train", type=str, default="ginka-dataset.json")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    return args

def clip_weights(model, clip_value=0.01):
    for param in model.parameters():
        param.data = torch.clamp(param.data, -clip_value, clip_value)

def train():
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} to train model.")
    
    c_steps = 1
    g_steps = 3
    
    args = parse_arguments()
    
    ginka = GinkaModel()
    minamo = MinamoScoreModule()
    ginka.to(device)
    minamo.to(device)
    
    dataset = GinkaWGANDataset(args.train, device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    optimizer_ginka = optim.RMSprop(ginka.parameters(), lr=2e-4)
    optimizer_minamo = optim.RMSprop(minamo.parameters(), lr=1e-5)
    
    criterion = WGANGinkaLoss()
    
    # 用于生成图片
    tile_dict = dict()
    for file in os.listdir('tiles'):
        name = os.path.splitext(file)[0]
        tile_dict[name] = cv2.imread(f"tiles/{file}", cv2.IMREAD_UNCHANGED)
        
    if args.resume:
        data = torch.load(args.state_ginka, map_location=device)
        ginka.load_state_dict(data["model_state"], strict=False)
        data = torch.load(args.state_minamo, map_location=device)
        minamo.load_state_dict(data["model_state"], strict=False)
        print("Train from loaded state.")
        
    for epoch in tqdm(range(args.epochs), desc="GAN Training"):
        loss_total_minamo = torch.Tensor([0]).to(device)
        loss_total_ginka = torch.Tensor([0]).to(device)
        dis_total = torch.Tensor([0]).to(device)
        
        for real_data in tqdm(dataloader, leave=False, desc="Epoch Progress"):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            real_graph = batch_convert_soft_map_to_graph(real_data)
            
            optimizer_ginka.zero_grad()
            
            # ---------- 训练判别器
            for _ in range(c_steps):
                # 生成假样本
                optimizer_minamo.zero_grad()
                z = torch.randn(batch_size, 1024, device=device)
                fake_data = ginka(z)
                fake_data = fake_data.detach()
                
                # 计算判别器输出
                # 反向传播
                dis, loss_d = criterion.discriminator_loss(minamo, real_data, real_graph, fake_data)
                loss_d.backward()
                # torch.nn.utils.clip_grad_norm_(minamo_vis.parameters(), max_norm=1.0)
                optimizer_minamo.step()
                
                loss_total_minamo += loss_d
                dis_total += dis
            
            # ---------- 训练生成器
            
            for _ in range(g_steps):
                z1 = torch.randn(batch_size, 1024, device=device)
                z2 = torch.randn(batch_size, 1024, device=device)
                fake_softmax1, fakse_softmax2 = ginka(z1), ginka(z2)
                
                loss_g = criterion.generator_loss(minamo, fake_softmax1, fakse_softmax2)
                loss_g.backward()
                optimizer_ginka.step()
                
                loss_total_ginka += loss_g
            # tqdm.write(f"{dis.item():.12f}, {loss_d.item():.12f}, {loss_g.item():.12f}")
            
        avg_loss_ginka = loss_total_ginka.item() / len(dataloader) / g_steps
        avg_loss_minamo = loss_total_minamo.item() / len(dataloader) / c_steps
        avg_dis = dis_total.item() / len(dataloader) / c_steps
        tqdm.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: {epoch + 1} | Wasserstein Loss: {avg_dis:.8f} | Loss Ginka: {avg_loss_ginka:.8f} | Loss Minamo: {avg_loss_minamo:.8f}"
        )
        
        if avg_dis < -9:
            g_steps = 7
        elif avg_dis < -6:
            g_steps = 5
        elif avg_dis < -3:
            g_steps = 3
        else:
            g_steps = 1
        
        # 每五轮输出一次图片，并保存检查点
        if (epoch + 1) % 5 == 0:
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
                "model_state": ginka.state_dict()
            }, f"result/wgan/ginka-{epoch + 1}.pth")
            torch.save({
                "model_state": minamo.state_dict()
            }, f"result/wgan/minamo-{epoch + 1}.pth")
    
    print("Train ended.")
    torch.save({
        "model_state": ginka.state_dict()
    }, f"result/ginka.pth")
    torch.save({
        "model_state": minamo.state_dict()
    }, f"result/minamo.pth")

if __name__ == "__main__":
    torch.set_num_threads(4)
    train()
