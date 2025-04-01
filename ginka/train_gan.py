import argparse
import socket
import struct
import os
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
from .model.model import GinkaModel
from .model.loss import GinkaLoss
from .dataset import GinkaDataset, MinamoGANDataset
from minamo.model.model import MinamoModel
from minamo.model.loss import MinamoLoss
from shared.image import matrix_to_image_cv

BATCH_SIZE = 32
EPOCHS_GINKA = 30
EPOCHS_MINAMO = 10
SOCKET_PATH = "./tmp/ginka_uds"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("result", exist_ok=True)
os.makedirs("result/ginka_checkpoint", exist_ok=True)
os.makedirs("tmp", exist_ok=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--from_state", type=str, default="result/ginka.pth")
    parser.add_argument("--train", type=str, default="ginka-dataset.json")
    parser.add_argument("--validate", type=str, default='ginka-eval.json')
    parser.add_argument("--from_cycle", type=int, default=2)
    parser.add_argument("--to_cycle", type=int, default=100)
    args = parser.parse_args()
    return args

def parse_ginka_batch(batch):
    target = batch["target"].to(device)
    target_vision_feat = batch["target_vision_feat"].to(device)
    target_topo_feat = batch["target_topo_feat"].to(device)
    feat_vec = torch.cat([target_vision_feat, target_topo_feat], dim=-1).to(device).squeeze(1)
    
    return target, target_vision_feat, target_topo_feat, feat_vec

def parse_minamo_batch(batch):
    map1, map2, vision_simi, topo_simi, graph1, graph2 = batch
    map1 = map1.to(device) # 转为 [B, C, H, W]
    map2 = map2.to(device)
    topo_simi = topo_simi.to(device)
    vision_simi = vision_simi.to(device)
    graph1 = graph1.to(device)
    graph2 = graph2.to(device)
    return map1, map2, vision_simi, topo_simi, graph1, graph2

def send_all(sock, data):
    total_sent = 0
    while total_sent < len(data):
        sent = sock.send(data[total_sent:])
        if sent == 0:
            raise RuntimeError("Socket connection broken")
        total_sent += sent
        
def parse_minamo_data(sock: socket.socket, maps: np.ndarray):
    # 数据通讯 node 输出协议，单位字节：
    # 2 - Tensor count; 2 - Review count. Review is right behind train data;
    # 1*tc - Compare count for every map tensor delivered.
    # 2*4*(N+rc) - Vision similarity and topo similarity, like vis, topo, vis, topo;
    # N*1*H*W - Compare map for every map tensor. rc*2*H*W - Review map tensor.
    _, _, H, W = maps.shape
    tc_buf = sock.recv(2)
    rc_buf = sock.recv(2)
    tc = struct.unpack('>h', tc_buf)[0]
    rc = struct.unpack('>h', rc_buf)[0]
    count_buf = sock.recv(1 * tc)
    count: list = struct.unpack(f">{tc}b", count_buf)[0]
    N = sum(count)
    sim_buf = sock.recv(2 * 4 * (N + rc))
    com_buf = sock.recv(N * 1 * H * W)
    review_buf = sock.recv(rc * 2 * H * W) if rc > 0 else bytes()
        
    sim = struct.unpack(f">{(N + rc) * 2}f", sim_buf)[0]
    com = struct.unpack(f">{N * 1 * H * W}b", com_buf)[0]
    review = struct.unpack(f">{rc * 2 * H * W}", review_buf)[0] if rc > 0 else list()
    
    res = list()
    flatten_idx = 0
    # 读取当前这一轮生成器的数据
    for idx in range(N):
        com_count = count[idx]
        for i in range(com_count):
            com_start = flatten_idx * H * W
            com_end = (flatten_idx + 1) * H * W
            vis_sim = sim[flatten_idx * 2]
            topo_sim = sim[flatten_idx * 2 + 1]
            com_data = com[com_start:com_end]
            flatten_idx += 1
            com_map = np.fromiter(com_data, np.int8).view(H, W)
            # map1, map2, vision_similarity, topo_similarity, is_review
            res.append((maps[idx], com_map, vis_sim, topo_sim, False))
            
    return res    

def train():
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} to train model.")
    
    args = parse_arguments("result/ginka.pth", "ginka-dataset.json", 'ginka-eval.json')
    
    ginka = GinkaModel()
    ginka.to(device)
    minamo = MinamoModel(32)
    minamo.load_state_dict(torch.load("result/minamo.pth", map_location=device)["model_state"])
    minamo.to(device)
    minamo.eval()

    # 准备数据集
    ginka_dataset = GinkaDataset(args.train, device, minamo)
    ginka_dataset_val = GinkaDataset(args.validate, device, minamo)
    minamo_dataset = MinamoGANDataset()
    minamo_dataset_val = MinamoGANDataset()
    ginka_dataloader = DataLoader(ginka_dataset, batch_size=BATCH_SIZE, shuffle=True)
    ginka_dataloader_val = DataLoader(ginka_dataset_val, batch_size=BATCH_SIZE, shuffle=True)
    minamo_dataloader = DataLoader(minamo_dataset, batch_size=BATCH_SIZE, shuffle=True)
    minamo_dataloader_val = DataLoader(minamo_dataset_val, batch_size=BATCH_SIZE, shuffle=True)
    
    # 设定优化器与调度器
    optimizer_ginka = optim.AdamW(ginka.parameters(), lr=1e-3)
    scheduler_ginka = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ginka, T_0=10, T_mult=2, eta_min=1e-6)
    criterion_ginka = GinkaLoss(minamo)
    
    optimizer_minamo = optim.AdamW(minamo.parameters(), lr=1e-3)
    scheduler_minamo = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_minamo, T_0=10, T_mult=2, eta_min=1e-6)
    criterion_minamo = MinamoLoss()
    
    # 用于生成图片
    tile_dict = dict()
    for file in os.listdir('tiles'):
        name = os.path.splitext(file)[0]
        tile_dict[name] = cv2.imread(f"tiles/{file}", cv2.IMREAD_UNCHANGED)
        
    # 与 node 端通讯
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    
    if args.resume:
        data = torch.load(args.from_state, map_location=device)
        ginka.load_state_dict(data["model_state"], strict=False)
        if args.load_optim:
            optimizer_ginka.load_state_dict(data["optimizer_state"])
        print("Train from loaded state.")
        
    else:
        # 从头开始训练的话，初始时先把 minamo 损失值权重改为 0
        criterion_ginka.weight[0] = 0.0
    
    for cycle in tqdm(range(args.from_cycle, args.to_cycle)):
        # -------------------- 训练生成器
        gen_list: np.ndarray = np.empty(np.int8)
        prob_list: np.ndarray = np.empty(np.float32)
        for epoch in tqdm(range(args.epochs), desc="Training Ginka Model"):
            ginka.train()
            minamo.eval()
            total_loss = 0
            
            # 从头开始训练的，在第 10 个 epoch 将 minamo 损失值权重改回来
            if not args.resume and epoch == 10:
                criterion_ginka.weight[0] = 0.5
            
            for batch in ginka_dataloader:
                # 数据迁移到设备
                target, target_vision_feat, target_topo_feat, feat_vec = parse_ginka_batch(batch)
                # 前向传播
                optimizer_ginka.zero_grad()
                _, output_softmax = ginka(feat_vec)
                # 计算损失
                losses = criterion_ginka(output_softmax, target, target_vision_feat, target_topo_feat)
                # 反向传播
                losses.backward()
                optimizer_ginka.step()
                total_loss += losses.item()
                
            avg_loss = total_loss / len(ginka_dataloader)
            tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: {epoch + 1} | loss: {avg_loss:.6f} | lr: {(optimizer_ginka.param_groups[0]['lr']):.6f}")
            
            # 学习率调整
            scheduler_ginka.step()
            
            if (epoch + 1) % 5 == 0:
                loss_val = 0
                ginka.eval()
                idx = 0
                with torch.no_grad():
                    for batch in ginka_dataloader_val:
                        target, target_vision_feat, target_topo_feat, feat_vec = parse_ginka_batch(batch)
                        output, output_softmax = ginka(feat_vec)
                        losses = criterion_ginka(output_softmax, target, target_vision_feat, target_topo_feat)
                        loss_val += losses.item()
                        if epoch + 1 == EPOCHS_GINKA:
                            # 最后一次验证的时候顺带生成图片
                            prob = output_softmax.cpu().numpy()
                            np.concatenate((prob_list, prob), axis=1)
                            map_matrix = torch.argmax(output, dim=1).cpu().numpy()
                            gen_list = np.concatenate((gen_list, map_matrix), axis=1)
                            for matrix in map_matrix:
                                image = matrix_to_image_cv(matrix, tile_dict)
                                cv2.imwrite(f"result/ginka_img/{idx}.png", image)
                                idx += 1
                
                avg_val_loss = loss_val / len(ginka_dataloader_val)
                tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation::loss: {avg_val_loss:.6f}")
                torch.save({
                    "model_state": ginka.state_dict()
                }, f"result/ginka_checkpoint/{epoch + 1}.pth")
        
        tqdm.write(f"Cycle {cycle} Ginka train ended.")
        torch.save({
            "model_state": ginka.state_dict()
        }, f"result/ginka.pth")
        
        # -------------------- 生成 Minamo 的训练数据
        
        # 数据通讯 python 输出协议，单位字节：
        # 2 - Tensor count; 1 - Map height; 1 - Map Width; N*1*H*W - Map tensor, int8 type.
        N, H, W = gen_list.shape
        gen_bytes = gen_list.astype(np.int8).tobytes()
        buf = bytearray()
        buf.extend(struct.pack('>h', N)) # Tensor count
        buf.extend(struct.pack('>b', H)) # Map height
        buf.extend(struct.pack('>b', W)) # Map width
        buf.extend(gen_bytes) # Map tensor
        server.sendall(buf)
        data = parse_minamo_data(server, prob_list)
        minamo_dataset.set_data(data)
        
        # -------------------- 训练判别器
        for epoch in tqdm(range(EPOCHS_MINAMO), leave=False, desc="Training Minamo Model"):
            ginka.eval()
            minamo.train()
            total_loss = 0
            
            for batch in minamo_dataloader:
                map1, map2, vision_simi, topo_simi, graph1, graph2 = parse_minamo_batch(batch)
                
                if map1.shape[0] == 1:
                    continue
                
                # 前向传播
                optimizer_minamo.zero_grad()
                vision_feat1, topo_feat1 = minamo(map1, graph1)
                vision_feat2, topo_feat2 = minamo(map2, graph2)
                
                vision_pred = F.cosine_similarity(vision_feat1, vision_feat2, -1).unsqueeze(-1)
                topo_pred = F.cosine_similarity(topo_feat1, topo_feat2, -1).unsqueeze(-1)
                
                # 计算损失
                loss = criterion_minamo(vision_pred, topo_pred, vision_simi, topo_simi)
                
                # 反向传播
                loss.backward()
                optimizer_minamo.step()
                total_loss += loss.item()
                
            ave_loss = total_loss / len(minamo_dataloader)
            tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: {epoch + 1} | loss: {ave_loss:.6f} | lr: {(optimizer_ginka.param_groups[0]['lr']):.6f}")

            scheduler_minamo.step()
            
            # 每十轮推理一次验证集
            if (epoch + 1) % 5 == 0:
                minamo.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_batch in tqdm(minamo_dataloader_val, leave=False):
                        map1_val, map2_val, vision_simi_val, topo_simi_val, graph1, graph2 = parse_minamo_batch(val_batch)
                        
                        vision_feat1, topo_feat1 = minamo(map1_val, graph1)
                        vision_feat2, topo_feat2 = minamo(map2_val, graph2)
                
                        vision_pred = F.cosine_similarity(vision_feat1, vision_feat2, -1).unsqueeze(-1)
                        topo_pred = F.cosine_similarity(topo_feat1, topo_feat2, -1).unsqueeze(-1)
                        
                        # 计算损失
                        loss_val = criterion_minamo(vision_pred, topo_pred, vision_simi_val, topo_simi_val)
                        val_loss += loss_val.item()
                        
                avg_val_loss = val_loss / len(minamo_dataloader_val)
                tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation::loss: {avg_val_loss:.6f}")
                torch.save({
                    "model_state": minamo.state_dict()
                }, f"result/minamo_checkpoint/{epoch + 1}.pth")
                
        tqdm.write(f"Cycle {cycle} Minamo train ended.")
        torch.save({
            "model_state": minamo.state_dict()
        }, f"result/ginka.pth")
        
    print("Train ended.")

if __name__ == "__main__":
    torch.set_num_threads(4)
    train()
