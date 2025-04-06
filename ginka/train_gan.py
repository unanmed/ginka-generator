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
from .model.loss import GinkaLoss, WGANGinkaLoss
from .dataset import GinkaDataset, MinamoGANDataset
from minamo.model.model import MinamoModel
from minamo.model.loss import MinamoLoss
from shared.image import matrix_to_image_cv

BATCH_SIZE = 32
EPOCHS_GINKA = 5
EPOCHS_MINAMO = 2
SOCKET_PATH = "./tmp/ginka_uds"
LOSS_PATH = "result/gan/a-loss.txt"
REPLAY_PATH = "datasets/replay.bin"
VISION_ALPHA = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("result", exist_ok=True)
os.makedirs("result/ginka_checkpoint", exist_ok=True)
os.makedirs("result/gan", exist_ok=True)
os.makedirs("tmp", exist_ok=True)

with open(LOSS_PATH, 'a', encoding='utf-8') as f:
    f.write(f"---------- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ----------\n")

if not os.path.exists(REPLAY_PATH):
    with open(REPLAY_PATH, 'wb') as f:
        f.write(b'\x00\x00\x00\x00')

def parse_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--from_state", type=str, default="result/ginka.pth")
    parser.add_argument("--train", type=str, default="ginka-dataset.json")
    parser.add_argument("--validate", type=str, default='ginka-eval.json')
    parser.add_argument("--from_cycle", type=int, default=0)
    parser.add_argument("--to_cycle", type=int, default=100)
    args = parser.parse_args()
    return args

def parse_ginka_batch(batch):
    target = batch["target"].to(device)
    target_vision_feat = batch["target_vision_feat"].to(device).squeeze(1)
    target_topo_feat = batch["target_topo_feat"].to(device).squeeze(1)
    feat_vec = torch.cat([target_vision_feat, target_topo_feat], dim=1).to(device)
    
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
        
def recv_all(sock: socket.socket, length: int):
    """循环接收直到获得指定长度的数据"""
    data = bytes()
    while len(data) < length:
        packet = sock.recv(length - len(data))  # 只请求剩余部分
        if not packet:
            raise ConnectionError("连接中断")
        data += packet
    return data
        
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
    count_buf = recv_all(sock, 1 * tc)
    count: list = struct.unpack(f">{tc}b", count_buf)
    N = sum(count)
    sim_buf = recv_all(sock, 2 * 4 * (N + rc))
    com_buf = recv_all(sock, N * 1 * H * W)
    review_buf = recv_all(sock, rc * 2 * H * W) if rc > 0 else bytes()
        
    sim = struct.unpack(f">{(N + rc) * 2}f", sim_buf)
    com = struct.unpack(f">{N * 1 * H * W}b", com_buf)
    review = struct.unpack(f">{rc * 2 * H * W}", review_buf) if rc > 0 else list()
    
    res = list()
    flatten_idx = 0
    # 读取当前这一轮生成器的数据
    for idx in range(tc):
        com_count = count[idx]
        for i in range(com_count):
            com_start = flatten_idx * H * W
            com_end = (flatten_idx + 1) * H * W
            vis_sim = sim[flatten_idx * 2]
            topo_sim = sim[flatten_idx * 2 + 1]
            com_data = com[com_start:com_end]
            flatten_idx += 1
            com_map = np.array(com_data, dtype=np.int8).reshape(H, W)
            # map1, map2, vision_similarity, topo_similarity, is_review
            res.append((maps[idx], com_map, vis_sim, topo_sim, False))

    return res

def train():
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} to train model.")
    
    args = parse_arguments()
    
    ginka = GinkaModel()
    ginka.to(device)
    minamo = MinamoModel(32)
    minamo.load_state_dict(torch.load("result/minamo.pth", map_location=device)["model_state"])
    minamo.to(device)
    minamo.eval()

    # 准备数据集
    ginka_dataset = GinkaDataset(args.train, device, minamo)
    ginka_dataset_val = GinkaDataset(args.validate, device, minamo)
    minamo_dataset = MinamoGANDataset("datasets/minamo-dataset-1.json")
    minamo_dataset_val = MinamoGANDataset("datasets/minamo-eval-1.json")
    ginka_dataloader = DataLoader(ginka_dataset, batch_size=BATCH_SIZE, shuffle=True)
    ginka_dataloader_val = DataLoader(ginka_dataset_val, batch_size=BATCH_SIZE, shuffle=True)
    minamo_dataloader = DataLoader(minamo_dataset, batch_size=BATCH_SIZE // 2, shuffle=True)
    minamo_dataloader_val = DataLoader(minamo_dataset_val, batch_size=BATCH_SIZE // 2, shuffle=True)
    
    # 设定优化器与调度器
    optimizer_ginka = optim.Adam(ginka.parameters(), lr=1e-4, betas=(0.0, 0.9))
    scheduler_ginka = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ginka, T_0=10, T_mult=2, eta_min=1e-6)
    criterion_ginka = GinkaLoss(minamo)
    
    optimizer_minamo = optim.Adam(minamo.parameters(), lr=2e-5, betas=(0.0, 0.9))
    scheduler_minamo = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_minamo, T_0=EPOCHS_MINAMO, T_mult=2, eta_min=1e-6)
    criterion_minamo = MinamoLoss()
    
    criterion = WGANGinkaLoss()
    
    # 用于生成图片
    tile_dict = dict()
    for file in os.listdir('tiles'):
        name = os.path.splitext(file)[0]
        tile_dict[name] = cv2.imread(f"tiles/{file}", cv2.IMREAD_UNCHANGED)
        
    # 与 node 端通讯
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    
    if args.resume:
        data = torch.load(args.from_state, map_location=device)
        ginka.load_state_dict(data["model_state"], strict=False)
        print("Train from loaded state.")
        
    print("Waiting for client connection...")
    conn, _ = server.accept()
    print("Client connected.")
    
    for cycle in tqdm(range(args.from_cycle, args.to_cycle), desc="Total Progress"):
        # -------------------- 训练生成器
        for epoch in tqdm(range(EPOCHS_GINKA), desc="Training Ginka Model", leave=False):
            ginka.train()
            minamo.eval()
            total_loss = 0
            
            for batch in tqdm(ginka_dataloader, leave=False, desc="Epoch Progress"):
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
            scheduler_ginka.step(epoch + 1)
            
            if (epoch + 1) % 5 == 0:
                loss_val = 0
                ginka.eval()
                idx = 0
                with torch.no_grad():
                    for batch in tqdm(ginka_dataloader_val, leave=False, desc="Validating Ginka Model"):
                        target, target_vision_feat, target_topo_feat, feat_vec = parse_ginka_batch(batch)
                        output, output_softmax = ginka(feat_vec)
                        losses = criterion_ginka(output_softmax, target, target_vision_feat, target_topo_feat)
                        loss_val += losses.item()
                        if epoch + 1 == EPOCHS_GINKA:
                            # 最后一次验证的时候顺带生成图片
                            map_matrix = torch.argmax(output, dim=1).cpu().numpy()
                            for matrix in map_matrix:
                                image = matrix_to_image_cv(matrix, tile_dict)
                                cv2.imwrite(f"result/ginka_img/{idx}.png", image)
                                idx += 1
                
                avg_val_loss = loss_val / len(ginka_dataloader_val)
                tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation::loss: {avg_val_loss:.6f}")
                torch.save({
                    "model_state": ginka.state_dict()
                }, f"result/ginka_checkpoint/{epoch + 1}.pth")
                
        # 使用训练集生成 minamo 训练数据，更准确
        gen_list: np.ndarray = np.empty((0, 13, 13), np.int8)
        prob_list: np.ndarray = np.empty((0, 32, 13, 13), np.float32)
        with torch.no_grad():
            for batch in ginka_dataloader:
                target, target_vision_feat, target_topo_feat, feat_vec = parse_ginka_batch(batch)
                output, output_softmax = ginka(feat_vec)
                prob = output_softmax.cpu().numpy()
                prob_list = np.concatenate((prob_list, prob), axis=0)
                map_matrix = torch.argmax(output, dim=1).cpu().numpy()
                gen_list = np.concatenate((gen_list, map_matrix), axis=0)
        
        tqdm.write(f"Cycle {cycle} Ginka train ended.")
        torch.save({
            "model_state": ginka.state_dict()
        }, f"result/gan/ginka-{cycle}.pth")
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
        conn.sendall(buf)
        data = parse_minamo_data(conn, prob_list)

        vis_sim = 0
        topo_sim = 0
        for _, _, vis, topo, _ in data:
            vis_sim += vis
            topo_sim += topo
        
        vis_sim /= len(data)
        topo_sim /= len(data)
        
        with open(LOSS_PATH, 'a', encoding='utf-8') as f:
            f.write(f'Cycle {cycle} | Ginka Vision Similarity: {vis_sim:.12f} | Ginka Topo Similarity: {topo_sim:.12f} | Ginka Loss: {avg_val_loss:.12f}')
        
        # 经验回放部分
        with open(REPLAY_PATH, 'r+b') as f:
            # 读取文件开头获取总长度
            f.seek(0)
            count = struct.unpack('>i', f.read(4))[0]  # 取出整数
            if count > 0:
                replay = np.random.choice(count, size=min(count, len(data) // 4), replace=False)
                
                replay_data = np.empty((len(replay), 32, 13, 13))
                for i, n in enumerate(replay):
                    f.seek(n * 32 * 13 * 13 + 4)
                    arr = np.frombuffer(f.read(32 * 13 * 13 * 4), dtype=np.float32).reshape(32, 13, 13)
                    replay_data[i] = arr
                
                map_data: np.ndarray = replay_data.argmax(axis=1)
                buf = bytearray()
                buf.extend(struct.pack('>h', len(replay)))  # Tensor count
                buf.extend(struct.pack('>b', H))  # Map height
                buf.extend(struct.pack('>b', W))  # Map width
                buf.extend(map_data.astype(np.int8).tobytes())  # Map tensor
                conn.sendall(buf)
                data.extend(parse_minamo_data(conn, replay_data))

            # 把新的内容写入文件末尾
            to_write = np.random.choice(N, size=min(N, 100), replace=False)
            write_data = bytearray()
            for n in to_write:
                write_data.extend(prob_list[n].tobytes())
            
            f.seek(0, 2)  # 定位到文件末尾
            f.write(write_data)
            
            f.seek(0)  # 定位到文件开头
            f.write(struct.pack('>i', count + len(to_write)))
            f.flush()  # 确保数据被刷新到磁盘
        
        minamo_dataset.set_data(data)
        
        # -------------------- 训练判别器
        for epoch in tqdm(range(EPOCHS_MINAMO), leave=False, desc="Training Minamo Model"):
            ginka.eval()
            minamo.train()
            total_loss = 0
            
            for batch in tqdm(minamo_dataloader, leave=False, desc="Epoch Progress"):
                map1, map2, vis_sim, topo_sim, graph1, graph2 = parse_minamo_batch(batch)
                batch_size = map1.shape[0]
                
                if batch_size == 1:
                    continue
                
                # 前向传播
                optimizer_minamo.zero_grad()
                vis_feat_real, topo_feat_real = minamo(map1, graph1)
                vis_feat_ref, topo_feat_ref = minamo(map2, graph2)
                
                # 生成假数据
                with torch.no_grad():
                    fake_feat = torch.randn((batch_size, 1024), device=device)
                    fake_data = ginka(fake_feat)
                
                # 创建插值样本
                alpha = torch.rand((batch_size, 1, 1, 1), device=device)
                interpolates = (alpha * map2 + (1 - alpha) * fake_data).requires_grad_(True)
                
                vis_feat_fake, topo_feat_fake = minamo(fake_data)
                vis_feat_interp, topo_feat_interp = minamo(interpolates)
                
                vis_pred_real = F.cosine_similarity(vis_feat_real, vis_feat_ref, dim=1).unsqueeze(-1)
                topo_pred_real = F.cosine_similarity(topo_feat_real, topo_feat_ref, dim=1).unsqueeze(-1)
                vis_pred_fake = F.cosine_similarity(vis_feat_fake, vis_feat_ref, dim=1).unsqueeze(-1)
                topo_pred_fake = F.cosine_similarity(topo_feat_fake, topo_feat_ref, dim=1).unsqueeze(-1)
                vis_pred_interp = F.cosine_similarity(vis_feat_interp, vis_feat_ref, dim=1).unsqueeze(-1)
                topo_pred_interp = F.cosine_similarity(topo_feat_interp, topo_feat_ref, dim=1).unsqueeze(-1)
                
                # 计算相似度
                score_real = F.l1_loss(vis_pred_real, vis_sim) * VISION_ALPHA + F.l1_loss(topo_pred_real, topo_sim) * (1 - VISION_ALPHA)
                score_fake = vis_pred_fake * VISION_ALPHA + topo_pred_fake * (1 - VISION_ALPHA)
                score_interp = vis_pred_interp * VISION_ALPHA + topo_pred_interp * (1 - VISION_ALPHA)
                
                # 计算损失
                loss = criterion.discriminator_loss(score_real, score_fake, score_interp)
                
                # 反向传播
                loss.backward()
                optimizer_minamo.step()
                total_loss += loss.item()
                
            ave_loss = total_loss / len(minamo_dataloader)
            tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: {epoch + 1} | loss: {ave_loss:.6f} | lr: {(optimizer_minamo.param_groups[0]['lr']):.6f}")

            scheduler_minamo.step(epoch + 1)
            
            # 每十轮推理一次验证集
            if epoch + 1 == EPOCHS_MINAMO:
                minamo.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_batch in tqdm(minamo_dataloader_val, leave=False, desc="Validating Minamo Model"):
                        map1_val, map2_val, vision_simi_val, topo_simi_val, graph1, graph2 = parse_minamo_batch(val_batch)
                        
                        vis_feat_real, topo_feat_real = minamo(map1_val, graph1)
                        vis_feat_ref, topo_feat_ref = minamo(map2_val, graph2)
                
                        vis_pred_real = F.cosine_similarity(vis_feat_real, vis_feat_ref, dim=1).unsqueeze(-1)
                        topo_pred_real = F.cosine_similarity(topo_feat_real, topo_feat_ref, dim=1).unsqueeze(-1)
                        
                        # 计算损失
                        loss_val = criterion_minamo(vis_pred_real, topo_pred_real, vision_simi_val, topo_simi_val)
                        val_loss += loss_val.item()
                        
                avg_val_loss = val_loss / len(minamo_dataloader_val)
                tqdm.write(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation::loss: {avg_val_loss:.6f}")
                torch.save({
                    "model_state": minamo.state_dict()
                }, f"result/minamo_checkpoint/{epoch + 1}.pth")
                
        tqdm.write(f"Cycle {cycle} Minamo train ended.")
        torch.save({
            "model_state": minamo.state_dict()
        }, f"result/gan/minamo-{cycle}.pth")
        torch.save({
            "model_state": minamo.state_dict()
        }, f"result/minamo.pth")
        with open(LOSS_PATH, 'a', encoding='utf-8') as f:
            f.write(f' | Minamo: {avg_val_loss:.12f}\n')
        
    print("Train ended.")

if __name__ == "__main__":
    torch.set_num_threads(4)
    train()
