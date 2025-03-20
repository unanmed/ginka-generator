import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from minamo.model.model import MinamoModel
from .dataset import GinkaDataset
from .model.loss import GinkaLoss
from .model.model import GinkaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('result/ginka_img', exist_ok=True)

def blend_alpha(bg, fg, alpha):
    """ 使用 alpha 通道混合前景图块和背景图 """
    for c in range(3):  # 只混合 RGB 三个通道
        bg[:, :, c] = (1 - alpha) * bg[:, :, c] + alpha * fg[:, :, c]
    return bg

def matrix_to_image_cv(map_matrix, tile_set, tile_size=32):
    """
    使用OpenCV加速的版本（适合大尺寸地图）
    :param map_matrix: [H, W] 的numpy数组
    :param tile_set: 字典 {tile_id: cv2图像（BGR格式）}
    :param tile_size: 图块边长（像素）
    """
    H, W = map_matrix.shape  # 获取地图尺寸
    canvas = np.zeros((H * tile_size, W * tile_size, 3), dtype=np.uint8)  # 画布（黑色背景）
    
    # 遍历地图矩阵
    for row in range(H):
        for col in range(W):
            tile_index = str(map_matrix[row, col])  # 获取当前坐标的图块类型
            x, y = col * tile_size, row * tile_size  # 计算像素位置

            # 先绘制地面（0）
            if '0' in tile_set:
                canvas[y:y+tile_size, x:x+tile_size] = tile_set['0'][:, :, :3]  # 仅填充 RGB

            if tile_index == '11':
                if row == 0:
                    tile_index = '11_2'
                elif row == W - 1:
                    tile_index = '11_4'
                elif col == 0:
                    tile_index = '11_1'
                elif col == H - 1:
                    tile_index = '11_3'

            # 叠加其他透明图块
            if tile_index in tile_set and tile_index != 0:
                tile_rgba = tile_set[tile_index]
                tile_rgb = tile_rgba[:, :, :3]  # 提取 RGB
                alpha = tile_rgba[:, :, 3] / 255.0  # 归一化 alpha

                # 混合当前图块到背景
                canvas[y:y+tile_size, x:x+tile_size] = blend_alpha(
                    canvas[y:y+tile_size, x:x+tile_size], tile_rgb, alpha
                )

    return canvas

def validate():
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} to validate model.")
    model = GinkaModel()
    state = torch.load("result/ginka_checkpoint/15.pth", map_location=device)["model_state"]
    model.load_state_dict(state)
    model.to(device)
    
    minamo = MinamoModel(32)
    minamo.load_state_dict(torch.load("result/minamo.pth", map_location=device)["model_state"])
    minamo.to(device)

    # 准备数据集
    val_dataset = GinkaDataset("ginka-eval.json", device, minamo)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True
    )
    
    criterion = GinkaLoss(minamo)
    
    tile_dict = dict()
    val_output = dict()
    
    for file in os.listdir('tiles'):
        name = os.path.splitext(file)[0]
        tile_dict[name] = cv2.imread(f"tiles/{file}", cv2.IMREAD_UNCHANGED)
    
    minamo.eval()
    model.eval()
    val_loss = 0
    idx = 0
    with torch.no_grad():
         for batch in tqdm(val_loader):
            # 数据迁移到设备
            target = batch["target"].to(device)
            target_vision_feat = batch["target_vision_feat"].to(device)
            target_topo_feat = batch["target_topo_feat"].to(device)
            feat_vec = torch.cat([target_vision_feat, target_topo_feat], dim=-1).to(device)
            # 前向传播
            output = model(feat_vec)
            map_matrix = torch.argmax(output, dim=1)
            
            for matrix in map_matrix[:].cpu():
                image = matrix_to_image_cv(matrix.numpy(), tile_dict)
                cv2.imwrite(f"result/ginka_img/{idx}.png", image)
                val_output[f"val_{idx}"] = matrix.tolist()
                idx += 1
            
            # 计算损失
            loss = criterion(output, target, target_vision_feat, target_topo_feat)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    tqdm.write(f"Validation::loss: {avg_val_loss:.6f}")
    
    with open('result/ginka_val.json', 'w') as f:
        json.dump(val_output, f)
    
if __name__ == "__main__":
    torch.set_num_threads(2)
    validate()
    