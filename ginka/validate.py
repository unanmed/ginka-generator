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
from shared.image import matrix_to_image_cv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('result/ginka_img', exist_ok=True)

def validate():
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} to validate model.")
    model = GinkaModel()
    state = torch.load("result/ginka.pth", map_location=device)["model_state"]
    model.load_state_dict(state)
    model.to(device)
    
    for name, param in model.named_parameters():
        print(f"Layer: {name}, Params: {param.numel()}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
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
            feat_vec = torch.cat([target_vision_feat, target_topo_feat], dim=-1).to(device).squeeze(1)
            # 前向传播
            output, output_softmax = model(feat_vec)
            map_matrix = torch.argmax(output, dim=1)
            
            for matrix in map_matrix[:].cpu():
                image = matrix_to_image_cv(matrix.numpy(), tile_dict)
                cv2.imwrite(f"result/ginka_img/{idx}.png", image)
                val_output[f"val_{idx}"] = matrix.tolist()
                idx += 1
            
            # 计算损失
            _, loss = criterion(output_softmax, target, target_vision_feat, target_topo_feat)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    tqdm.write(f"Validation::loss: {avg_val_loss:.6f}")
    
    with open('result/ginka_val.json', 'w') as f:
        json.dump(val_output, f)
    
if __name__ == "__main__":
    torch.set_num_threads(2)
    validate()
    