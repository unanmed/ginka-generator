import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from minamo.model.model import MinamoModel
from .dataset import GinkaDataset
from .model.loss import GinkaLoss
from .model.model import GinkaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate():
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} to validate model.")
    model = GinkaModel()
    
    minamo = MinamoModel(32)
    minamo.load_state_dict(torch.load("result/minamo.pth", map_location=device)["model_state"])
    minamo.to(device)

    # 准备数据集
    val_dataset = GinkaDataset("ginka-eval.json")
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True
    )
    
    criterion = GinkaLoss(minamo)
    
    minamo.eval()
    model.eval()
    val_loss = 0
    with torch.no_grad():
         for batch in val_loader:
            # 数据迁移到设备
            target = batch["target"].to(device)
            target_vision_feat = batch["target_vision_feat"].to(device)
            target_topo_feat = batch["target_topo_feat"].to(device)
            feat_vec = torch.cat([target_vision_feat, target_topo_feat], dim=-1).to(device)
            # 前向传播
            output, _ = model(feat_vec)
            map_matrix = torch.argmax(output, dim=1)
            
            # 计算损失
            loss = criterion(output, target, target_vision_feat, target_topo_feat)
            total_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    tqdm.write(f"Validation::loss: {avg_val_loss:.6f}")
    
if __name__ == "__main__":
    torch.set_num_threads(2)
    validate()
    