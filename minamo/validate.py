import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from .model.model import MinamoModel
from .model.loss import MinamoLoss
from .dataset import MinamoDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate():
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} to validate model.")
    model = MinamoModel(32)
    model.load_state_dict(torch.load("result/minamo.pth", map_location=device)["model_state"])
    model.to(device)
    
    for name, param in model.named_parameters():
        print(f"Layer: {name}, Params: {param.numel()}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # 准备数据集
    val_dataset = MinamoDataset("datasets/minamo-eval-1.json")
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True
    )
    
    criterion = MinamoLoss()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_batch in tqdm(val_loader):
            map1_val, map2_val, vision_simi_val, topo_simi_val, graph1, graph2 = val_batch
            map1_val = map1_val.to(device)
            map2_val = map2_val.to(device)
            vision_simi_val = vision_simi_val.to(device)
            topo_simi_val = topo_simi_val.to(device)
            graph1 = graph1.to(device)
            graph2 = graph2.to(device)
            
            vision_feat1, topo_feat1 = model(map1_val, graph1)
            vision_feat2, topo_feat2 = model(map2_val, graph2)
            
            print(vision_feat1.isnan().any().item(), topo_feat1.isnan().any().item(), vision_feat2.isnan().any().item(), topo_feat2.isnan().any().item())
            
            vision_pred_val = F.cosine_similarity(vision_feat1, vision_feat2, -1).unsqueeze(-1)
            topo_pred_val = F.cosine_similarity(topo_feat1, topo_feat2, -1).unsqueeze(-1)
            loss_val = criterion(
                vision_pred_val, topo_pred_val,
                vision_simi_val, topo_simi_val
            )
            val_loss += loss_val.item()
            
    avg_val_loss = val_loss / len(val_loader)
    tqdm.write(f"Validation::loss: {avg_val_loss:.6f}")
    
if __name__ == "__main__":
    torch.set_num_threads(2)
    validate()
    