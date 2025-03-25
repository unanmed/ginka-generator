import torch.nn as nn

class MinamoLoss(nn.Module):
    def __init__(self, vision_weight=0.4, topo_weight=0.6):
        super().__init__()
        self.vision_weight = vision_weight
        self.topo_weight = topo_weight
        self.loss = nn.L1Loss()

    def forward(self, vis_pred, topo_pred, vis_true, topo_true):
        # print(vis_pred.shape, topo_pred.shape, vis_true.shape, topo_true.shape)
        # print(vis_pred[0].item(), topo_pred[0].item(), vis_true[0].item(), topo_true[0].item())
        vis_loss = self.loss(vis_pred, vis_true)
        topo_loss = self.loss(topo_pred, topo_true)
        # print(vis_loss.item(), topo_loss.item())
        return self.vision_weight * vis_loss + self.topo_weight * topo_loss
