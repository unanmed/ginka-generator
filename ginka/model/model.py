import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from ...shared.attention import CBAM, SpatialAttention
from .sample import HybridUpsample, FinalUpsample, GumbelSampler
    
class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels)
        )
    
    def forward(self, x):
        return x + self.conv(x)

class DynamicPadConv(nn.Module):
    """支持动态处理奇数尺寸的智能卷积"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel,
            stride=stride,
            padding=kernel//2
        )
        self.requires_pad = (stride > 1)  # 仅在下采样时需要填充

    def forward(self, x):
        if self.requires_pad:
            # 动态计算各维度需要填充的量
            pad_h = x.size(-2) % 2
            pad_w = x.size(-1) % 2
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))  # 右下填充
        return self.conv(x)
    
class ConditionInjector(nn.Module):
    """基于注意力机制的条件注入"""
    def __init__(self, cond_dim=128, feat_dim=256):
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, feat_dim * 2),
            nn.GELU(),
            nn.LayerNorm(feat_dim * 2)
        )
        self.channel_att = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim//8, 1),
            nn.GELU(),
            nn.Conv2d(feat_dim//8, feat_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, cond):
        # 投影条件向量
        gamma, beta = self.cond_proj(cond).chunk(2, dim=1)  # [B, D]
        
        # 通道注意力调制
        att = self.channel_att(x)  # [B, C, H, W]
        modulated = x * att
        
        # 添加条件偏置
        return modulated + beta.view(-1, gamma.size(1), 1, 1)
    
class GinkaEncoder(nn.Module):
    """编码器（下采样）部分"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encoder = nn.Sequential(
            DynamicPadConv(in_ch, out_ch, stride=1),
            ResidualBlock(out_ch),
            CBAM(out_ch),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.encoder(x)

class GinkaModel(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, num_classes=32):
        """Ginka Model 模型定义部分

        Args:
            in_ch (int, optional): 输入通道数，默认是 1
            base_ch (int, optional): UNet 上下采样卷积基础通道数，默认 64
            num_classes (int, optional): 图块种类数量，默认 32 预留出一部分以供后续拓展功能
        """
        super().__init__()
        
        # 轻量级文本编码器（使用BERT前4层）
        self.bert = BertModel.from_pretrained('google-bert/bert-base-chinese', output_hidden_states=True)
        self.text_proj = nn.Linear(768, 128)
        
        # 动态尺寸处理系统
        self.size_embed = nn.Embedding(32, 16)  # 处理最大32的尺寸
        
        # 编码器
        self.enc1 = GinkaEncoder(in_ch, base_ch)
        self.enc2 = GinkaEncoder(base_ch, base_ch * 2)
        # self.enc3 = GinkaEncoder(base_ch * 2, base_ch * 4)
        
        # 中间层
        self.mid = nn.Sequential(
            DynamicPadConv(base_ch * 2, base_ch * 4),
            ConditionInjector(160, base_ch * 4)
        )
        
        # 解码器，解码器仅使用空间注意力
        self.dec1 = HybridUpsample(base_ch * 4, base_ch * 2)
        self.dec1_att = SpatialAttention()
        
        self.dec2 = HybridUpsample(base_ch * 2, base_ch)
        self.dec2_att = SpatialAttention()
        
        # self.dec3 = HybridUpsample(base_ch * 2, base_ch)
        # self.dec3_att = SpatialAttention()

        # 输出层
        self.out = FinalUpsample(base_ch, num_classes)
        
    def forward(self, noise, input_ids, attention_mask, map_size):
        """
        Args:
            noise: 噪声输入 [BS, H, W, 1]
            input_ids: 文本token id [BS, seq_len]
            attention_mask: 文本attention mask [BS, seq_len]
            map_size: 地图尺寸 [BS, 2] (height, width)
        Returns:
            logits: 输出logits [BS, num_classes, H, W]
        """
        # 文本特征提取
        with torch.no_grad():  # 冻结BERT参数
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        # 取前4层隐藏状态的平均
        hidden_states = torch.stack(bert_outputs.hidden_states[1:5])  # [4, BS, seq_len, 768]
        text_features = torch.mean(hidden_states, dim=0)[:, 0, :]     # [BS, 768]
        text_features = self.text_proj(text_features)                 # [BS, 128]

        # 尺寸特征处理
        h_emb = self.size_embed(map_size[:, 0])  # [BS, 16]
        w_emb = self.size_embed(map_size[:, 1])  # [BS, 16]
        size_features = torch.cat([h_emb, w_emb], dim=1)  # [BS, 32]

        # 特征融合
        conditional = torch.cat([text_features, size_features], dim=1)  # [BS, 160]
        
        # 调整噪声输入维度
        x = noise.permute(0, 3, 1, 2)  # [BS, 1, H, W]
        
        # 编码器路径
        x1 = self.enc1(x)  # [BS, 64, H / 2, W / 2]
        x2 = self.enc2(x1)  # [BS, 128, H / 4, W / 4]
        
        # 中间层（注入条件）
        x_mid = self.mid[0](x2)  # [BS, 256, H / 4, W / 4]
        x_mid = self.mid[1](x_mid, conditional)
        
        # 解码器路径
        d1 = self.dec1(x_mid, x2)  # [BS, 128, H / 2, W / 2]
        d1 = self.dec1_att(d1)
        d2 = self.dec2(d1, x1)      # [BS, 64, H, W]
        d2 = self.dec2_att(d2)
        # d3 = self.dec3(d2, x1)
        # d3 = self.dec3_att(d3)
        
        # 最终自适应上采样
        h, w = noise.shape[1:3]  # 获取原始输入尺寸
        return self.out(d2, (h, w))
        
    