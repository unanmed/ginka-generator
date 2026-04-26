import torch.nn as nn

class Transformer(nn.Module):
    def __init__(
        self, d_model=256, dim_ff=512, nhead=8, num_layers=4,
    ):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True, activation='gelu'),
            num_layers=num_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True, activation='gelu'),
            num_layers=num_layers
        )
        
    def forward(self, x, memory=None):
        # x:      [B, S, d_model]  地图 token 序列
        # memory: [B, L, d_model]  可选的 z 投影，用于 cross-attention
        # 若 memory 为 None，则退化为原始自编解码行为（向后兼容）
        enc_out = self.encoder(x)
        if memory is not None:
            # encoder 输出作为 query，z 作为 key/value
            out = self.decoder(enc_out, memory)
        else:
            out = self.decoder(x, enc_out)
        return out
        