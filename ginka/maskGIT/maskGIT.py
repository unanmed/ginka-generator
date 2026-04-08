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
        
    def forward(self, x):
        # x: [B, L, d_model]
        m = self.encoder(x)
        out = self.decoder(x, m)
        return out
        