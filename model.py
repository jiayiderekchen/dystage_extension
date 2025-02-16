import torch
import torch.nn as nn
import math

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, n_heads=16, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.embedding_dim = input_dim
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(input_dim, 1)
        
    def forward(self, x, mask=None):
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)
        
        # Take the output corresponding to the last time step
        out = self.fc(x[:, -1, :])
        return out.squeeze(-1) 