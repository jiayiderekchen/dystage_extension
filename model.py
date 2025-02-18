import torch
import torch.nn as nn
import math

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, n_heads=16, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Adjust input_dim to be divisible by n_heads
        self.n_heads = n_heads
        self.adjusted_dim = math.ceil(input_dim / n_heads) * n_heads
        
        # Add a linear projection if input_dim needs adjustment
        self.input_projection = nn.Linear(input_dim, self.adjusted_dim) if input_dim != self.adjusted_dim else nn.Identity()
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.adjusted_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(self.adjusted_dim, 1)
        
    def forward(self, x, mask=None):
        # Handle NaN values in input
        x = torch.nan_to_num(x, nan=0.0)
        
        # Project input if necessary
        x = self.input_projection(x)
        # Add sequence dimension if input is 2D
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)
        
        # Take the output corresponding to the last time step
        out = self.fc(x[:, -1, :])
        return out.squeeze(-1) 