import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class TransModelAccOnly(nn.Module):
    def __init__(self,
                acc_frames=128,
                num_classes=1, 
                num_heads=4, 
                acc_coords=4,  # Changed to 4 for x,y,z,smv
                num_layer=2, 
                norm_first=True, 
                embed_dim=32, 
                activation='relu',
                **kwargs):
        super().__init__()
        
        self.data_shape = (acc_frames, acc_coords)
        self.length = self.data_shape[0]
        self.input_proj = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=8, stride=1, padding='same'), 
            nn.BatchNorm1d(embed_dim)
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,  
            activation=activation, 
            dim_feedforward=embed_dim*2, 
            nhead=num_heads,
            dropout=0.5
        )
        
        # Define TransformerEncoderWAttention equivalent
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, 
            num_layers=num_layer, 
            norm=nn.LayerNorm(embed_dim)
        )

        self.output = nn.Linear(embed_dim, num_classes)
        self.temporal_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, acc_data):
        # Process accelerometer data
        x = rearrange(acc_data, 'b l c -> b c l')
        x = self.input_proj(x)
        x = rearrange(x, 'b c l -> l b c')
        x = self.encoder(x)
        x = rearrange(x, 'l b c -> b l c')

        x = self.temporal_norm(x)
        feature = x
        
        # Final processing and output
        x = rearrange(x, 'b f c -> b c f')
        x = F.avg_pool1d(x, kernel_size=x.shape[-1], stride=1)
        x = rearrange(x, 'b c f -> b (c f)')
        x = self.output(x)
        
        return x, feature
