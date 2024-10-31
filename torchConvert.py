import torch
from torch import nn
from typing import Dict, Tuple
from torch.nn import Linear, LayerNorm, TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math
import ai_edge_torch  # install using pip install ai-edge-torch


class TransformerEncoderWAttention(nn.TransformerEncoder):
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        # Make sure there are no mutable state changes by not storing attention weights
        for layer in self.layers:
            
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output

class TransModel(nn.Module):
    def __init__(self,
                 mocap_frames=128,
                 num_joints=32,
                 acc_frames=128,
                 num_classes=2,     
                 num_heads=2,
                 acc_coords=4,
                 av=False,
                 num_layer=2,
                 norm_first=True,
                 embed_dim=32,     
                 activation='relu',
                 **kwargs):
        super().__init__()
        self.data_shape = (acc_frames, acc_coords)
        self.length = self.data_shape[0]  # 128
        size = self.data_shape[1]         # 4

       
        self.input_proj = nn.Sequential(
            nn.Conv1d(size, embed_dim, kernel_size=3, stride=1, padding='same'),          
            nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=3, stride=1, padding='same'), 
            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=3, stride=1, padding='same')  
        )

        
        self.encoder_layer = TransformerEncoderLayer(
            d_model=self.length,          # 128
            nhead=num_heads,              # 2
            dim_feedforward=32,
            activation=activation,
            dropout=0.5,
            batch_first=False,
            norm_first=norm_first
        )

        
        self.encoder = TransformerEncoderWAttention(
            encoder_layer=self.encoder_layer,
            num_layers=num_layer,
            norm=nn.LayerNorm(embed_dim)  # LayerNorm over embed_dim (32)
        )

        
        self.ln1 = nn.Linear(self.length, 32)  # 128 -> 32
        self.ln2 = nn.Linear(32, 16)
        self.drop2 = nn.Dropout(p=0.5)
        self.output = Linear(16, num_classes)  # 16 -> 2
        nn.init.normal_(self.output.weight, 0, math.sqrt(2. / num_classes))

    def forward(self, acc_data, skl_data):
        b, l, c = acc_data.shape  # b, 128, 4
        x = rearrange(acc_data, 'b l c -> b c l')  # [b, 4, 128]
        x = self.input_proj(x)                     # [b, 32, 128]
        x = rearrange(x, 'b c l -> c b l')         # [32, b, 128]
        x = self.encoder(x)                        # [32, b, 128]
        x = rearrange(x, 'c b l -> b l c')         # [b, 128, 32]

        
        x = F.avg_pool1d(x, kernel_size=x.shape[-1], stride=1)  # [b, 128, 1]
        x = x.view(b, -1)                          # [b, 128]
        x = F.relu(self.ln1(x))                    # [b, 32]
        x = self.drop2(x)
        x = F.relu(self.ln2(x))                    # [b, 16]
        x = self.output(x)                         # [b, num_classes]
        return x


def load_and_convert_model(pth_file_path, tflite_file_path):
    
    model = TransModel()
    model.eval()

    
    checkpoint = torch.load(pth_file_path, map_location='cpu')

    
    model.load_state_dict(checkpoint, strict=True)

    
    class ModelWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, acc_data):
            skl_data_dummy = torch.zeros(acc_data.shape[0], 128, 32, 3)
            return self.model(acc_data, skl_data_dummy)

    wrapped_model = ModelWrapper()
    wrapped_model.eval()
    wrapped_model.to('cpu')

    
    acc_data_sample = torch.randn(1, 128, 4)

   
   
    edge_model = ai_edge_torch.convert(
        wrapped_model,
        (acc_data_sample,)
    )

    
    edge_model.export(tflite_file_path)
    print(f"Model successfully converted to TFLite format and saved as '{tflite_file_path}'.")

if __name__ == "__main__":
    # Path to your saved .pth file
    pth_file_path = "ttfWeights.pth"  
    tflite_file_path = "transmodel_converted.tflite"

    
    load_and_convert_model(pth_file_path, tflite_file_path)
