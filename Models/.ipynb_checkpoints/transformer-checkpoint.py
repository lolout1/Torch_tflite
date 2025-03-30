import torch
from torch import nn
from torch.nn import Linear, LayerNorm, TransformerEncoderLayer
import torch.nn.functional as F
import math

class ExportCompatibleEncoder(nn.Module):
    """
    Export-compatible encoder that mimics the original transformer's behavior
    without storing attention weights.
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for layer in self.layers:
            # Process through attention (like original implementation)
            attn_output, _ = layer.self_attn(
                output, output, output,
                attn_mask=mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True
            )

            # Process through full layer
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransModel(nn.Module):
    """
    TransModel fixed to handle tensor dimensions correctly for pooling while
    preserving the original model's architecture and behavior.
    """
    def __init__(self,
                 mocap_frames=64,
                 num_joints=32,
                 acc_frames=64,
                 num_classes=2,
                 num_heads=4,
                 acc_coords=3,
                 av=False,
                 num_layers=4,
                 norm_first=True,
                 embed_dim=64,
                 activation='relu',
                 **kwargs):
        super().__init__()
        self.data_shape = (acc_frames, acc_coords)
        self.length = self.data_shape[0]
        size = self.data_shape[1]

        # Input projection with explicit padding
        self.input_proj = nn.Sequential(
            nn.Conv1d(size, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(embed_dim*2, embed_dim, kernel_size=3, stride=1, padding=1)
        )

        # Transformer encoder layer with embed_dim
        self.encoder_layer = TransformerEncoderLayer(
            d_model=64,  # Match embed_dim for compatibility
            nhead=num_heads,
            dim_feedforward=32,
            dropout=0.5,
            activation=activation
        )

        # Use our export-compatible encoder
        self.encoder = ExportCompatibleEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )

        # Linear layers for classification
        self.ln1 = nn.Linear(embed_dim, 32)
        self.ln2 = nn.Linear(32, 16)
        self.drop2 = nn.Dropout(p=0.5)
        self.output = Linear(16, num_classes)
        nn.init.normal_(self.output.weight, 0, math.sqrt(2. / num_classes))

    def forward(self, acc_data):
        """Forward pass with correct tensor handling for pooling"""
        b, l, c = acc_data.shape

        # Step 1: Transform to channels-first for Conv1d
        x = acc_data.transpose(1, 2)  # (b, c, l)

        # Step 2: Apply convolutional layers
        x = self.input_proj(x)  # (b, embed_dim, l)

        # Step 3: Transform for transformer - sequence first
        x = x.permute(2, 0, 1)  # (l, b, embed_dim)

        # Step 4: Apply transformer encoder
        x = self.encoder(x)  # (l, b, embed_dim)

        # Step 5: Reshape for pooling - CRITICAL FIX HERE
        # Need to go from (l, b, embed_dim) to (b, embed_dim, l)
        x = x.permute(1, 2, 0)  # This gives (b, embed_dim, l)

        # Step 6: Now we can correctly apply pooling along the sequence dimension
        # kernel_size should be the sequence length (l)
        x = F.avg_pool1d(x, kernel_size=x.size(2), stride=1)

        # Step 7: Squeeze the pooled dimension
        x = x.squeeze(2)  # (b, embed_dim)

        # Step 8: Classification head
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = self.output(x)

        return x

