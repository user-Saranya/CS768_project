import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


# ---------------- Transformer Block ---------------- #
class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, dropout):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Multi-head attention (graph-aware)
        self.attn = GATConv(
            in_dim,
            hidden_dim // num_heads,
            num_heads=num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            residual=False,
            activation=None
        )

        # Residual + Norm
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, x):
        # --- Attention ---
        h = self.attn(graph, x)  # [N, heads, dim]
        h = h.flatten(1)         # [N, hidden_dim]

        # Residual + Norm
        x = self.norm1(x + self.dropout(h))

        # --- Feedforward ---
        h2 = self.ffn(x)

        # Residual + Norm
        x = self.norm2(x + self.dropout(h2))

        return x


# ---------------- Full Transformer ---------------- #
class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, out_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, x):
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout(h)

        # Transformer layers
        for layer in self.layers:
            h = layer(graph, h)

        # Output
        out = self.output_layer(h)

        return out
