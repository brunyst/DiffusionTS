from utils.imports_statiques import *

# ---------------------------------------------------------
# Attention Block 1D
# ---------------------------------------------------------


class AttentionBlock1D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=False  # PyTorch: expects (L, B, C)
        )

        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.SiLU(),
            nn.Linear(4 * channels, channels),
        )

    def forward(self, x):
        B, C, L = x.shape

        # Attention
        h = x.permute(2, 0, 1) # reshape to (L, B, C)
        h_norm = self.norm1(h)
        h_attn, _ = self.attn(h_norm, h_norm, h_norm)
        h = h + h_attn

        # Feed-forward
        h_norm2 = self.norm2(h)
        h_ff = self.mlp(h_norm2)
        h = h + h_ff

        return h.permute(1, 2, 0)











class LinearAttention1D(nn.Module):
    """
    1D Linear Attention mechanism — reduces computational cost to O(L * C).
    Applies softmax over sequence length for keys and over channels for queries.
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        """
        :param dim: [int]; input channel dimension.
        :param heads: [int]; number of attention heads (default: 4).
        :param dim_head: [int]; per-head dimensionality (default: 32).
        """
        super().__init__()
        self.heads = heads
        hidden = heads * dim_head
        self.to_qkv = nn.Conv1d(dim, hidden * 3, kernel_size=1, bias=False)
        self.out = nn.Sequential(nn.Conv1d(hidden, dim, kernel_size=1), LayerNorm1D(dim))
        self.scale = dim_head ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [torch.Tensor]; input tensor of shape (B, C, L).
        :return: [torch.Tensor]; output tensor of shape (B, C, L).
        """
        b, c, n = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)                 # (B, H*C', L) * 3
        q = rearrange(q, "b (h d) n -> b h d n", h=self.heads)   # (B, H, D, L)
        k = rearrange(k, "b (h d) n -> b h d n", h=self.heads)
        v = rearrange(v, "b (h d) n -> b h d n", h=self.heads)

        q = q.softmax(dim=-2) * self.scale      # softmax over D (channels)
        k = k.softmax(dim=-1)                   # softmax over L (sequence length)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)  # (B, H, D, D)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q) # (B, H, D, L)
        out = rearrange(out, "b h d n -> b (h d) n")
        return self.out(out)


class Attention1D(nn.Module):
    """
    Full (scaled dot-product) self-attention for 1D sequences — cost O(L²).
    Used in bottleneck layers to capture long-range dependencies.
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        """
        :param dim: [int]; input channel dimension.
        :param heads: [int]; number of attention heads (default: 4).
        :param dim_head: [int]; per-head dimensionality (default: 32).
        """
        super().__init__()
        self.heads = heads
        hidden = heads * dim_head
        self.to_qkv = nn.Conv1d(dim, hidden * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(hidden, dim, kernel_size=1)
        self.scale = dim_head ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [torch.Tensor]; input tensor (B, C, L).
        :return: [torch.Tensor]; output tensor (B, C, L).
        """
        b, c, n = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = rearrange(q, "b (h d) n -> b h d n", h=self.heads)
        k = rearrange(k, "b (h d) n -> b h d n", h=self.heads)
        v = rearrange(v, "b (h d) n -> b h d n", h=self.heads)

        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)   # (B, H, L, L)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v) # (B, H, L, D)
        out = rearrange(out, "b h n d -> b (h d) n")
        return self.to_out(out)
