from utils.imports_statiques import *
from networks.time_embedding import *


# ---------------------------------------------------------
# Residual block
# ---------------------------------------------------------


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        gn_groups = min(32, max(4, channels // 4))

        self.scale = 0.1

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(gn_groups, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(gn_groups, channels)
        )

    def forward(self, x):
        return x + self.scale * self.block(x)


# ---------------------------------------------------------
# Residual block with FiLM
# ---------------------------------------------------------


class ResidualBlockFiLM1D(nn.Module):
    def __init__(self, channels, kernel_size, embed_dim, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        gn = min(32, max(4, channels // 4))

        self.film = FiLM1D(embed_dim, channels)
        self.scale = 0.1

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(gn, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(gn, channels)
        )

    def forward(self, h, embed):
        h = self.film(h, embed)
        return h + self.scale * self.block(h)


class ResnetBlock1D(nn.Module):
    """
    1D ResNet block with optional time conditioning (FiLM modulation).
    If `time_emb_dim` is provided, a small MLP generates (scale, shift)
    vectors applied to the activations of the first block.
    """
    def __init__(self, dim_in: int, dim_out: int, time_emb_dim: int | None = None, groups: int = 8):
        """
        :param dim_in: [int]; number of input channels.
        :param dim_out: [int]; number of output channels.
        :param time_emb_dim: [int | None]; dimension of time embedding (if FiLM conditioning is used).
        :param groups: [int]; number of groups for GroupNorm (default: 8).
        """
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None else None
        )
        self.block1 = ConvGNAct(dim_in, dim_out, groups)
        self.block2 = ConvGNAct(dim_out, dim_out, groups)
        self.res = nn.Conv1d(dim_in, dim_out, kernel_size=1) if dim_in != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param x: [torch.Tensor]; input tensor of shape (B, C_in, L).
        :param t_emb: [torch.Tensor | None]; time embedding (B, time_emb_dim).
        :return: [torch.Tensor]; output tensor (B, C_out, L).
        """
        scale_shift = None
        if self.mlp is not None and t_emb is not None:
            ss = self.mlp(t_emb)                 # (B, 2*dim_out)
            ss = rearrange(ss, "b c -> b c 1")   # (B, 2*dim_out, 1)
            scale_shift = ss.chunk(2, dim=1)     # (scale, shift)

        h = self.block1(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            h = h * (scale + 1) + shift

        h = self.block2(h)
        return h + self.res(x)