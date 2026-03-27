from utils.imports_statiques import *


# ---------------------------------------------------------
# Norm
# ---------------------------------------------------------


class LayerNorm1D(nn.Module):
    """
    Channel-wise layer normalization for 1D feature maps (B, C, L).
    Normalizes each channel independently across the temporal dimension.
    """
    def __init__(self, dim: int):
        """
        :param dim: [int]; number of input channels (C).
        """
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))  # learnable scale per channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [torch.Tensor]; shape (B, C, L), input sequence.
        :return: [torch.Tensor]; same shape, normalized per channel.
        """
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var  = torch.var(x, dim=2, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=2, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    """
    Applies channel-wise LayerNorm1D before a given submodule.
    Useful in attention or residual blocks for better stability.
    """
    def __init__(self, dim: int, fn: nn.Module):
        """
        :param dim: [int]; number of input channels (C).
        :param fn: [nn.Module]; submodule to apply after normalization.
        """
        super().__init__()
        self.norm = LayerNorm1D(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [torch.Tensor]; shape (B, C, L).
        :return: [torch.Tensor]; output of fn(norm(x)).
        """
        return self.fn(self.norm(x))


class Residual(nn.Module):
    """
    Simple residual wrapper: computes y = f(x) + x.
    Helps gradient flow and stabilizes training in deep networks.
    """
    def __init__(self, fn: nn.Module):
        """
        :param fn: [nn.Module]; submodule to wrap with a residual connection.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [torch.Tensor]; input tensor.
        :return: [torch.Tensor]; output = f(x) + x.
        """
        return self.fn(x) + x