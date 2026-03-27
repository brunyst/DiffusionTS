from utils.imports_statiques import *


# ---------------------------------------------------------
# CausalConv1D
# ---------------------------------------------------------


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=0, bias=bias)
        self.pad_left = (kernel_size - 1)

    def forward(self, x):
        x = F.pad(x, (self.pad_left, 0), mode='constant', value=0.0)
        return super().forward(x)


# ---------------------------------------------------------
# Upsample + Conv
# ---------------------------------------------------------

class UpConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, h):
        return self.conv(self.up(h))


class ConvGNAct(nn.Module):
    """
    1D convolutional block with GroupNorm and SiLU activation.
    Uses kernel_size=3 and padding=1 to preserve sequence length.
    """
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8, k: int = 3, s: int = 1, p: int = 1):
        """
        :param in_ch: [int]; number of input channels.
        :param out_ch: [int]; number of output channels.
        :param groups: [int]; number of groups for GroupNorm (default: 8).
        :param k: [int]; kernel size of the convolution (default: 3).
        :param s: [int]; stride (default: 1).
        :param p: [int]; padding (default: 1).
        """
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.gn   = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act  = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [torch.Tensor]; input tensor (B, C_in, L).
        :return: [torch.Tensor]; output tensor (B, C_out, L).
        """
        return self.act(self.gn(self.conv(x)))


# ---------------------------------------------------------
# QuadraticConv1D
# ---------------------------------------------------------

def _unfold1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    x2 = x.unsqueeze(2)  # (B, Cin, 1, L)
    p = F.unfold(
        x2,
        kernel_size=(1, kernel_size),
        dilation=(1, 1),
        padding=(0, padding),
        stride=(1, stride),
    )  # (B, Cin*kernel_size, L_out)
    return p


class CrossQuadraticConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        rank: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rank = rank

        D = in_channels * kernel_size
        self.u = nn.Parameter(torch.randn(out_channels, rank, D) * (1.0 / (D ** 0.5)))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Cin, L)
        B, Cin, _ = x.shape
        if Cin != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {Cin}")

        p = _unfold1d(x, self.kernel_size, self.stride, self.padding)  # (B, D, L_out)

        # up = u^T p for each output channel and rank
        # (B, D, L_out) x (Cout, R, D) -> (B, Cout, R, L_out)
        up = torch.einsum("b d n, o r d -> b o r n", p, self.u)

        # (u^T p)^2 term
        term_sq = up.pow(2)

        # subtract diagonal squares: sum_i u_i^2 p_i^2
        diag = torch.einsum("b d n, o r d -> b o r n", p.pow(2), self.u.pow(2))

        y = (term_sq - diag).sum(dim=2)  # sum over rank R -> (B, Cout, L_out)

        if self.bias is not None:
            y = y + self.bias[None, :, None]

        return y


class QuadraticConv1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        rank: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rank = rank

        D = in_channels * kernel_size
        self.u = nn.Parameter(0.01 * torch.randn(out_channels, rank, D))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cin, _ = x.shape
        if Cin != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {Cin}")

        p = _unfold1d(x, self.kernel_size, self.stride, self.padding)  # (B, D, L_out)

        up = torch.einsum("b d n, o r d -> b o r n", p, self.u)  # (B, Cout, R, L_out)
        y = up.pow(2).sum(dim=2)  # (B, Cout, L_out)

        if self.bias is not None:
            y = y + self.bias[None, :, None]

        return y
