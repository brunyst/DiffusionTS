from utils.imports_statiques import *


# ---------------------------------------------------------
# Gaussian Fourier Features
# ---------------------------------------------------------


class GaussianFourierProjection(nn.Module):
    """
    This module maps scalar inputs (e.g., time steps) to a higher-dimensional
    space using fixed random Fourier features, based on a Gaussian distribution.
    """

    def __init__(self, embed_dim, scale=30.):
        """
        Initializes the GaussianFourierProjection module.

        :param embed_dim: [int]; output dimension of the encoding (must be divisible by 2).
        :param scale: [float]; std used to scale sampled Gaussian frequencies; controls frequency range.
        """
        super().__init__()
        random_weights = torch.randn(embed_dim // 2)
        scaled_weights = random_weights * scale
        self.W = nn.Parameter(scaled_weights, requires_grad=False)

    def forward(self, x):
        """
        Applies the random Fourier feature encoding to the input tensor.

        :param x: [torch.Tensor]; shape (B,), scalar values to encode (e.g., time steps in [0, 1]).
        :return: [torch.Tensor]; shape (B, embed_dim) = concat[sin, cos].
        """
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        x_sin = torch.sin(x_proj)
        x_cos = torch.cos(x_proj)
        return torch.cat([x_sin, x_cos], dim=-1)


# ---------------------------------------------------------
# Dense1D
# ---------------------------------------------------------


class Dense1D(nn.Module):
    """
    A fully connected (linear) layer that reshapes its output to match
    a 1D feature map format, typically used to inject time embeddings into 1D CNNs.
    """

    def __init__(self, input_dim, output_dim):
        """
        Initializes the Dense1D layer.

        :param input_dim: [int]; number of input features.
        :param output_dim: [int]; number of output features (channels).
        """
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Applies a linear transformation and reshapes to a 3D tensor.

        :param x: [torch.Tensor]; shape (B, input_dim).
        :return: [torch.Tensor]; shape (B, output_dim, 1).
        """
        out = self.dense(x)    # (batch_size, output_dim)    
        return out[:, :, None] # (batch_size, output_dim, 1)


# ---------------------------------------------------------
# FiLM layer (time conditioning)
# ---------------------------------------------------------


class FiLM1D(nn.Module):
    """ FiLM conditioning: gamma(t), beta(t) """
    def __init__(self, embed_dim, channels):
        super().__init__()
        self.gamma_dense = nn.Linear(embed_dim, channels)
        self.beta_dense  = nn.Linear(embed_dim, channels)

    def forward(self, h, embed):
        # embed : (B, embed_dim)
        gamma = self.gamma_dense(embed)[:, :, None]   # (B,C,1)
        beta  = self.beta_dense(embed)[:, :, None]    # (B,C,1)
        return h * (1 + gamma) + beta