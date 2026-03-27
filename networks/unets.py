from utils.imports_statiques import *
from networks.time_embedding import *
from networks.conv import *
from networks.norm import *
from networks.residual import *
from networks.attention import *


# ---------------------------------------------------------
# ScoreNet1D
# ---------------------------------------------------------

class ScoreNet1D_v1_2channels(nn.Module):

    def __init__(
        self,
        perturbation_kernel_std,
        channels=(32, 64),          # <- 2 niveaux seulement
        embed_dim=256,
        kernel_size=21,
        in_channels=1,
        out_channels=1,
        nb_groups=None,
        nb_heads=None,
        dim_head=None
    ):
        super().__init__()
        assert len(channels) == 2, "channels must be length 2, e.g. (32, 64)"

        self.perturbation_kernel_std = perturbation_kernel_std

        # Gaussian random feature embedding of time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Encoding path (downsampling)
        self.conv1 = nn.Conv1d(in_channels, channels[0], kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.dense1 = Dense1D(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, channels[0])

        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size, stride=2, padding=kernel_size//2, bias=False)
        self.dense2 = Dense1D(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, channels[1])

        # Decoding path (upsampling)
        self.tconv2 = nn.ConvTranspose1d(channels[1], channels[0], kernel_size, stride=2, padding=kernel_size//2, output_padding=1, bias=False)
        self.dense7 = Dense1D(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, channels[0])

        self.tconv1 = nn.ConvTranspose1d(channels[0] * 2, out_channels, kernel_size, padding=kernel_size//2, stride=1)

        # Swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        embed = self.act(self.embed(t))

        # Encoder
        h1 = self.act(self.conv1(x) + self.dense1(embed))
        #h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))

        # Decoder
        h = self.act(self.tgnorm2(self.tconv2(h2) + self.dense7(embed)))
        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h / self.perturbation_kernel_std(t)[:, None, None]
        return h



class ScoreNet1D_v1_quadra(nn.Module):
    """
    A time-dependent score-based model built upon a 1D U-Net architecture.

    This model takes as input a noisy 1D sample x(t) and a time step t,
    and learns the score (i.e., the gradient of the log-density) using a
    deep neural network conditioned on time via Fourier embeddings.
    """

    def __init__(
        self,
        perturbation_kernel_std,
        channels=(32, 64, 128, 256),
        embed_dim=256,
        kernel_size=21,
        in_channels=1,
        out_channels=1,
        nb_groups=None,
        nb_heads=None,
        dim_head=None
    ):
        """
        Initializes the ScoreNet1D model. ~ 1 500 000 params

        :param perturbation_kernel_std: [Callable]; t ↦ σ(t), the std of the perturbation kernel p_{0t}(X_t | X_0).
        :param channels: [Tuple[int]]; number of channels at each stage of the U-Net from shallow to deep.
        :param embed_dim: [int]; dimensionality of the Fourier-based time embedding.
        :param padding: [int]; convolution padding (commonly 1 with kernel_size=3 to preserve length).
        """
        super().__init__()

        self.perturbation_kernel_std = perturbation_kernel_std

        # Gaussian random feature embedding of time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Encoding path (downsampling)
        self.conv1 = nn.Conv1d(in_channels, channels[0], kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.dense1 = Dense1D(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, channels[0])

        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size, stride=2, padding=kernel_size//2, bias=False)
        self.dense2 = Dense1D(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, channels[1])

        self.conv3 = QuadraticConv1D(channels[1], channels[2], kernel_size, stride=2, padding=kernel_size//2, rank=4, bias=False)
        self.dense3 = Dense1D(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, channels[2])

        self.conv4 = QuadraticConv1D(channels[2], channels[3], kernel_size, stride=2, padding=kernel_size//2, rank=4, bias=False)
        self.dense4 = Dense1D(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, channels[3])

        # Decoding path (upsampling)
        self.tconv4 = nn.ConvTranspose1d(channels[3], channels[2], kernel_size, stride=2, padding=kernel_size//2, output_padding=1, bias=False)
        self.dense5 = Dense1D(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, channels[2])

        self.tconv3 = nn.ConvTranspose1d(channels[2] * 2, channels[1], kernel_size, stride=2, padding=kernel_size//2, output_padding=1, bias=False)
        self.dense6 = Dense1D(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, channels[1])

        self.tconv2 = nn.ConvTranspose1d(channels[1] * 2, channels[0], kernel_size, stride=2, padding=kernel_size//2, output_padding=1, bias=False)
        self.dense7 = Dense1D(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, channels[0])

        self.tconv1 = nn.ConvTranspose1d(channels[0] * 2, out_channels, kernel_size, padding=kernel_size//2, stride=1)

        # Swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        """
        Forward pass of the time-conditional 1D score network.

        :param x: [torch.Tensor]; shape (B, 1, T), noisy input at time t.
        :param t: [torch.Tensor]; shape (B,), time steps used to condition the network.
        :return: [torch.Tensor]; shape (B, 1, T), score estimate ∇_X log p_t(X) at the input resolution.
        """
        embed = self.act(self.embed(t))

        # Encoder
        h1 = self.act(self.conv1(x) + self.dense1(embed))
        #h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

        # Decoder
        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)))
        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h / self.perturbation_kernel_std(t)[:, None, None]
        return h



class ScoreNet1D_v1(nn.Module):
    """
    A time-dependent score-based model built upon a 1D U-Net architecture.

    This model takes as input a noisy 1D sample x(t) and a time step t,
    and learns the score (i.e., the gradient of the log-density) using a
    deep neural network conditioned on time via Fourier embeddings.
    """

    def __init__(
        self,
        perturbation_kernel_std,
        channels=(32, 64, 128, 256),
        embed_dim=256,
        kernel_size=21,
        in_channels=1,
        out_channels=1,
        nb_groups=None,
        nb_heads=None,
        dim_head=None
    ):
        """
        Initializes the ScoreNet1D model. ~ 1 500 000 params

        :param perturbation_kernel_std: [Callable]; t ↦ σ(t), the std of the perturbation kernel p_{0t}(X_t | X_0).
        :param channels: [Tuple[int]]; number of channels at each stage of the U-Net from shallow to deep.
        :param embed_dim: [int]; dimensionality of the Fourier-based time embedding.
        :param padding: [int]; convolution padding (commonly 1 with kernel_size=3 to preserve length).
        """
        super().__init__()

        self.perturbation_kernel_std = perturbation_kernel_std

        # Gaussian random feature embedding of time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Encoding path (downsampling)
        self.conv1 = nn.Conv1d(in_channels, channels[0], kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.dense1 = Dense1D(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, channels[0])

        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size, stride=2, padding=kernel_size//2, bias=False)
        self.dense2 = Dense1D(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(nb_groups, channels[1])

        self.conv3 = nn.Conv1d(channels[1], channels[2], kernel_size, stride=2, padding=kernel_size//2, bias=False)
        self.dense3 = Dense1D(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(nb_groups, channels[2])

        self.conv4 = nn.Conv1d(channels[2], channels[3], kernel_size, stride=2, padding=kernel_size//2, bias=False)
        self.dense4 = Dense1D(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(nb_groups, channels[3])

        # Decoding path (upsampling)
        self.tconv4 = nn.ConvTranspose1d(channels[3], channels[2], kernel_size, stride=2, padding=kernel_size//2, output_padding=1, bias=False)
        self.dense5 = Dense1D(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(nb_groups, channels[2])

        self.tconv3 = nn.ConvTranspose1d(channels[2] * 2, channels[1], kernel_size, stride=2, padding=kernel_size//2, output_padding=1, bias=False)
        self.dense6 = Dense1D(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(nb_groups, channels[1])

        self.tconv2 = nn.ConvTranspose1d(channels[1] * 2, channels[0], kernel_size, stride=2, padding=kernel_size//2, output_padding=1, bias=False)
        self.dense7 = Dense1D(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(nb_groups, channels[0])

        self.tconv1 = nn.ConvTranspose1d(channels[0] * 2, out_channels, kernel_size, padding=kernel_size//2, stride=1)

        # Swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        """
        Forward pass of the time-conditional 1D score network.

        :param x: [torch.Tensor]; shape (B, 1, T), noisy input at time t.
        :param t: [torch.Tensor]; shape (B,), time steps used to condition the network.
        :return: [torch.Tensor]; shape (B, 1, T), score estimate ∇_X log p_t(X) at the input resolution.
        """
        embed = self.act(self.embed(t))

        # Encoder
        h1 = self.act(self.conv1(x) + self.dense1(embed))
        #h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

        # Decoder
        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)))
        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h / self.perturbation_kernel_std(t)[:, None, None]
        return h


class ScoreNet1D_v1_adapt(nn.Module):
    """
    Adaptive version of ScoreNet1D_v1 supporting any number of resolution levels.
    Reproduces exactly the same logic as ScoreNet1D_v1 when len(channels) == 4.

    Architecture:
      - Level 1 : Conv1d stride=1, GroupNorm(4, ·)  [gnorm skipped in forward, same as v1]
      - Levels 2..N : Conv1d stride=2, GroupNorm(nb_groups, ·)
      - Decoder mirrors encoder with skip connections (U-Net style)
    """

    def __init__(
        self,
        perturbation_kernel_std,
        channels=(32, 64, 128, 256),
        embed_dim=256,
        kernel_size=21,
        in_channels=1,
        out_channels=1,
        nb_groups=8,
        nb_heads=None,
        dim_head=None,
    ):
        super().__init__()
        assert len(channels) >= 2, "channels must have at least 2 levels"
        self.perturbation_kernel_std = perturbation_kernel_std
        self.nb_levels = len(channels)

        # Time embedding
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc_convs  = nn.ModuleList()
        self.enc_denses = nn.ModuleList()
        self.enc_gnorms = nn.ModuleList()

        # Level 1 : stride=1, GroupNorm(4, ·)
        self.enc_convs.append(nn.Conv1d(in_channels, channels[0], kernel_size, stride=1, padding=kernel_size // 2, bias=False))
        self.enc_denses.append(Dense1D(embed_dim, channels[0]))
        self.enc_gnorms.append(nn.GroupNorm(4, channels[0]))

        # Levels 2..N : stride=2, GroupNorm(nb_groups, ·)
        for i in range(1, self.nb_levels):
            self.enc_convs.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size, stride=2, padding=kernel_size // 2, bias=False))
            self.enc_denses.append(Dense1D(embed_dim, channels[i]))
            self.enc_gnorms.append(nn.GroupNorm(nb_groups, channels[i]))

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec_tconvs = nn.ModuleList()
        self.dec_denses = nn.ModuleList()
        self.dec_gnorms = nn.ModuleList()

        # Stage 0 (deepest) : channels[-1] → channels[-2], no skip input yet
        self.dec_tconvs.append(nn.ConvTranspose1d(channels[-1], channels[-2], kernel_size, stride=2, padding=kernel_size // 2, output_padding=1, bias=False))
        self.dec_denses.append(Dense1D(embed_dim, channels[-2]))
        self.dec_gnorms.append(nn.GroupNorm(nb_groups, channels[-2]))

        # Stages 1..N-2 : skip connection doubles input channels
        for k in range(1, self.nb_levels - 1):
            in_ch  = channels[self.nb_levels - 1 - k] * 2
            out_ch = channels[self.nb_levels - 2 - k]
            self.dec_tconvs.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=2, padding=kernel_size // 2, output_padding=1, bias=False))
            self.dec_denses.append(Dense1D(embed_dim, out_ch))
            self.dec_gnorms.append(nn.GroupNorm(nb_groups, out_ch))

        # Final conv : channels[0]*2 → out_channels, stride=1
        self.final_tconv = nn.ConvTranspose1d(channels[0] * 2, out_channels, kernel_size, padding=kernel_size // 2, stride=1)

        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        embed = self.act(self.embed(t))

        # ── Encoder : collect skip connections ───────────────────────────────
        skips = []
        h = x
        for i in range(self.nb_levels):
            if i == 0:
                # Level 1 : gnorm skipped (same convention as ScoreNet1D_v1)
                h = self.act(self.enc_convs[i](h) + self.enc_denses[i](embed))
            else:
                h = self.act(self.enc_gnorms[i](self.enc_convs[i](h) + self.enc_denses[i](embed)))
            skips.append(h)

        # ── Decoder ──────────────────────────────────────────────────────────
        h = skips[-1]  # start from bottleneck

        # Stage 0 : upsample bottleneck (no skip yet)
        h = self.act(self.dec_gnorms[0](self.dec_tconvs[0](h) + self.dec_denses[0](embed)))

        # Stages 1..N-2 : cat with encoder skip (reverse order, excluding bottleneck)
        for k in range(1, self.nb_levels - 1):
            skip = skips[self.nb_levels - 1 - k]
            h = self.act(self.dec_gnorms[k](self.dec_tconvs[k](torch.cat([h, skip], dim=1)) + self.dec_denses[k](embed)))

        # Final : cat with shallowest encoder output
        h = self.final_tconv(torch.cat([h, skips[0]], dim=1))

        h = h / self.perturbation_kernel_std(t)[:, None, None]
        return h


class ScoreNet1D_v2(nn.Module):
    """
    A time-dependent score-based model built upon a 1D U-Net architecture.

    This model takes as input a noisy 1D sample x(t) and a time step t,
    and learns the score (i.e., the gradient of the log-density) using a
    deep neural network conditioned on time via Fourier embeddings.
    """

    def __init__(
        self,
        perturbation_kernel_std,
        channels=(32, 64, 128, 256),
        embed_dim=256,
        kernel_size=21,
        in_channels=1,
        out_channels=1,
        nb_groups=None,
        nb_heads=None,
        dim_head=None
    ):
        """
        Initializes the ScoreNet1D model. ~ 1 500 000 params

        :param perturbation_kernel_std: [Callable]; t ↦ σ(t), the std of the perturbation kernel p_{0t}(X_t | X_0).
        :param channels: [Tuple[int]]; number of channels at each stage of the U-Net from shallow to deep.
        :param embed_dim: [int]; dimensionality of the Fourier-based time embedding.
        :param padding: [int]; convolution padding (commonly 1 with kernel_size=3 to preserve length).
        """
        super().__init__()

        self.perturbation_kernel_std = perturbation_kernel_std

        # Gaussian random feature embedding of time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )   


        # Encoding path (downsampling)
        self.conv1 = nn.Conv1d(1, channels[0], kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.res1 = ResidualBlock1D(channels[0], kernel_size)
        self.dense1 = Dense1D(embed_dim, channels[0])

        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size, stride=2, padding=kernel_size//2, bias=False)
        self.res2 = ResidualBlock1D(channels[1], kernel_size)
        self.dense2 = Dense1D(embed_dim, channels[1])

        self.conv3 = nn.Conv1d(channels[1], channels[2], kernel_size, stride=2, padding=kernel_size//2, bias=False)
        self.res3 = ResidualBlock1D(channels[2], kernel_size)
        self.dense3 = Dense1D(embed_dim, channels[2])

        self.conv4 = nn.Conv1d(channels[2], channels[3], kernel_size, stride=2, padding=kernel_size//2, bias=False)
        self.res4 = ResidualBlock1D(channels[3], kernel_size)
        self.dense4 = Dense1D(embed_dim, channels[3])

        # Decoding path (upsampling)

        self.tconv4 = nn.ConvTranspose1d(channels[3], channels[2], kernel_size, stride=2, padding=kernel_size//2, output_padding=1, bias=False)
        self.resd4 = ResidualBlock1D(channels[2], kernel_size)
        self.dense5 = Dense1D(embed_dim, channels[2])

        self.tconv3 = nn.ConvTranspose1d(channels[2] * 2, channels[1], kernel_size, stride=2, padding=kernel_size//2, output_padding=1, bias=False)
        self.resd3 = ResidualBlock1D(channels[1], kernel_size)
        self.dense6 = Dense1D(embed_dim, channels[1])

        self.tconv2 = nn.ConvTranspose1d(channels[1] * 2, channels[0], kernel_size, stride=2, padding=kernel_size//2, output_padding=1, bias=False)
        self.resd2 = ResidualBlock1D(channels[0], kernel_size)
        self.dense7 = Dense1D(embed_dim, channels[0])

        self.tconv1 = nn.ConvTranspose1d(channels[0] * 2, 1, kernel_size, padding=kernel_size//2, stride=1)

        # Swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        """
        Forward pass of the time-conditional 1D score network.

        :param x: [torch.Tensor]; shape (B, 1, T), noisy input at time t.
        :param t: [torch.Tensor]; shape (B,), time steps used to condition the network.
        :return: [torch.Tensor]; shape (B, 1, T), score estimate ∇_X log p_t(X) at the input resolution.
        """
        embed = self.act(self.embed(t))

        # Encoder
        h1 = self.res1(self.act(self.conv1(x) + self.dense1(embed)))
        h2 = self.res2(self.act(self.conv2(h1) + self.dense2(embed)))
        h3 = self.res3(self.act(self.conv3(h2) + self.dense3(embed)))
        h4 = self.res4(self.act(self.conv4(h3) + self.dense4(embed)))

        # Decoder
        h = self.resd4(self.act(self.tconv4(h4) + self.dense5(embed)))
        h = self.resd3(self.act(self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)))
        h = self.resd2(self.act(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)))
        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h / self.perturbation_kernel_std(t)[:, None, None]
        return h


class ScoreNet1D_v3(nn.Module):
    def __init__(
            self, perturbation_kernel_std,
            channels=(32, 64, 128, 256),
            embed_dim=256,
            kernel_size=21,
            in_channels=1,
            out_channels=1,
            nb_groups=None,
            nb_heads=None,
            dim_head=None
        ):
        super().__init__()

        self.perturbation_kernel_std = perturbation_kernel_std

        # Time embedding
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU()
        )

        C1, C2, C3, C4 = channels

        # Encoder
        self.conv1 = nn.Conv1d(in_channels,  C1, kernel_size, padding=kernel_size//2)
        self.res1  = ResidualBlockFiLM1D(C1, kernel_size, embed_dim)

        self.conv2 = nn.Conv1d(C1, C2, kernel_size, stride=2, padding=kernel_size//2)
        self.res2  = ResidualBlockFiLM1D(C2, kernel_size, embed_dim)

        self.conv3 = nn.Conv1d(C2, C3, kernel_size, stride=2, padding=kernel_size//2)
        self.res3  = ResidualBlockFiLM1D(C3, kernel_size, embed_dim)

        self.conv4 = nn.Conv1d(C3, C4, kernel_size, stride=2, padding=kernel_size//2)
        self.res4  = ResidualBlockFiLM1D(C4, kernel_size, embed_dim)

        # 1×1 projections for skip connections
        self.skip3 = nn.Conv1d(C3, C3, 1)
        self.skip2 = nn.Conv1d(C2, C2, 1)
        self.skip1 = nn.Conv1d(C1, C1, 1)

        # Decoder
        self.up4 = UpConv1D(C4, C3, kernel_size)
        self.resd4 = ResidualBlockFiLM1D(C3, kernel_size, embed_dim)

        self.up3 = UpConv1D(C3, C2, kernel_size)
        self.resd3 = ResidualBlockFiLM1D(C2, kernel_size, embed_dim)

        self.up2 = UpConv1D(C2, C1, kernel_size)
        self.resd2 = ResidualBlockFiLM1D(C1, kernel_size, embed_dim)

        self.final = nn.Conv1d(C1, out_channels, kernel_size, padding=kernel_size//2)


    def forward(self, x, t):
        embed = self.embed(t)

        # Encoder
        h1 = self.res1(self.conv1(x), embed)
        h2 = self.res2(self.conv2(h1), embed)
        h3 = self.res3(self.conv3(h2), embed)
        h4 = self.res4(self.conv4(h3), embed)

        # Decoder
        h = self.resd4(self.up4(h4) + self.skip3(h3), embed)
        h = self.resd3(self.up3(h) + self.skip2(h2), embed)
        h = self.resd2(self.up2(h) + self.skip1(h1), embed)
        h = self.final(h)

        # Normalisation score
        h = h / self.perturbation_kernel_std(t)[:, None, None]
        return h


class ScoreNet1D_v3_v2(nn.Module):
    def __init__(
        self,
        perturbation_kernel_std,
        channels=(32, 64, 128, 256),
        embed_dim=256,
        kernel_size=21,
        in_channels=1,
        out_channels=1,
        nb_groups=None,
        nb_heads=None,
        dim_head=None,
    ):
        super().__init__()

        if len(channels) < 1:
            raise ValueError("channels must have length >= 1")

        self.perturbation_kernel_std = perturbation_kernel_std
        self.kernel_size = int(kernel_size)

        # Time embedding
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

        # -------- Encoder (levels) --------
        # conv: in_ch -> channels[i], with stride=1 for i=0 else stride=2
        # res: ResidualBlockFiLM1D(channels[i])
        self.enc_convs = nn.ModuleList()
        self.enc_res   = nn.ModuleList()

        prev_c = int(in_channels)
        for i, c in enumerate(channels):
            c = int(c)
            stride = 1 if i == 0 else 2
            self.enc_convs.append(
                nn.Conv1d(prev_c, c, self.kernel_size, stride=stride, padding=self.kernel_size // 2)
            )
            self.enc_res.append(ResidualBlockFiLM1D(c, self.kernel_size, embed_dim))
            prev_c = c

        # -------- Skip projections (for all skips except the bottleneck) --------
        # We will add skip from encoder level i to decoder level i (same channel size),
        # so we need 1x1 conv on encoder features for i = 0..L-2
        self.skip_projs = nn.ModuleList([
            nn.Conv1d(int(channels[i]), int(channels[i]), 1)
            for i in range(len(channels) - 1)
        ])

        # -------- Decoder --------
        # For L levels:
        # start at bottleneck channels[-1]
        # up to channels[-2], then [-3], ... , channels[0]
        self.dec_ups = nn.ModuleList()
        self.dec_res = nn.ModuleList()

        for i in range(len(channels) - 1, 0, -1):
            cin  = int(channels[i])
            cout = int(channels[i - 1])
            self.dec_ups.append(UpConv1D(cin, cout, self.kernel_size))
            self.dec_res.append(ResidualBlockFiLM1D(cout, self.kernel_size, embed_dim))

        self.final = nn.Conv1d(int(channels[0]), int(out_channels), self.kernel_size, padding=self.kernel_size // 2)

    def forward(self, x, t):
        embed = self.embed(t)

        # ----- Encoder -----
        hs = []  # store all encoder activations after res blocks
        h = x
        for conv, res in zip(self.enc_convs, self.enc_res):
            h = res(conv(h), embed)
            hs.append(h)

        # hs[-1] is bottleneck, hs[0] is shallow

        # ----- Decoder -----
        h = hs[-1]  # bottleneck (no skip add here)
        # Iterate decoder stages; each stage corresponds to adding skip from encoder level (i-1)
        # We traverse i = L-1 -> 1 in terms of channel indices
        for stage_idx, (up, res) in enumerate(zip(self.dec_ups, self.dec_res)):
            # encoder skip level index corresponding to this stage:
            # first decoder stage uses skip from hs[-2], then hs[-3], ..., hs[0]
            skip_level = (len(hs) - 2) - stage_idx
            skip = self.skip_projs[skip_level](hs[skip_level])

            h = up(h) + skip
            h = res(h, embed)

        h = self.final(h)

        # Normalisation score
        h = h / self.perturbation_kernel_std(t)[:, None, None]
        return h


class ScoreNet1D_v4(nn.Module):
    def __init__(
            self, 
            perturbation_kernel_std,
            channels=(32, 64, 128, 256),
            embed_dim=256,
            kernel_size=21,
            num_heads=4,
            in_channels=1,
            out_channels=1,
            nb_groups=None,
            nb_heads=None,
            dim_head=None
        ):
        super().__init__()

        self.perturbation_kernel_std = perturbation_kernel_std

        # Time embedding
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU()
        )

        C1, C2, C3, C4 = channels

        # ---------------- Encoder ----------------
        self.conv1 = nn.Conv1d(1,  C1, kernel_size, padding=kernel_size//2)
        self.res1  = ResidualBlockFiLM1D(C1, kernel_size, embed_dim)

        self.conv2 = nn.Conv1d(C1, C2, kernel_size, stride=2, padding=kernel_size//2)
        self.res2  = ResidualBlockFiLM1D(C2, kernel_size, embed_dim)

        self.conv3 = nn.Conv1d(C2, C3, kernel_size, stride=2, padding=kernel_size//2)
        self.res3  = ResidualBlockFiLM1D(C3, kernel_size, embed_dim)

        self.conv4 = nn.Conv1d(C3, C4, kernel_size, stride=2, padding=kernel_size//2)
        self.res4  = ResidualBlockFiLM1D(C4, kernel_size, embed_dim)

        # --- Self-Attention at bottleneck ---
        self.attn4 = AttentionBlock1D(C4, num_heads=num_heads)

        # ---------------- Skip projections ----------------
        self.skip3 = nn.Conv1d(C3, C3, 1)
        self.skip2 = nn.Conv1d(C2, C2, 1)
        self.skip1 = nn.Conv1d(C1, C1, 1)

        # ---------------- Decoder ----------------
        self.up4   = UpConv1D(C4, C3, kernel_size)
        self.resd4 = ResidualBlockFiLM1D(C3, kernel_size, embed_dim)

        self.up3   = UpConv1D(C3, C2, kernel_size)
        self.resd3 = ResidualBlockFiLM1D(C2, kernel_size, embed_dim)

        self.up2   = UpConv1D(C2, C1, kernel_size)
        self.resd2 = ResidualBlockFiLM1D(C1, kernel_size, embed_dim)

        self.final = nn.Conv1d(C1, 1, kernel_size, padding=kernel_size//2)


    def forward(self, x, t):
        embed = self.embed(t)

        # ----- Encoder -----
        h1 = self.res1(self.conv1(x), embed)
        h2 = self.res2(self.conv2(h1), embed)
        h3 = self.res3(self.conv3(h2), embed)
        h4 = self.res4(self.conv4(h3), embed)

        # ----- Bottleneck Attention -----
        h4 = self.attn4(h4)

        # ----- Decoder -----
        h = self.resd4(self.up4(h4) + self.skip3(h3), embed)
        h = self.resd3(self.up3(h) + self.skip2(h2), embed)
        h = self.resd2(self.up2(h) + self.skip1(h1), embed)
        h = self.final(h)

        return h / self.perturbation_kernel_std(t)[:, None, None]


class ScoreNet1D_v5(nn.Module):
    def __init__(
        self,
        perturbation_kernel_std,
        channels=(32, 64, 128, 256),
        embed_dim=256,
        kernel_size=15,
        in_channels=1,           # canaux bruts (X), ici 1
        out_channels=1,
        nb_groups=None,
        nb_heads=None,
        dim_head=None,
    ):
        super().__init__()

        self.perturbation_kernel_std = perturbation_kernel_std
        self.feature_extractor = FeatureExtractor()
        first_in_channels = self.feature_extractor.out_channels

        # Time embedding
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Encoding path
        self.conv1 = nn.Conv1d(first_in_channels, channels[0],
                               kernel_size, stride=1,
                               padding=kernel_size//2, bias=False)
        self.dense1 = Dense1D(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, channels[0])

        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size, stride=2,
                               padding=kernel_size//2, bias=False)
        self.dense2 = Dense1D(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, channels[1])

        self.conv3 = nn.Conv1d(channels[1], channels[2], kernel_size, stride=2,
                               padding=kernel_size//2, bias=False)
        self.dense3 = Dense1D(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, channels[2])

        self.conv4 = nn.Conv1d(channels[2], channels[3], kernel_size, stride=2,
                               padding=kernel_size//2, bias=False)
        self.dense4 = Dense1D(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, channels[3])

        # Decoding path
        self.tconv4 = nn.ConvTranspose1d(channels[3], channels[2], kernel_size,
                                         stride=2, padding=kernel_size//2,
                                         output_padding=1, bias=False)
        self.dense5 = Dense1D(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, channels[2])

        self.tconv3 = nn.ConvTranspose1d(channels[2] * 2, channels[1], kernel_size,
                                         stride=2, padding=kernel_size//2,
                                         output_padding=1, bias=False)
        self.dense6 = Dense1D(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, channels[1])

        self.tconv2 = nn.ConvTranspose1d(channels[1] * 2, channels[0], kernel_size,
                                         stride=2, padding=kernel_size//2,
                                         output_padding=1, bias=False)
        self.dense7 = Dense1D(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, channels[0])

        self.tconv1 = nn.ConvTranspose1d(channels[0] * 2, out_channels, kernel_size,
                                         padding=kernel_size//2, stride=1)

        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        """
        x : (B,1,T) niveaux bruités x_t
        t : (B,)
        """
        # 1) features internes
        x = self.feature_extractor(x)   # (B, C_feat, T)

        embed = self.act(self.embed(t))

        # Encoder
        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

        # Decoder
        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)))
        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h / self.perturbation_kernel_std(t)[:, None, None]
        return h
        

class ScoreNet1DCausal(nn.Module):
    """
    A time-dependent score-based model built upon a 1D U-Net architecture.

    This model takes as input a noisy 1D sample x(t) and a time step t,
    and learns the score (i.e., the gradient of the log-density) using a
    deep neural network conditioned on time via Fourier embeddings.
    """

    def __init__(
        self,
        perturbation_kernel_std,
        channels=(32, 64, 128, 256),
        embed_dim=256,
        kernel_size=15
    ):
        """
        Initializes the ScoreNet1D model.

        :param perturbation_kernel_std: [Callable]; t ↦ σ(t), the std of the perturbation kernel p_{0t}(X_t | X_0).
        :param channels: [Tuple[int]]; number of channels at each stage of the U-Net from shallow to deep.
        :param embed_dim: [int]; dimensionality of the Fourier-based time embedding.
        :param padding: [int]; convolution padding (commonly 1 with kernel_size=3 to preserve length).
        """
        super().__init__()

        self.perturbation_kernel_std = perturbation_kernel_std

        # Gaussian random feature embedding of time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Encoding path (downsampling)
        self.conv1 = CausalConv1d(1, channels[0], kernel_size, stride=1, bias=False)
        self.dense1 = Dense1D(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, channels[0])

        self.conv2 = CausalConv1d(channels[0], channels[1], kernel_size, stride=2, bias=False)
        self.dense2 = Dense1D(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, channels[1])

        self.conv3 = CausalConv1d(channels[1], channels[2], kernel_size, stride=2, bias=False)
        self.dense3 = Dense1D(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, channels[2])

        self.conv4 = CausalConv1d(channels[2], channels[3], kernel_size, stride=2, bias=False)
        self.dense4 = Dense1D(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, channels[3])
        
        # Decoding path (upsampling)
        self.tconv4 = nn.ConvTranspose1d(channels[3], channels[2], kernel_size, stride=2, padding=kernel_size//2, output_padding=1, bias=False)
        self.dense5 = Dense1D(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, channels[2])

        self.tconv3 = nn.ConvTranspose1d(channels[2] * 2, channels[1], kernel_size, stride=2, padding=kernel_size//2, output_padding=1, bias=False)
        self.dense6 = Dense1D(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, channels[1])

        self.tconv2 = nn.ConvTranspose1d(channels[1] * 2, channels[0], kernel_size, stride=2, padding=kernel_size//2, output_padding=1, bias=False)
        self.dense7 = Dense1D(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, channels[0])

        self.tconv1 = nn.ConvTranspose1d(channels[0] * 2, 1, kernel_size, padding=kernel_size//2, stride=1)

        # Swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        """
        Forward pass of the time-conditional 1D score network.

        :param x: [torch.Tensor]; shape (B, 1, T), noisy input at time t.
        :param t: [torch.Tensor]; shape (B,), time steps used to condition the network.
        :return: [torch.Tensor]; shape (B, 1, T), score estimate ∇_X log p_t(X) at the input resolution.
        """
        embed = self.act(self.embed(t))

        # Encoder
        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

        # Decoder
        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)))
        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h / self.perturbation_kernel_std(t)[:, None, None]
        return h


class ScoreAttentionNet1DExplicite(nn.Module):
    def __init__(
        self,
        perturbation_kernel_std,
        channels = (32, 64, 128, 256),
        embed_dim = 256,
        nb_groups = 8,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.perturbation_kernel_std = perturbation_kernel_std
        c0, c1, c2, c3 = channels

        # ------- Entrée -------
        self.init_conv = nn.Conv1d(1, c0, kernel_size=7, padding=3)

        # ------- Embedding temporel -------
        time_dim = c0 * 4
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        Block = ResnetBlock1D

        # ================= ENCODER =================
        # Niveau 1 (c0 -> c1)
        self.e1_b1   = Block(c0, c0, time_emb_dim=time_dim, groups=nb_groups)
        self.e1_b2   = Block(c0, c0, time_emb_dim=time_dim, groups=nb_groups)
        self.e1_attn = Residual(PreNorm(c0, LinearAttention1D(c0, heads, dim_head)))
        self.down1   = nn.Conv1d(c0, c1, kernel_size=4, stride=2, padding=1)   # L -> L/2

        # Niveau 2 (c1 -> c2)
        self.e2_b1   = Block(c1, c1, time_emb_dim=time_dim, groups=nb_groups)
        self.e2_b2   = Block(c1, c1, time_emb_dim=time_dim, groups=nb_groups)
        self.e2_attn = Residual(PreNorm(c1, LinearAttention1D(c1, heads, dim_head)))
        self.down2   = nn.Conv1d(c1, c2, kernel_size=4, stride=2, padding=1)   # L/2 -> L/4

        # Niveau 3 (c2 -> c3)
        self.e3_b1   = Block(c2, c2, time_emb_dim=time_dim, groups=nb_groups)
        self.e3_b2   = Block(c2, c2, time_emb_dim=time_dim, groups=nb_groups)
        self.e3_attn = Residual(PreNorm(c2, LinearAttention1D(c2, heads, dim_head)))
        self.down3   = nn.Conv1d(c2, c3, kernel_size=3, padding=1)             # pas de stride (dernier "down")

        # ================= BOTTLENECK =================
        self.mid_b1  = Block(c3, c3, time_emb_dim=time_dim, groups=nb_groups)
        self.mid_attn= Residual(PreNorm(c3, Attention1D(c3, heads, dim_head)))
        self.mid_b2  = Block(c3, c3, time_emb_dim=time_dim, groups=nb_groups)

        # ================= DECODER =================
        # Up depuis c3 -> c2
        self.up3     = nn.ConvTranspose1d(c3, c2, kernel_size=3, padding=1)    # s=1

        self.d3_b1   = Block(c2 + c2, c2, time_emb_dim=time_dim, groups=nb_groups)
        self.d3_b2   = Block(c2 + c2, c2, time_emb_dim=time_dim, groups=nb_groups)
        self.d3_attn = Residual(PreNorm(c2, LinearAttention1D(c2, heads, dim_head)))

        # Up depuis c2 -> c1  (double la longueur : L/4 -> L/2)
        self.up2     = nn.ConvTranspose1d(c2, c1, kernel_size=4, stride=2, padding=1)

        self.d2_b1   = Block(c1 + c1, c1, time_emb_dim=time_dim, groups=nb_groups)
        self.d2_b2   = Block(c1 + c1, c1, time_emb_dim=time_dim, groups=nb_groups)
        self.d2_attn = Residual(PreNorm(c1, LinearAttention1D(c1, heads, dim_head)))

        # Up final c1 -> c0  (double la longueur : L/2 -> L)
        self.up1     = nn.ConvTranspose1d(c1, c0, kernel_size=4, stride=2, padding=1)

        self.d1_b1   = Block(c0 + c0, c0, time_emb_dim=time_dim, groups=nb_groups)
        self.d1_b2   = Block(c0 + c0, c0, time_emb_dim=time_dim, groups=nb_groups)
        self.d1_attn = Residual(PreNorm(c0, LinearAttention1D(c0, heads, dim_head)))

        # ------- Tête -------
        self.final_block = Block(c0 * 2, c0, time_emb_dim=time_dim, groups=nb_groups)  # concat avec résiduel global
        self.final_conv  = nn.Conv1d(c0, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Entrée
        x = self.init_conv(x)
        residual = x.clone()

        # Embedding temporel
        t_emb = self.time_mlp(t)

        # ===== ENCODE =====
        # Niveau 1
        h1_a = self.e1_b1(x, t_emb)            # skip A1
        h1_b = self.e1_b2(h1_a, t_emb)
        h1_b = self.e1_attn(h1_b)              # skip B1
        h2_in = self.down1(h1_b)

        # Niveau 2
        h2_a = self.e2_b1(h2_in, t_emb)        # skip A2
        h2_b = self.e2_b2(h2_a, t_emb)
        h2_b = self.e2_attn(h2_b)              # skip B2
        h3_in = self.down2(h2_b)

        # Niveau 3
        h3_a = self.e3_b1(h3_in, t_emb)        # skip A3
        h3_b = self.e3_b2(h3_a, t_emb)
        h3_b = self.e3_attn(h3_b)              # skip B3
        bottleneck_in = self.down3(h3_b)       # pas de réduction de L, juste C2->C3

        # ===== MID =====
        h = self.mid_b1(bottleneck_in, t_emb)
        h = self.mid_attn(h)
        h = self.mid_b2(h, t_emb)

        # ===== DECODE =====
        # Up vers niveau 3 (utilise skips B3 puis A3)
        h = self.up3(h)
        h = torch.cat([h, h3_b], dim=1); h = self.d3_b1(h, t_emb)
        h = torch.cat([h, h3_a], dim=1); h = self.d3_b2(h, t_emb)
        h = self.d3_attn(h)

        # Up vers niveau 2 (skips B2 puis A2)
        h = self.up2(h)
        h = torch.cat([h, h2_b], dim=1); h = self.d2_b1(h, t_emb)
        h = torch.cat([h, h2_a], dim=1); h = self.d2_b2(h, t_emb)
        h = self.d2_attn(h)

        # Up vers niveau 1 (skips B1 puis A1)
        h = self.up1(h)
        h = torch.cat([h, h1_b], dim=1); h = self.d1_b1(h, t_emb)
        h = torch.cat([h, h1_a], dim=1); h = self.d1_b2(h, t_emb)
        h = self.d1_attn(h)

        # Tête
        h = torch.cat([h, residual], dim=1)
        h = self.final_block(h, t_emb)
        h = self.final_conv(h)

        # Normalisation VP-SDE
        h = h / self.perturbation_kernel_std(t)[:, None, None]
        return h


class ScoreAttentionNet1D(nn.Module):
    def __init__(
        self,
        perturbation_kernel_std,
        channels = (32, 64, 128, 256),
        embed_dim = 256,
        nb_groups = 8,
        heads = 4,
        dim_head = 32,
        kernel_size = None,
        in_channels=1,
        out_channels=1
    ):
        super().__init__()
        self.perturbation_kernel_std = perturbation_kernel_std
        in_out = list(zip(list(channels)[:-1], list(channels)[1:]))

        # Init conv
        self.init_conv = nn.Conv1d(in_channels, channels[0], kernel_size=7, padding=3)

        # Gaussian random feature embedding of time
        time_dim = channels[0] * 4
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Chemins
        Block = ResnetBlock1D
        self.downs = nn.ModuleList([])
        self.ups   = nn.ModuleList([])

        # Descente (2 blocs + attn linéaire optionnelle + downsample)
        for i, (cin, cout) in enumerate(in_out):
            is_last = (i == len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                Block(cin,  cin,  time_emb_dim=time_dim, groups=nb_groups),
                Block(cin,  cin,  time_emb_dim=time_dim, groups=nb_groups),
                Residual(PreNorm(cin, LinearAttention1D(cin, heads, dim_head))),
                nn.Conv1d(cin, cout, kernel_size=3, padding=1) if is_last
                else nn.Conv1d(cin, cout, kernel_size=4, stride=2, padding=1),
            ]))

        # Bottleneck
        self.mid_block1 = Block(channels[-1], channels[-1], time_emb_dim=time_dim, groups=nb_groups)
        self.mid_attn   = Residual(PreNorm(channels[-1], Attention1D(channels[-1], heads, dim_head)))
        self.mid_block2 = Block(channels[-1], channels[-1], time_emb_dim=time_dim, groups=nb_groups)

        # Montée (symétrique)
        for i, (cin, cout) in enumerate(reversed(in_out)):
            is_last = (i == len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                Block(cout + cin, cout, time_emb_dim=time_dim, groups=nb_groups),
                Block(cout + cin, cout, time_emb_dim=time_dim, groups=nb_groups),
                Residual(PreNorm(cout, LinearAttention1D(cout, heads, dim_head))),
                nn.ConvTranspose1d(cout, cin, kernel_size=3, padding=1) if is_last
                else nn.ConvTranspose1d(cout, cin, kernel_size=4, stride=2, padding=1),
            ]))

        # Tête finale : retour vers 1 canal
        self.final_block = Block(channels[0] * 2, channels[0], time_emb_dim=time_dim, groups=nb_groups)
        self.final_conv  = nn.Conv1d(channels[0], out_channels, kernel_size=1)  # sortie score (1 canal)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, L)
        t : (B,)
        → score ∇ₓ log pₜ(x) de shape (B, 1, L)
        """
        # Entrée
        x = self.init_conv(x)
        residual = x.clone()

        # Embedding temporel
        t_emb = self.time_mlp(t)

        # Descente
        skips: list[torch.Tensor] = []
        for block1, block2, attn, down in self.downs:
            x = block1(x, t_emb); skips.append(x)
            x = block2(x, t_emb); x = attn(x); skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # Montée
        for block1, block2, attn, up in self.ups:
            x = torch.cat([x, skips.pop()], dim=1); x = block1(x, t_emb)
            x = torch.cat([x, skips.pop()], dim=1); x = block2(x, t_emb)
            x = attn(x); x = up(x)

        # Tête
        x = torch.cat([x, residual], dim=1)
        x = self.final_block(x, t_emb)
        s = self.final_conv(x)

        s = s / self.perturbation_kernel_std(t)[:, None, None]
        return s


class ScoreNet1DDiffusersV2(nn.Module):
    def __init__(
        self,
        perturbation_kernel_std,
        channels=(32, 64, 128, 256),
        embed_dim=256,
        kernel_size=15,
        in_channels=1,
        out_channels=1,
        T: float = 1.0,
    ):
        super().__init__()
        self.perturbation_kernel_std = perturbation_kernel_std
        self.T = float(T)

        self.unet = UNet1DModel(
            sample_size=None,
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=tuple(channels),
            down_block_types=tuple(["DownBlock1D"] * len(channels)),
            up_block_types=tuple(["UpBlock1D"]   * len(channels)),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t continu normalisé en [0,1] pour l’embedding fourier
        t_norm = (t / self.T).to(dtype=x.dtype, device=x.device)  # (B,)

        out = self.unet(
            sample=x,
            timestep=t_norm,   # tensor float, pas int -> continu
        ).sample  # (B, out_channels, L)

        # normalisation ScoreSDE comme dans tes autres archis
        out = out / self.perturbation_kernel_std(t)[:, None, None]
        return out


class ScoreNet1DDiffusers(nn.Module):
    def __init__(
        self,
        perturbation_kernel_std,
        channels=(32, 64, 128, 256),
        embed_dim=256,      # gardé pour compat signature
        kernel_size=15,     # gardé pour compat signature
        in_channels=1,
        out_channels=1,
        T: float = 1.0,
    ):
        super().__init__()
        self.perturbation_kernel_std = perturbation_kernel_std
        self.T = float(T)

        channels = tuple(channels)
        n_blocks = len(channels)

        down_block_types = ["DownBlock1D"] * (n_blocks - 1) + ["AttnDownBlock1D"]
        up_block_types   = ["AttnUpBlock1D"] + ["UpBlock1D"] * (n_blocks - 1)

        self.unet = UNet1DModel(
            sample_size=None,
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=channels,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # temps continu normalisé pour l’embedding fourier interne
        t_norm = (t / self.T).to(dtype=x.dtype, device=x.device)  # (B,)

        out = self.unet(
            sample=x,
            timestep=t_norm,   # float tensor => time embedding continu
        ).sample  # (B, out_channels, L)

        # normalisation style Score-SDE
        out = out / self.perturbation_kernel_std(t)[:, None, None]
        return out


def _group_norm(num_channels: int, max_groups: int = 32):
    """
    GroupNorm robuste: choisit un nb de groupes qui divise num_channels.
    """
    g = min(max_groups, num_channels)
    while num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class EDMUNet1DBackbone(nn.Module):
    """
    Backbone Fθ pour EDM:
    - Entrée: x_in (B, C, L) déjà préconditionnée par c_in(σ)
    - Conditionnement: c_noise (B,) typiquement 0.25*log(σ)
    - Sortie: résidu (B, C_out, L)
    """
    def __init__(
        self,
        channels=(32, 64, 128, 256),
        embed_dim=256,
        kernel_size=15,
        in_channels=1,
        out_channels=1,
        max_groups=32,
        perturbation_kernel_std=None
    ):
        super().__init__()

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Encoder
        self.conv1  = nn.Conv1d(in_channels, channels[0], kernel_size, padding=kernel_size//2, bias=False)
        self.dense1 = Dense1D(embed_dim, channels[0])
        self.gn1    = _group_norm(channels[0], max_groups)

        self.conv2  = nn.Conv1d(channels[0], channels[1], kernel_size, stride=2, padding=kernel_size//2, bias=False)
        self.dense2 = Dense1D(embed_dim, channels[1])
        self.gn2    = _group_norm(channels[1], max_groups)

        self.conv3  = nn.Conv1d(channels[1], channels[2], kernel_size, stride=2, padding=kernel_size//2, bias=False)
        self.dense3 = Dense1D(embed_dim, channels[2])
        self.gn3    = _group_norm(channels[2], max_groups)

        self.conv4  = nn.Conv1d(channels[2], channels[3], kernel_size, stride=2, padding=kernel_size//2, bias=False)
        self.dense4 = Dense1D(embed_dim, channels[3])
        self.gn4    = _group_norm(channels[3], max_groups)

        # Decoder
        self.tconv4 = nn.ConvTranspose1d(channels[3], channels[2], kernel_size, stride=2,
                                         padding=kernel_size//2, output_padding=1, bias=False)
        self.dense5 = Dense1D(embed_dim, channels[2])
        self.tgn4   = _group_norm(channels[2], max_groups)

        self.tconv3 = nn.ConvTranspose1d(channels[2] * 2, channels[1], kernel_size, stride=2,
                                         padding=kernel_size//2, output_padding=1, bias=False)
        self.dense6 = Dense1D(embed_dim, channels[1])
        self.tgn3   = _group_norm(channels[1], max_groups)

        self.tconv2 = nn.ConvTranspose1d(channels[1] * 2, channels[0], kernel_size, stride=2,
                                         padding=kernel_size//2, output_padding=1, bias=False)
        self.dense7 = Dense1D(embed_dim, channels[0])
        self.tgn2   = _group_norm(channels[0], max_groups)

        self.tconv1 = nn.ConvTranspose1d(channels[0] * 2, out_channels, kernel_size,
                                         stride=1, padding=kernel_size//2)

        self.act = lambda x: x * torch.sigmoid(x)  # SiLU-like (swish)

    def forward(self, x, c_noise):
        """
        x: (B, C, L)
        c_noise: (B,)  (ex: 0.25*log(sigma))
        """
        emb = self.act(self.embed(c_noise))

        # Encoder
        h1 = self.act(self.gn1(self.conv1(x)  + self.dense1(emb)))
        h2 = self.act(self.gn2(self.conv2(h1) + self.dense2(emb)))
        h3 = self.act(self.gn3(self.conv3(h2) + self.dense3(emb)))
        h4 = self.act(self.gn4(self.conv4(h3) + self.dense4(emb)))

        # Decoder
        h  = self.act(self.tgn4(self.tconv4(h4) + self.dense5(emb)))
        h  = self.act(self.tgn3(self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(emb)))
        h  = self.act(self.tgn2(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(emb)))
        out = self.tconv1(torch.cat([h, h1], dim=1))

        return out


class EDMPrecond1D(nn.Module):
    def __init__(self, F, sigma_data=0.5):
        super().__init__()
        self.F = F
        self.sigma_data = sigma_data

    def forward(self, x, sigma):
        # x: (B,1,L), sigma: (B,)
        sd = self.sigma_data
        sigma2 = sigma**2
        denom = torch.sqrt(sigma2 + sd**2)

        c_skip  = (sd**2) / (sigma2 + sd**2)
        c_out   = (sigma * sd) / denom
        c_in    = 1.0 / denom
        c_noise = 0.25 * torch.log(sigma)  #  [oai_citation:11‡openreview.net](https://openreview.net/pdf?id=k7FuTOWMOc7)

        x_in = x * c_in[:, None, None]
        F_out = self.F(x_in, c_noise)      # ici ton backbone prend "t-like" (B,)
        D = c_skip[:, None, None]*x + c_out[:, None, None]*F_out
        return D
