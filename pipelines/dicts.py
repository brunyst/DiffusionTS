from utils.imports_statiques import *
from models.schedules import *
from networks.unets import *
from metrics.MLE_params import *


DATA_CONFIG: dict[str, dict] = {
    "OUrange": dict(
        simulate="simulate_OU_range",
        get_params="get_params_OU_range",
        params=["L", "dt", "x0", "theta_range", "mu_range", "sigma_range"],
    ),
    "OU": dict(
        simulate="simulate_OU",
        get_params="get_params_OU",
        params=["L", "dt", "x0", "theta", "mu", "sigma"],
    ),
    "OUmodes": dict(
        simulate="simulate_OUmodes",
        get_params="get_params_OUmodes",
        params=["L", "dt", "x0", "theta1", "mu1", "sigma1", "theta2", "mu2", "sigma2"],
    ),
    "BM": dict(
        simulate="simulate_BM",
        get_params="get_params_BM",
        params=["L", "dt", "x0", "sigma"],
    ),
    "GBM": dict(
        simulate="simulate_GBM",
        get_params="get_params_GBM",
        params=["L", "dt", "x0", "mu", "sigma"],
    ),
    "Lines": dict(
        simulate="simulate_lines",
        get_params="get_params_lines",
        params=["L", ("a_range", (-1, 1)), ("b_range", (0, 1))],
    ),
    "Sines": dict(
        simulate="simulate_sines_1D",
        get_params="get_params_sines_1D",
        params=[
            "L",
            ("amp_range", (0.5, 1.5)),
            ("freq_range", (1, 5)),
            ("phase_range", (0, 2 * np.pi)),
        ],
    ),
    "LinearODEs": dict(
        simulate="simulate_linear_ODE",
        get_params="get_params_linear_ODE",
        params=[
            "L",
            ("a0_range", (-1, 1)),
            ("a1_range", (-1, 1)),
            ("b_range", (-1, 1)),
            ("x0_range", (-1, 1)),
            ("dt", 1 / 252),
        ],
    ),
    "CIR": dict(
        simulate="simulate_CIR",
        get_params="get_params_CIR",
        params=["L", "dt", "x0", "theta", "mu", "sigma"],
    ),
    "CIRrange": dict(
        simulate="simulate_CIR_range",
        get_params="get_params_CIR_range",
        params=["L", "dt", "x0", "theta_range", "mu_range", "sigma_range"],
    ),
    "Heston": dict(
        simulate="simulate_Heston",
        get_params="get_params_Heston",
        params=["L", "dt", "kappa", "theta", "xi", "rho", "r", ("S0", 1.0), ("v0", 1.0), ("observe_v", True)],
    ),
    "HestonRange": dict(
        simulate="simulate_Heston_range",
        get_params="get_params_Heston_range",
        params=[
            "L",
            "dt",
            "kappa_range",
            "theta_range",
            "xi_range",
            "rho_range",
            "r_range",
            ("S0", 1.0),
            ("v0", 1.0),
            ("observe_v", True),
        ],
    ),
    "PDV2factor": dict(
        simulate="simulate_PDV2factor",
        get_params="get_params_PDV2factor",
        params=[
            "L",
            "dt",
            ("S0", 1.0),
            "beta0",
            "beta1",
            "beta2",
            "lambda1",
            "lambda2",
            "a1",
            "a2",
            ("R1_0", 0.0),
            ("R2_0", 0.0),
            ("sigma_floor", 1e-6),
        ],
    ),
}



SCHEDULE_CONFIG: dict[str, dict] = {
    # VP / SubVP / CosineVP / VPOU are "dsm" losses in your pipeline
    "VP": dict(
        schedule=VPSchedule,
        params=["betamin", "betamax"],
        loss="dsm",
    ),
    "SubVP": dict(
        schedule=SubVPSchedule,
        params=["betamin", "betamax", "T", "eps"],
        loss="dsm",
    ),
    "CosineVP": dict(
        schedule=CosineVPSchedule,
        params=["s", "T", "eps"],
        loss="dsm",
    ),
    "VPOU": dict(
        schedule=VPOUSchedule,
        params=["T", "eps"],
        loss="dsm",
    ),
    "VE": dict(
        schedule=VESchedule,
        params=["sigmamin", "sigmamax", "T", "eps"],
        loss="dsm",
    ),
    "EDM": dict(
        schedule=EDMSchedule,
        params=["sigmamin", "sigmamax", "rho", "edmPmean", "edmPstd", "edmsigmadata", "T", "eps"],
        loss="edm",
    ),
    "GaussianFlow": dict(
        schedule=GaussianFlowSchedule,
        params=["T", "eps"],
        loss="dsm",
    ),
    "LogSNR": dict(
        schedule=LogSNRSchedule,
        params=["lambdamin", "lambdamax", "T", "eps"],
        loss="dsm",
    ),
}


MODEL_CONFIG: dict[str, dict] = {
    "ScoreNet1D Baseline": dict(
        model=ScoreNet1D_v1,
        params=[
            "embed_dim", "kernel_size", "channels", "in_channels", "out_channels",
        ],
    ),
    "ScoreNet1D Adapt": dict(
        model=ScoreNet1D_v1_adapt,
        params=[
            "embed_dim", "kernel_size", "channels", "in_channels", "out_channels",
        ],
    ),
    "ScoreNet1D Baseline 2channels": dict(
        model=ScoreNet1D_v1_2channels,
        params=[
            "embed_dim", "kernel_size", "channels", "in_channels", "out_channels",
        ],
    ),
    "ScoreNet1D Baseline Quadratic Encoder": dict(
        model=ScoreNet1D_v1_quadra,
        params=[
            "embed_dim", "kernel_size", "channels", "in_channels", "out_channels",
        ],
    ),
    "ScoreNet1D_v2 ResBlocks": dict(
        model=ScoreNet1D_v2,
        params=[
            "embed_dim", "kernel_size", "channels", "in_channels", "out_channels",
        ],
    ),
    "ScoreNet1D ResBlocks": dict(
        model=ScoreNet1D_v3,
        params=[
            "embed_dim", "kernel_size", "channels", "in_channels", "out_channels",
        ],
    ),
    "ScoreNet1D ResBlocks v2": dict(
        model=ScoreNet1D_v3_v2,
        params=[
            "embed_dim", "kernel_size", "nb_groups", "nb_heads", "dim_head",
            "in_channels", "out_channels", "channels", "num_layers", "dim_ff", "dropout",
        ],
    ),
    "ScoreNet1D Attention Bottleneck": dict(
        model=ScoreNet1D_v4,
        params=[
            "embed_dim", "kernel_size", "nb_groups", "nb_heads", "dim_head",
            "in_channels", "out_channels", "channels", "num_layers", "dim_ff", "dropout",
        ],
    ),
    "ScoreNet1D Forme Quadratique": dict(
        model=ScoreNet1D_v5,
        params=[
            "embed_dim", "kernel_size", "nb_groups", "nb_heads", "dim_head",
            "in_channels", "out_channels", "channels", "num_layers", "dim_ff", "dropout",
        ],
    ),
    "ScoreNet1D Dilations": dict(
        model=ScoreNet1D_v5,
        params=[
            "embed_dim", "kernel_size", "nb_groups", "nb_heads", "dim_head",
            "in_channels", "out_channels", "channels", "num_layers", "dim_ff", "dropout",
        ],
    ),
    "ScoreNet1D Attention": dict(
        model=ScoreAttentionNet1D,
        params=[
            "embed_dim", "kernel_size", "nb_groups", "nb_heads", "dim_head",
            "in_channels", "out_channels", "channels", "num_layers", "dim_ff", "dropout",
        ],
    ),
    "ScoreNet1D Diffusers": dict(
        model=ScoreNet1DDiffusers,
        params=[
            "embed_dim", "kernel_size", "nb_groups", "nb_heads", "dim_head",
            "in_channels", "out_channels", "channels", "num_layers", "dim_ff", "dropout",
        ],
    ),
    "EDMUNet1D": dict(
        model=EDMUNet1DBackbone,
        params=[
            "embed_dim", "kernel_size", "nb_groups", "nb_heads", "dim_head",
            "in_channels", "out_channels", "channels", "num_layers", "dim_ff", "dropout",
        ],
    ),
    "TransFusion1D": dict(
        model=ScoreAttentionNet1D,
        params=[
            "embed_dim", "kernel_size", "nb_groups", "nb_heads", "dim_head",
            "in_channels", "out_channels", "channels", "num_layers", "dim_ff", "dropout",
        ],
    ),
    "TransFusionRope1D": dict(
        model=ScoreAttentionNet1D,
        params=[
            "embed_dim", "kernel_size", "nb_groups", "nb_heads", "dim_head",
            "in_channels", "out_channels", "channels", "num_layers", "dim_ff", "dropout",
        ],
    ),
}


SAMPLING_CONFIG = {
    "OU": dict(
        series_labels=("X",),
        plot_titles=("Simulated vs. Generated (X) — OU",),
    ),
    "OUrange": dict(
        series_labels=("X",),
        plot_titles=("Simulated vs. Generated (X) — OUrange",),
    ),
    "OUmodes": dict(
        series_labels=("X",),
        plot_titles=("Simulated vs. Generated (X) — OUmodes",),
    ),
    "Heston": dict(
        series_labels=("S", "v"),
        plot_titles=("Simulated vs. Generated (S)", "Simulated vs. Generated (v)"),
    ),
    "HestonRange": dict(
        series_labels=("S", "v"),
        plot_titles=("Simulated vs. Generated (S)", "Simulated vs. Generated (v)"),
    ),
    "Heston_dS2": dict(
        series_labels=("S", "v"),
        plot_titles=("Simulated vs. Generated (S)", "Simulated vs. Generated (v)"),
    ),
    "HestonRange_dS2": dict(
        series_labels=("S", "v"),
        plot_titles=("Simulated vs. Generated (S)", "Simulated vs. Generated (v)"),
    ),
    "PDV2factor": dict(
        series_labels=("S", "sigma"),
        plot_titles=("Simulated vs. Generated (S)", "Simulated vs. Generated (sigma)"),
    ),
    "Heston1D": dict(
        series_labels=("S",),
        plot_titles=("Simulated vs. Generated (S) — Heston1D",),
    ),
    "HestonRange1D": dict(
        series_labels=("S",),
        plot_titles=("Simulated vs. Generated (S) — HestonRange1D",),
    ),
    "Lines": dict(
        series_labels=("X",),
        plot_titles=("Simulated vs. Generated",),
    ),
    "Sines": dict(
        series_labels=("X",),
        plot_titles=("Simulated vs. Generated",),
    ),
    "LinearODEs": dict(
        series_labels=("X",),
        plot_titles=("Simulated vs. Generated",),
    ),
}

METRICS_CONFIG = {
    "OU": dict(
        mle_fn=plot_params_distrib_OU,
        true_params=("theta", "mu", "sigma"),
        suptitle=r"MLE of OU params ($\theta$, $\mu$, $\sigma$)",
    ),
    "OUrange": dict(
        mle_fn=plot_params_distrib_OU,
        true_params=("theta", "mu", "sigma"),
        suptitle=r"MLE of OU params ($\theta$, $\mu$, $\sigma$)",
    ),
    "OUmodes": dict(
        mle_fn=plot_params_distrib_OU,
        true_params=("theta", "mu", "sigma"),
        suptitle=r"MLE of OU params ($\theta$, $\mu$, $\sigma$)",
    ),
    "CIR": dict(
        mle_fn=plot_params_distrib_CIR,
        true_params=("theta", "mu", "sigma"),
        suptitle=r"MLE of CIR params ($\theta$, $\mu$, $\sigma$)",
    ),
    "CIRrange": dict(
        mle_fn=plot_params_distrib_CIR,
        true_params=("theta", "mu", "sigma"),
        suptitle=r"MLE of CIR params ($\theta$, $\mu$, $\sigma$)",
    ),
    "Heston": dict(
        mle_fn=plot_params_distrib_Heston,
        true_params=("kappa", "theta", "xi", "rho", "r"),
        suptitle=r"MLE of Heston params ($\kappa$, $\theta$, $\xi$, $\rho$, $r$)",
    ),
    "HestonRange": dict(
        mle_fn=plot_params_distrib_Heston,
        true_params=("kappa", "theta", "xi", "rho", "r"),
        suptitle=r"MLE of Heston params ($\kappa$, $\theta$, $\xi$, $\rho$, $r$)",
    ),
    "BM": dict(
        mle_fn=plot_params_distrib_BM,
        true_params=("sigma",),
        suptitle=r"MLE of BM params ($\sigma$)",
    ),
    "GBM": dict(
        mle_fn=plot_params_distrib_GBM,
        true_params=("mu", "sigma"),
        suptitle=r"MLE of GBM params ($\mu$, $\sigma$)",
    ),
    "Lines": dict(
        mle_fn=plot_linreg_params_distrib,
        true_params=("a", "b"),
        suptitle=r"Linear regression on lines: slope $a$, intercept $b$",
    ),
    "Sines": dict(
        mle_fn=plot_sine_params_distrib_1D,
        true_params=("A", "f", "phi"),
        suptitle=r"Sinusoidal fit: amplitude $A$, frequency $f$, phase $\phi$",
    ),
    "LinearODEs": dict(
        mle_fn=plot_ode_params_distrib_1D,
        true_params=("x0", "a0", "a1", "b"),
        suptitle=r"Linear ODE fit: $(x_0, a_0, a_1, b)$",
    ),
}