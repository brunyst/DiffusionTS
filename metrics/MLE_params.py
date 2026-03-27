from utils.imports_statiques import *
from utils.imports_dynamiques import *


def bounded_kde_1d(data, grid, lower, upper, bw="scott"):
    """
    KDE 1D avec support borné [lower, upper].
    data : 1D array
    grid : points où évaluer la densité (dans [lower, upper])
    """
    kde = KDEUnivariate(data)
    kde.fit(bw=bw, cut=0)  # cut=0 : n'étend pas trop loin aux bords

    dens = kde.evaluate(grid)

    # On force à zéro hors des bornes (au cas où)
    mask = (grid >= lower) & (grid <= upper)
    dens[~mask] = 0.0

    # Renormalisation pour que ∫ dens dx ≈ 1
    area = np.trapz(dens, grid)
    if area > 0:
        dens /= area

    return dens


def bounded_kde_logit(data, grid, a, b, bw_method="scott", eps=1e-6):
    """
    KDE 1D borné sur [a, b] via la transformation logit.

    data : array 1D des échantillons dans [a,b]
    grid : points x où évaluer la densité (dans [a,b])
    a,b  : bornes du support
    """
    data = np.asarray(data)
    grid = np.asarray(grid)

    # 1) Transforme x -> u = logit( (x-a)/(b-a) )  dans R
    z = (data - a) / (b - a)
    z = np.clip(z, eps, 1.0 - eps)        # évite 0 ou 1
    u = logit(z)

    kde = gaussian_kde(u, bw_method=bw_method)

    # 2) Projection de la grille x dans R
    z_g = (grid - a) / (b - a)
    z_g = np.clip(z_g, eps, 1.0 - eps)
    u_g = logit(z_g)

    # 3) Densité en u, puis changement de variable
    p_u = kde(u_g)                        # p_U(u)
    jac = 1.0 / ((b - a) * z_g * (1.0 - z_g))   # du/dx

    p_x = p_u * jac                       # p_X(x) = p_U(u) * du/dx

    # 4) Force à 0 hors [a,b] et renormalise
    p_x[(grid < a) | (grid > b)] = 0.0
    area = np.trapz(p_x, grid)
    if area > 0:
        p_x /= area

    return p_x


def MLE_OU_robust(params, X_data, dt):
    """
    Compute the MLE on Ornstein-Uhlenbeck (OU) data.

    :param params: [list or tuple]; parameters [θ, μ, σ] of the OU process.
    :param X_data: [np.ndarray]; observed time series of shape (N,).
    :param dt: [float]; time step between observations.

    Return: [float]; negative log-likelihood value (to be minimized).
    """
    theta, mu, sigma = params
    N = len(X_data)
    logL = 0

    exp_neg_theta_dt = np.exp(-theta * dt)
    one_minus_exp_neg_theta_dt = 1 - exp_neg_theta_dt
    sigma_eta2 = (sigma ** 2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))

    for t in range(N - 1):
        mu_t = X_data[t] * exp_neg_theta_dt + mu * one_minus_exp_neg_theta_dt
        residual = X_data[t + 1] - mu_t
        logL += -0.5 * np.log(2 * np.pi * sigma_eta2) - (residual ** 2) / (2 * sigma_eta2)

    return -logL


def MLE_OU_robust_vect(params, X_data, dt):
    """
    Compute the MLE on Ornstein-Uhlenbeck (OU) data (vectorized version).

    :param params: [list or tuple]; parameters [θ, μ, σ] of the OU process.
    :param X_data: [np.ndarray]; observed time series of shape (N,).
    :param dt: [float]; time step between observations.

    Return: [float]; negative log-likelihood value (to be minimized).
    """
    theta, mu, sigma = params

    # Sécurité de base (optionnelle si tu as déjà des bounds dans minimize)
    if theta <= 0 or sigma <= 0:
        return 1e20

    X_data = np.asarray(X_data, dtype=float)
    N = X_data.size
    if N < 2:
        return 1e20  # pas assez de points

    exp_neg_theta_dt = np.exp(-theta * dt)
    one_minus_exp_neg_theta_dt = 1.0 - exp_neg_theta_dt

    # Variance de l'innovation (scalaire)
    sigma_eta2 = (sigma ** 2 / (2.0 * theta)) * (1.0 - np.exp(-2.0 * theta * dt))
    if sigma_eta2 <= 0 or not np.isfinite(sigma_eta2):
        return 1e20

    # Vecteurs X_k et X_{k+1}
    x_t   = X_data[:-1]
    x_tp1 = X_data[1:]

    # Moyennes conditionnelles m(x_t) vectorisées
    mu_t = x_t * exp_neg_theta_dt + mu * one_minus_exp_neg_theta_dt

    # Résidus vectorisés
    residuals = x_tp1 - mu_t

    # Log-vraisemblance (sans boucle)
    const_term = -0.5 * (N - 1) * np.log(2.0 * np.pi * sigma_eta2)
    quad_term  = -0.5 * np.sum(residuals ** 2) / sigma_eta2
    logL = const_term + quad_term

    return -logL  # on minimise la négative log-vraisemblance


def _clean_1d(x):
    x = np.asarray(x).ravel().astype(float)
    x = x[np.isfinite(x)]
    return x

# ---------------------------------------------------------------------------
# Shared visual constants for all MLE distribution plots
# ---------------------------------------------------------------------------

# Paul Tol "bright" palette — colorblind-safe, publication-quality
_COLOR_TRAIN = "#4477AA"   # blue  — reference / train data
_COLOR_GEN   = "#228833"   # green — generated data

# W1 annotation style: barely-visible white backing keeps text readable over
# the coloured KDE fills without drawing attention to the box itself
_BOX_STYLE = dict(boxstyle="round,pad=0.25", facecolor="white",
                  alpha=0.40, edgecolor="none")

# Params annotation style: completely transparent — text floats seamlessly on
# the plot, matching the clean look of the reference publication figures
_PARAMS_STYLE = dict(boxstyle="round,pad=0.15", facecolor="none",
                     edgecolor="none")

# Mapping from Python param names → LaTeX symbols (used in titles / xlabels)
_LATEX_PARAMS = {
    "theta": r"\theta", "mu":    r"\mu",   "sigma": r"\sigma",
    "kappa": r"\kappa", "xi":    r"\xi",   "rho":   r"\rho",
    "phi":   r"\phi",   "r":     "r",
    "A":     "A",       "f":     "f",
    "a":     "a",       "b":     "b",
    "x0":    "x_0",     "a0":    "a_0",   "a1":    "a_1",
}

# ---------------------------------------------------------------------------
# Publication-quality rcParams applied to every MLE figure
# Inspired by matplotlib + Computer Modern — no LaTeX required
# ---------------------------------------------------------------------------
_MLE_RCPARAMS = {
    # --- Font: STIXGeneral — exact font used in the reference paper figures ---
    # (matches Computer Modern / LaTeX appearance without requiring a TeX install)
    "font.family":                  "STIXGeneral",   # exact font from the reference paper
    "font.weight":                  "normal",
    "mathtext.fontset":             "cm",            # Computer Modern for all math
    "font.size":                    22,              # base size — matches reference paper
    "axes.labelsize":               22,
    "xtick.labelsize":              18,
    "ytick.labelsize":              18,
    "legend.fontsize":              18,
    "axes.titlesize":               22,
    "figure.titlesize":             24,

    # --- Axis tick formatters: mathtext for powers (e.g. 10³ → $10^3$) ---
    "axes.formatter.use_mathtext":  True,
    "axes.formatter.limits":        (-6, 6),

    # --- Tick style: outward on all 4 sides + minor ticks ---
    "xtick.direction":              "out",
    "ytick.direction":              "out",
    "xtick.top":                    True,
    "ytick.right":                  True,
    "xtick.minor.visible":          True,
    "ytick.minor.visible":          True,
    "xtick.major.size":             5.0,
    "xtick.minor.size":             2.5,
    "ytick.major.size":             5.0,
    "ytick.minor.size":             2.5,
    "xtick.major.width":            0.8,
    "xtick.minor.width":            0.6,
    "ytick.major.width":            0.8,
    "ytick.minor.width":            0.6,

    # --- Axes / figure ---
    "axes.linewidth":               0.5,
    "axes.facecolor":               "white",
    "figure.facecolor":             "white",
    "axes.grid":                    False,

    # --- Lines ---
    "lines.linewidth":              1.5,

    # --- Legend: no frame ---
    "legend.frameon":               False,

    # --- Publication output quality ---
    "figure.dpi":                   150,
    "savefig.dpi":                  300,
    "pdf.fonttype":                 42,
    "ps.fonttype":                  42,
    "savefig.bbox":                 "tight",
}


def _set_gt_title(ax, params_dict, fontsize=14, pad=8):
    """
    Set subplot title of the form "Ground truth: $X \\in [min, max]$".

    params_dict keys must contain 'min' / 'max' (case-insensitive) to detect
    the range; the base param symbol is inferred from the key prefix via
    _LATEX_PARAMS.  Falls back to listing all key=value pairs if no min/max.
    """
    if not params_dict:
        return
    min_key = next((k for k in params_dict if "min" in k.lower()), None)
    max_key = next((k for k in params_dict if "max" in k.lower()), None)
    if min_key and max_key:
        _lmin, v_min = params_dict[min_key]
        _lmax, v_max = params_dict[max_key]
        # Infer base param name: "theta_min" → "theta"
        base = min_key.lower().replace("_min", "").replace("min", "").strip("_")
        lat  = _LATEX_PARAMS.get(base, base)
        ax.set_title(
            f"Ground truth: ${lat} \\in [{float(v_min):.2f},\\,{float(v_max):.2f}]$",
            fontsize=fontsize, pad=pad,
        )
    else:
        parts = [
            f"${l} = {float(v):.2f}$"
            for _, (l, v) in params_dict.items()
            if isinstance(v, (int, float, np.integer, np.floating))
        ]
        if parts:
            ax.set_title("Ground truth: " + ", ".join(parts), fontsize=fontsize, pad=pad)


def _mle_kdeplot(
    ax, data_col, gen_col, param_name,
    title_fontsize=18, label_fontsize=14, tick_fontsize=14,
    true_val=None, fix=False, m_train=None, m_gen=None,
):
    """
    Unified helper for a single MLE parameter subplot.

    Layout:
      subplot title  → ground truth range (set by _set_gt_title in caller)
      loc='best'     → legend with colored patches + M_train / M_gen labels

    Visual: histogram (step-filled, low alpha) as base layer showing the raw
    bar distribution, with a KDE smooth curve (line-only, no fill) on top.
    Publication-standard combo that makes the empirical distribution and its
    smooth estimate both visible without visual clutter.

    Both distributions are aligned to the same sample count N_mle = len(gen_col)
    so that M_train = M_gen in the labels.  If params_data is larger it is
    randomly subsampled with a fixed seed for reproducibility.
    """
    from matplotlib.ticker import AutoMinorLocator

    # --- Align sample sizes so M_train = M_gen in the display ---
    # Subsample whichever set is larger to match the smaller one.
    # Fixed seed ensures reproducibility across subplots of the same figure.
    N_mle = min(len(data_col), len(gen_col))
    rng   = np.random.default_rng(seed=0)
    if len(data_col) > N_mle:
        data_col = rng.choice(data_col, N_mle, replace=False)
    if len(gen_col) > N_mle:
        gen_col = rng.choice(gen_col, N_mle, replace=False)
    lat = _LATEX_PARAMS.get(param_name, param_name)

    label_train = fr"Train ($M_{{\rm train}}={N_mle}$)"
    label_gen   = fr"Gen ($M_{{\rm gen}}={N_mle}$)"

    # --- Histogram: step-filled, low alpha — shows the raw bar distribution ---
    bins = min(60, max(15, int(4 * np.sqrt(N_mle))))
    ax.hist(data_col, density=True, bins=bins,
            color=_COLOR_TRAIN, alpha=0.25, histtype="stepfilled", linewidth=0,
            label=label_train)
    ax.hist(gen_col, density=True, bins=bins,
            color=_COLOR_GEN,   alpha=0.25, histtype="stepfilled", linewidth=0,
            label=label_gen)

    # --- KDE smooth curve: line-only on top of histogram, no fill ---
    sns.kdeplot(ax=ax, data=data_col, fill=False,
                color=_COLOR_TRAIN, linewidth=2.0)
    sns.kdeplot(ax=ax, data=gen_col,  fill=False,
                color=_COLOR_GEN,   linewidth=2.0)

    if fix and true_val is not None:
        ax.axvline(true_val, color="#333333", linestyle="--", linewidth=1.0)

    # --- Labels: hat notation on x-axis, title set by caller ---
    ax.set_xlabel(f"$\\hat{{{lat}}}$", fontsize=label_fontsize)
    ax.set_ylabel("Empirical Density", fontsize=label_fontsize)

    # --- Ticks: outward on all 4 sides, minor ticks enabled ---
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize,
                   direction="out", top=True, right=True)
    ax.tick_params(axis="both", which="minor",
                   direction="out", top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # --- Legend: auto-placed, colored patches from histogram entries ---
    ax.legend(fontsize=label_fontsize - 2, frameon=False, loc="best")


def wasserstein_1D(a, b, empty_value=np.nan):
    a = _clean_1d(a)
    b = _clean_1d(b)
    if a.size == 0 or b.size == 0:
        return float(empty_value)   # ou 0.0 si tu préfères, mais nan est plus honnête
    return float(wasserstein_distance(a, b))


def plot_params_distrib_OU(
    X_data, X_gen, dt,
    fix=False,
    theta=None, mu=None, sigma=None,
    data_label="Data", gen_label="Gen",
    filter_outliers=True,
    save_img=True,
    plot_dir="plots",
    plot_path="params_distrib.png",
    param_dir="params",
    show=True,
    force=False,
    vectorized=True,
    returns=False,
    returns_type='simple',
    suptitle=None,
    params1=None,
    params2=None,
    params3=None,
    compute_w1=True,
    save_w1=True,
    w1_dir="metrics",
    verbose=False
):

    # === Style cohérent avec plot_random_time_series ===
    title_fontsize = 18
    label_fontsize = 14
    tick_fontsize  = 14
    suptitle_size  = 20

    params1 = params1 or {}
    params2 = params2 or {}
    params3 = params3 or {}

    plt.rcParams.update({'font.size': label_fontsize})

    # ---------- Basic infos ----------
    N_data = len(X_data)
    N_gen  = len(X_gen)

    # ---------- Paths for parameter cache ----------
    base = os.path.basename(plot_path)
    stem, _ = os.path.splitext(base)

    if returns:
        if returns_type == 'log':
            data_label = f"logr{data_label}"
            gen_label  = f"logr{gen_label}"
        elif returns_type == 'simple':
            data_label = f"r{data_label}"
            gen_label  = f"r{gen_label}"

    data_param_path = os.path.join(param_dir, f"{data_label}_{stem}.npy")
    gen_param_path  = os.path.join(param_dir,  f"{gen_label}_{stem}.npy")

    # ---------- Load or compute parameters ----------
    if os.path.exists(data_param_path) and os.path.exists(gen_param_path) and not force:
        if verbose:
            print("[INFO] Loading cached OU parameters from:")
            print(f"       {data_param_path}")
            print(f"       {gen_param_path}")
        params_data = np.load(data_param_path)
        params_gen  = np.load(gen_param_path)

    else:
        if verbose:
            print("[INFO] No cached parameters found. Estimating OU parameters...")

        # ----- Estimate OU on real data -----
        params_data = np.zeros((N_data, 3))
        for m in range(N_data):
            params_init_data = [1, np.mean(X_data[m]), np.std(X_data[m])]
            if vectorized:
                result_data = minimize(
                    MLE_OU_robust_vect,
                    np.array(params_init_data),
                    args=(X_data[m], dt),
                    bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                    method='L-BFGS-B'
                )
            else:
                result_data = minimize(
                    MLE_OU_robust,
                    np.array(params_init_data),
                    args=(X_data[m], dt),
                    bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                    method='L-BFGS-B'
                )
            params_data[m] = result_data.x

        # ----- Estimate OU on generated data -----
        params_gen = np.zeros((N_gen, 3))
        for m in range(N_gen):
            params_init_gen = [1, np.mean(X_gen[m]), np.std(X_gen[m])]
            if vectorized:
                result_gen = minimize(
                    MLE_OU_robust_vect,
                    np.array(params_init_gen),
                    args=(X_gen[m], dt),
                    bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                    method='L-BFGS-B'
                )
            else:
                result_gen = minimize(
                    MLE_OU_robust,
                    np.array(params_init_gen),
                    args=(X_gen[m], dt),
                    bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                    method='L-BFGS-B'
                )
            params_gen[m] = result_gen.x

        # ----- Optional filtering -----
        if filter_outliers:
            lb = np.percentile(params_data, 3, axis=0)
            ub = np.percentile(params_data, 97, axis=0)
            params_data = params_data[((params_data >= lb) & (params_data <= ub)).all(axis=1)]

            lb = np.percentile(params_gen, 3, axis=0)
            ub = np.percentile(params_gen, 97, axis=0)
            params_gen = params_gen[((params_gen >= lb) & (params_gen <= ub)).all(axis=1)]

        # ----- Save -----
        os.makedirs(param_dir, exist_ok=True)
        np.save(data_param_path, params_data)
        np.save(gen_param_path, params_gen)

        if verbose:
            print(f"[INFO] Saved OU parameters to:")
            print(f"       {data_param_path}")
            print(f"       {gen_param_path}")
    
    # ---------- Plot ----------
    plt.rcParams.update(_MLE_RCPARAMS)
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))

    _kw = dict(title_fontsize=title_fontsize, label_fontsize=label_fontsize, tick_fontsize=tick_fontsize, m_train=N_data, m_gen=N_gen)
    _mle_kdeplot(axs[0], params_data[:, 0], params_gen[:, 0], "theta", fix=fix, true_val=theta, **_kw)
    _mle_kdeplot(axs[1], params_data[:, 1], params_gen[:, 1], "mu",    fix=fix, true_val=mu,    **_kw)
    _mle_kdeplot(axs[2], params_data[:, 2], params_gen[:, 2], "sigma", fix=fix, true_val=sigma, **_kw)

    _set_gt_title(axs[0], params1, label_fontsize)
    _set_gt_title(axs[1], params2, label_fontsize)
    _set_gt_title(axs[2], params3, label_fontsize)
    
    # ---------- Suptitle ----------
    # suptitle suppressed — context is given by the output folder and config

    plt.tight_layout()

    # ---------- Save ----------
    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"MLE_{data_label}_{stem}.png"
        full_plot_path = os.path.join(plot_dir, filename)
        fig.savefig(full_plot_path, dpi=300, bbox_inches='tight')
        
        if verbose:
            print(f"[INFO] Saved plot at: {full_plot_path}")

    # ---------- Show ----------
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_params_distrib_OU_multi(
    X_data, X_gen_dict, dt,
    fix=False, theta=None, mu=None, sigma=None,
    data_label="Data",
    filter_outliers=True,
    save_img=True,
    plot_dir="plots",
    plot_path="params_distrib_multi.png",
    param_dir="params",
    show=True,
    force=False,
    vectorized=True,
    suptitle=None,
    params1=None, params2=None, params3=None,
    compute_w1=True,
    save_w1=True,
    w1_dir="metrics",
):
    """
    X_gen_dict: dict {gen_label: X_gen_array}
      ex: {"ScoreNet1DV1": X_gen_v1, "ScoreAttentionNet1D": X_gen_attn}
    """
    title_fontsize = 18
    label_fontsize = 14
    tick_fontsize  = 14
    suptitle_size  = 20

    params1 = params1 or {}
    params2 = params2 or {}
    params3 = params3 or {}

    plt.rcParams.update({'font.size': label_fontsize})

    # --- estimate params on data (cached) ---
    base = os.path.basename(plot_path)
    stem, _ = os.path.splitext(base)

    data_param_path = os.path.join(param_dir, f"{data_label}_{stem}.npy")

    if os.path.exists(data_param_path) and not force:
        params_data = np.load(data_param_path)
    else:
        params_data = np.zeros((len(X_data), 3))
        for m in range(len(X_data)):
            params_init = [1, np.mean(X_data[m]), np.std(X_data[m])]
            fun = MLE_OU_robust_vect if vectorized else MLE_OU_robust
            res = minimize(
                fun, np.array(params_init), args=(X_data[m], dt),
                bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                method="L-BFGS-B",
            )
            params_data[m] = res.x

        if filter_outliers:
            lb = np.percentile(params_data, 3, axis=0)
            ub = np.percentile(params_data, 97, axis=0)
            params_data = params_data[((params_data >= lb) & (params_data <= ub)).all(axis=1)]

        os.makedirs(param_dir, exist_ok=True)
        np.save(data_param_path, params_data)

    # --- estimate params on each generated set (cached) ---
    params_gen_map = {}
    for gen_label, X_gen in X_gen_dict.items():
        gen_param_path = os.path.join(param_dir, f"{gen_label}_{stem}.npy")
        if os.path.exists(gen_param_path) and not force:
            params_gen = np.load(gen_param_path)
        else:
            params_gen = np.zeros((len(X_gen), 3))
            for m in range(len(X_gen)):
                params_init = [1, np.mean(X_gen[m]), np.std(X_gen[m])]
                fun = MLE_OU_robust_vect if vectorized else MLE_OU_robust
                res = minimize(
                    fun, np.array(params_init), args=(X_gen[m], dt),
                    bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                    method="L-BFGS-B",
                )
                params_gen[m] = res.x

            if filter_outliers:
                lb = np.percentile(params_gen, 3, axis=0)
                ub = np.percentile(params_gen, 97, axis=0)
                params_gen = params_gen[((params_gen >= lb) & (params_gen <= ub)).all(axis=1)]

            os.makedirs(param_dir, exist_ok=True)
            np.save(gen_param_path, params_gen)

        params_gen_map[gen_label] = params_gen

    # --- compute W1 per architecture ---
    w1 = None
    if compute_w1:
        w1 = {}
        for gen_label, params_gen in params_gen_map.items():
            w1[gen_label] = {
                "theta": wasserstein_1D(params_data[:, 0], params_gen[:, 0]),
                "mu":    wasserstein_1D(params_data[:, 1], params_gen[:, 1]),
                "sigma": wasserstein_1D(params_data[:, 2], params_gen[:, 2]),
            }

        if save_w1:
            os.makedirs(w1_dir, exist_ok=True)
            w1_path = os.path.join(w1_dir, f"w1_multi_{data_label}_{stem}.json")
            with open(w1_path, "w", encoding="utf-8") as f:
                json.dump(w1, f, indent=2, ensure_ascii=False)

    # --- plot ---
    _gen_colors = ["#2CA25F", "#CB181D", "#8856A7", "#F16913"]  # green, red, purple, orange
    _params_names = [("theta", theta), ("mu", mu), ("sigma", sigma)]
    _true_vals    = [theta, mu, sigma]

    plt.rcParams.update(_MLE_RCPARAMS)
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))

    for col_i, (pname, true_val) in enumerate(_params_names):
        lat = _LATEX_PARAMS.get(pname, pname)
        ax  = axs[col_i]

        # Train KDE
        sns.kdeplot(ax=ax, data=params_data[:, col_i], fill=True,
                    color=_COLOR_TRAIN, alpha=0.5,
                    label=f"Train (N={len(params_data)})")

        # Gen KDE(s)
        for k, (glabel, pg) in enumerate(params_gen_map.items()):
            c = _gen_colors[k % len(_gen_colors)]
            sns.kdeplot(ax=ax, data=pg[:, col_i], fill=True,
                        color=c, alpha=0.5,
                        label=f"Gen (N={len(pg)})" if len(params_gen_map) == 1 else f"{glabel} (N={len(pg)})")

        if fix and true_val is not None:
            ax.axvline(true_val, color="black", linestyle="--")

        # W1 annotation per gen
        w1_lines = []
        for k, (glabel, pg) in enumerate(params_gen_map.items()):
            w1v     = wasserstein_1D(params_data[:, col_i], pg[:, col_i])
            std_ref = float(np.std(params_data[:, col_i]))
            w1_norm = (w1v / std_ref) if std_ref > 0 else float("nan")
            prefix  = "" if len(params_gen_map) == 1 else f"{glabel}: "
            w1_lines.append(f"{prefix}$W_1$={w1v:.2f}  $W_1^{{\\rm norm}}$={w1_norm:.2f}")
        ax.text(
            0.04, 0.96, "\n".join(w1_lines),
            transform=ax.transAxes,
            fontsize=label_fontsize - 2, va="top", ha="left", color="#333333",
            bbox=_BOX_STYLE,
        )

        ax.set_title(f"$\\hat{{{lat}}}$",        fontsize=title_fontsize)
        ax.set_xlabel(f"${lat}$",                 fontsize=label_fontsize)
        ax.set_ylabel("Empirical density (KDE)",  fontsize=label_fontsize)
        from matplotlib.ticker import AutoMinorLocator
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize,
                       direction="out", top=True, right=True)
        ax.tick_params(axis="both", which="minor",
                       direction="out", top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(
            fontsize=label_fontsize - 2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False,
        )

    # suptitle suppressed — context is given by the output folder and config

    plt.tight_layout()

    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
        full_plot_path = os.path.join(plot_dir, plot_path if plot_path.endswith(".png") else plot_path + ".png")
        fig.savefig(full_plot_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved plot at: {full_plot_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return w1



def MLE_CIR_robust_vect(params, X_data, dt):
    """
    Compute the MLE on Cox–Ingersoll–Ross (CIR) data using the exact
    noncentral chi-square transition density (vectorized version).

    :param params: [list or tuple]; parameters [θ, μ, sigma] of the CIR process.
    :param X_data: [np.ndarray]; observed time series of shape (N,).
    :param dt: [float]; time step between observations.

    Return: [float]; negative log-likelihood value (to be minimized).
    """
    theta, mu, sigma = params

    # Contraintes : CIR bien défini
    if theta <= 0 or sigma <= 0 or mu < 0:
        return 1e20

    # Condition de Feller (optionnelle mais tu l'as mise)
    if 2.0 * theta * mu < sigma**2:
        return 1e18

    # Convertir en np.array, forcer >= 0
    X = np.asarray(X_data, dtype=float)
    if X.size < 2:
        return 1e20  # pas assez de points

    X = np.clip(X, 0.0, None)
    x_t   = X[:-1]
    x_tp1 = X[1:]
    N = x_t.size

    # Sécurisation des termes numériques
    kappa = max(theta, 1e-12)
    vol   = max(sigma, 1e-12)
    mean  = max(mu, 0.0)

    exp_neg_kdt   = np.exp(-kappa * dt)
    one_minus_exp = 1.0 - exp_neg_kdt

    denom = (vol**2) * one_minus_exp
    if denom <= 0:
        return 1e20

    # Constantes de la transition exacte
    c = denom / (4.0 * kappa)
    d = 4.0 * kappa * mean / (vol**2)  # degrés de liberté

    if c <= 0 or d <= 0:
        return 1e20

    # Paramètres de la loi non centrale (vectorisés)
    lam = (4.0 * kappa * exp_neg_kdt * x_t) / denom
    z   = x_tp1 / c

    # Si des z < 0 apparaissent (numériquement), pénaliser
    if np.any(z < 0):
        return 1e20

    # Densité log du chi-2 non central, vectorisée
    log_c = np.log(c)

    try:
        log_pdf = ncx2.logpdf(z, d, lam) - log_c
    except Exception:
        return 1e20

    # Si NaN ou ±inf → pénalité
    if not np.all(np.isfinite(log_pdf)):
        return 1e20

    logL = np.sum(log_pdf)

    return -logL  # on minimise la négative log-vraisemblance


def MLE_CIR_robust(params, X_data, dt):
    """
    Compute the MLE on Cox–Ingersoll–Ross (CIR) data using the exact
    noncentral chi-square transition density.

    :param params: [list or tuple]; parameters [θ, μ, sigma] of the CIR process.
    :param X_data: [np.ndarray]; observed time series of shape (N,).
    :param dt: [float]; time step between observations.

    Return: [float]; negative log-likelihood value (to be minimized).
    """
    theta, mu, sigma = params

    if theta <= 0 or sigma <= 0 or mu < 0:
        return 1e20  # grosse pénalité

    if 2.0 * theta * mu < sigma**2:
        return 1e18

    N = len(X_data)
    logL = 0.0

    # Sécurisation des termes numériques
    kappa = max(theta, 1e-12)
    vol   = max(sigma, 1e-12)
    mean  = max(mu, 0.0)

    exp_neg_kdt = np.exp(-kappa * dt)
    one_minus_exp = 1.0 - exp_neg_kdt

    # Constantes de la transition exacte
    c = (vol**2) * one_minus_exp / (4.0 * kappa)
    d = 4.0 * kappa * mean / (vol**2)  # degrés de liberté

    if c <= 0 or d <= 0:
        return 1e20

    for t in range(N - 1):
        x_t   = X_data[t]
        x_tp1 = X_data[t + 1]

        # On force à >= 0 pour éviter les erreurs numériques
        x_t   = max(x_t, 0.0)
        x_tp1 = max(x_tp1, 0.0)

        denom = (vol**2 * one_minus_exp)
        if denom <= 0:
            return 1e20

        # Paramètre de non-centralité
        lam = (4.0 * kappa * exp_neg_kdt * x_t) / denom

        # Variable réduite
        z = x_tp1 / c

        if z < 0:
            return 1e20

        try:
            log_pdf = ncx2.logpdf(z, d, lam) - np.log(c)
        except Exception:
            return 1e20

        if not np.isfinite(log_pdf):
            return 1e20

        logL += log_pdf

    # On minimise la négative log-vraisemblance
    return -logL



def plot_params_distrib_CIR(
    X_data, X_gen, dt,
    fix=False,
    theta=None, mu=None, sigma=None,
    data_label="Data", gen_label="Gen",
    filter_outliers=True,
    save_img=True,
    plot_dir="plots",
    plot_path="params_distrib_CIR.png",
    param_dir="params",
    show=True,
    max_series=None,
    force=False,
    vectorized=True,
    suptitle=None
):
    # === Style cohérent avec plot_random_time_series ===
    title_fontsize = 18
    label_fontsize = 14
    tick_fontsize  = 14
    suptitle_size  = 20

    plt.rcParams.update({'font.size': label_fontsize})

    # ---------- Basic infos ----------
    N_data = len(X_data)
    N_gen  = len(X_gen)

    if max_series is not None:
        if N_data > max_series:
            idx_data = np.random.choice(N_data, max_series, replace=False)
            X_data   = X_data[idx_data]
            N_data   = len(X_data)
        if N_gen > max_series:
            idx_gen  = np.random.choice(N_gen, max_series, replace=False)
            X_gen    = X_gen[idx_gen]
            N_gen    = len(X_gen)

    # ---------- Paths for parameter cache ----------
    base = os.path.basename(plot_path)
    stem, _ = os.path.splitext(base)

    data_param_path = os.path.join(param_dir, f"{data_label}_{stem}.npy")
    gen_param_path  = os.path.join(param_dir, f"{gen_label}_{stem}.npy")

    # ---------- Try loading cached params ----------
    if os.path.exists(data_param_path) and os.path.exists(gen_param_path) and not force:
        print(f"[INFO] Loading cached CIR parameters from:")
        print(f"       {data_param_path}")
        print(f"       {gen_param_path}")
        params_data = np.load(data_param_path)
        params_gen  = np.load(gen_param_path)

    else:
        print("[INFO] No cached parameters found. Estimating CIR parameters...")
        print(f"       Will save to: {data_param_path} and {gen_param_path}")

        # ----- Estimate CIR params on real data -----
        params_data = np.zeros((N_data, 3))
        for m in range(N_data):
            series = X_data[m]
            mu_init    = max(np.mean(series), 1e-3)
            sigma_init = max(np.std(series), 1e-3)
            params_init_data = [1.0, mu_init, sigma_init]

            if vectorized:
                result_data = minimize(
                    MLE_CIR_robust_vect,
                    np.array(params_init_data),
                    args=(series, dt),
                    bounds=[
                        (1e-5, np.inf),   # theta > 0
                        (0.0,   np.inf),  # mu >= 0
                        (1e-5, np.inf)    # sigma > 0
                    ],
                    method='L-BFGS-B'
                )
            else:
                result_data = minimize(
                    MLE_CIR_robust,
                    np.array(params_init_data),
                    args=(series, dt),
                    bounds=[
                        (1e-5, np.inf),
                        (0.0,   np.inf),
                        (1e-5, np.inf)
                    ],
                    method='L-BFGS-B'
                )
            params_data[m] = result_data.x

        # ----- Estimate CIR params on generated data -----
        params_gen = np.zeros((N_gen, 3))
        for m in range(N_gen):
            series = X_gen[m]
            mu_init    = max(np.mean(series), 1e-3)
            sigma_init = max(np.std(series), 1e-3)
            params_init_gen = [1.0, mu_init, sigma_init]

            if vectorized:
                result_gen = minimize(
                    MLE_CIR_robust_vect,
                    np.array(params_init_gen),
                    args=(series, dt),
                    bounds=[
                        (1e-5, np.inf),
                        (0.0,   np.inf),
                        (1e-5, np.inf)
                    ],
                    method='L-BFGS-B'
                )
            else:
                result_gen = minimize(
                    MLE_CIR_robust,
                    np.array(params_init_gen),
                    args=(series, dt),
                    bounds=[
                        (1e-5, np.inf),
                        (0.0,   np.inf),
                        (1e-5, np.inf)
                    ],
                    method='L-BFGS-B'
                )
            params_gen[m] = result_gen.x

        # ----- Optional outlier filtering -----
        if filter_outliers:
            lb = np.percentile(params_data, 3, axis=0)
            ub = np.percentile(params_data, 97, axis=0)
            params_data = params_data[((params_data >= lb) & (params_data <= ub)).all(axis=1)]

            lb = np.percentile(params_gen, 3, axis=0)
            ub = np.percentile(params_gen, 97, axis=0)
            params_gen = params_gen[((params_gen >= lb) & (params_gen <= ub)).all(axis=1)]

        # ----- Save params to cache -----
        os.makedirs(param_dir, exist_ok=True)
        np.save(data_param_path, params_data)
        np.save(gen_param_path, params_gen)

        print(f"[INFO] Saved CIR parameters to:")
        print(f"       {data_param_path}")
        print(f"       {gen_param_path}")

    # ---------- Plot ----------
    plt.rcParams.update(_MLE_RCPARAMS)
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))

    _kw = dict(title_fontsize=title_fontsize, label_fontsize=label_fontsize, tick_fontsize=tick_fontsize, m_train=N_data, m_gen=N_gen)
    _mle_kdeplot(axs[0], params_data[:, 0], params_gen[:, 0], "theta", fix=fix, true_val=theta, **_kw)
    _mle_kdeplot(axs[1], params_data[:, 1], params_gen[:, 1], "mu",    fix=fix, true_val=mu,    **_kw)
    _mle_kdeplot(axs[2], params_data[:, 2], params_gen[:, 2], "sigma", fix=fix, true_val=sigma, **_kw)

    # --- Titre global optionnel ---
    # suptitle suppressed — context is given by the output folder and config

    plt.tight_layout()

    # ---------- Save plot ----------
    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"MLE_{data_label}_{stem}.png"
        full_plot_path = os.path.join(plot_dir, filename)
        fig.savefig(full_plot_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved CIR plot at: {full_plot_path}")

    # ---------- Show / close ----------
    if show:
        plt.show()
    else:
        plt.close(fig)



def MLE_Heston_robust(params, X, dt):
    """
    Compute the MLE on Heston data.
    :params params: parameters to estimate; [list]
    :params X: time series data; [np.array]
    :params dt: time step; [float]
    return: negative log-likelihood over X; [float]
    """
    kappa, theta, xi, rho, r = params
    N = len(X)
    logL = 0.0

    S = X[:, 0]
    v = X[:, 1]
    for t in range(N - 1):
        S_t, S_t_next = S[t], S[t + 1]
        v_t = v[t]

        mu_S = np.log(S_t) + (r - 0.5 * v_t) * dt
        mu_v = v_t + kappa * (theta - v_t) * dt

        var_S = v_t * dt
        var_v = xi ** 2 * v_t * dt

        cov_Sv = rho * xi * v_t * dt
        cov_matrix = np.array([[var_S, cov_Sv], [cov_Sv, var_v]])

        if np.linalg.det(cov_matrix) <= 0:
            return 1e10
        inv_cov = np.linalg.inv(cov_matrix)
        det_cov = np.linalg.det(cov_matrix)

        joint_observation = np.array([
            np.log(S_t_next) - mu_S,
            v[t + 1] - mu_v
        ])
        joint_log_pdf = -0.5 * (
                2 * np.log(2 * np.pi) + np.log(det_cov) + joint_observation.T @ inv_cov @ joint_observation
        )
        logL -= joint_log_pdf

    return logL


def MLE_Heston_robust_vect(params, X, dt):
    """
    Vectorized MLE on Heston data.

    :param params: [kappa, theta, xi, rho, r]
    :param X: array of shape (N, 2), columns [S_t, v_t]
    :param dt: time step
    :return: negative log-likelihood (float) to be minimized
    """
    kappa, theta, xi, rho, r = params

    # --- Sanity checks on params ---
    if kappa <= 0 or theta <= 0 or xi <= 0 or abs(rho) >= 1:
        return 1e20

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must be (N, 2), got {X.shape}")

    N = X.shape[0]
    if N < 2:
        return 1e20  # not enough points

    S = X[:, 0]
    v = X[:, 1]

    # Sécurité basique : prix positifs, variance non négative
    if np.any(S <= 0) or np.any(v < 0):
        return 1e20

    # -----------------------------------------
    # On travaille sur les paires (t, t+1) en vectoriel
    # -----------------------------------------
    S_t   = S[:-1]      # (N-1,)
    S_tp1 = S[1:]       # (N-1,)
    v_t   = v[:-1]      # (N-1,)
    v_tp1 = v[1:]       # (N-1,)

    # Moyennes conditionnelles
    mu_S = np.log(S_t) + (r - 0.5 * v_t) * dt
    mu_v = v_t + kappa * (theta - v_t) * dt

    # Variances / covariance
    var_S = v_t * dt                   # a
    var_v = (xi ** 2) * v_t * dt       # c
    cov_Sv = rho * xi * v_t * dt       # b

    # Matrice de covariance 2x2 = [[a, b], [b, c]]
    a = var_S
    b = cov_Sv
    c = var_v

    det = a * c - b ** 2               # det Σ_t

    # Si la covariance n'est pas définie positive → pénalité
    if np.any(det <= 0) or not np.all(np.isfinite(det)):
        return 1e20

    # Incréments (résidus)
    y1 = np.log(S_tp1) - mu_S          # (N-1,)
    y2 = v_tp1 - mu_v                  # (N-1,)

    # Quadratic form yᵀ Σ^{-1} y pour une 2x2 :
    # Σ^{-1} = 1/det * [[c, -b], [-b, a]]
    # yᵀ Σ^{-1} y = (1/det) * (c y1^2 - 2 b y1 y2 + a y2^2)
    quad = (c * y1**2 - 2.0 * b * y1 * y2 + a * y2**2) / det

    # log det Σ_t
    log_det = np.log(det)

    # Nombre d'observations (transitions)
    M = N - 1

    # log-likelihood total (somme sur t)
    # log p = -0.5 * (2 log(2π) + log det + quad)
    # NLL = -∑ log p = 0.5 * ∑ (2 log(2π) + log det + quad)
    const = 2.0 * np.log(2.0 * np.pi)
    nll = 0.5 * np.sum(const + log_det + quad)

    if not np.isfinite(nll):
        return 1e20

    return nll


def plot_params_distrib_Heston(
    X_data, X_gen, dt,
    fix=False,
    kappa=None, theta=None, xi=None, rho=None, r=None,
    data_label="Data", gen_label="Gen",
    filter_outliers=True,
    save_img=True,
    plot_dir="plots",
    plot_path="params_distrib_heston.png",
    param_dir="params_heston",
    show=True,
    force=False,
    returns=False,
    returns_type='log',
    vectorized=True,
    suptitle=None,      # <<< NOUVEL ARGUMENT POUR LE TITRE GLOBAL
    params1=None, params2=None, params3=None, params4=None, params5=None,
    
):
    """
    Estimate and visualize Heston parameter distributions for real and generated series.
    If parameter files already exist, they are loaded instead of recomputed.

    X_data : (N_data, T, 2)  with columns [S_t, v_t]
    X_gen  : (N_gen,  T, 2)  same format
    """

    # === Style cohérent avec plot_random_time_series / OU / CIR ===
    title_fontsize = 18
    label_fontsize = 14
    tick_fontsize  = 14
    suptitle_size  = 20

    plt.rcParams.update({'font.size': label_fontsize})
    params1 = params1 or {}
    params2 = params2 or {}
    params3 = params3 or {}
    params4 = params4 or {}
    params5 = params5 or {}
    

    # ---------- Basic infos ----------
    N_data = len(X_data)
    N_gen  = len(X_gen)

    # ---------- Paths / labels ----------
    base = os.path.basename(plot_path)
    stem, _ = os.path.splitext(base)

    # Adapter les labels si on travaille en rendements
    if returns:
        if returns_type == 'log':
            data_label = f"logr{data_label}"
            gen_label  = f"logr{gen_label}"
        elif returns_type == 'simple':
            data_label = f"r{data_label}"
            gen_label  = f"r{gen_label}"

    # fichiers de paramètres à sauver/charger
    data_filename = safe_filename(f"{data_label}_{stem}")
    gen_filename  = safe_filename(f"{gen_label}_{stem}")
    
    data_param_path = os.path.join(param_dir, data_filename)
    gen_param_path  = os.path.join(param_dir, gen_filename)

    # ---------- Try loading cached params ----------
    if os.path.exists(data_param_path) and os.path.exists(gen_param_path) and not force:
        print(f"[INFO] Loading cached Heston parameters from:")
        print(f"       {data_param_path}")
        print(f"       {gen_param_path}")
        params_data = np.load(data_param_path)
        params_gen  = np.load(gen_param_path)

    else:
        print("[INFO] No cached Heston parameters found or force=True. Estimating...")

        bounds = [
            (1e-6, None),   # kappa > 0
            (1e-6, None),   # theta > 0
            (1e-6, None),   # xi > 0
            (-1.0, 1.0),    # rho in [-1, 1]
            (None, None),   # r unrestricted
        ]

        if vectorized:
            MLE_fn = MLE_Heston_robust_vect
        else:
            MLE_fn = MLE_Heston_robust

        # ----- Estimate Heston params on real data -----
        params_data = np.zeros((N_data, 5))
        for m in range(N_data):
            # Initialisation (modifiable si vraies valeurs connues)
            x0 = np.array([3.0, 0.5, 0.7, 0.7, 0.02])
            res = minimize(
                MLE_fn,
                x0=x0,
                args=(X_data[m], dt),
                bounds=bounds,
                method='L-BFGS-B'
            )
            params_data[m] = res.x

        # ----- Estimate Heston params on generated data -----
        params_gen = np.zeros((N_gen, 5))
        for m in range(N_gen):
            x0 = np.array([3.0, 0.5, 0.7, 0.7, 0.02])
            res = minimize(
                MLE_fn,
                x0=x0,
                args=(X_gen[m], dt),
                bounds=bounds,
                method='L-BFGS-B'
            )
            params_gen[m] = res.x

        # ----- Optional outlier filtering (comme pour OU) -----
        if filter_outliers:
            lb_data = np.percentile(params_data, 3, axis=0)
            ub_data = np.percentile(params_data, 97, axis=0)
            params_data = params_data[((params_data >= lb_data) & (params_data <= ub_data)).all(axis=1)]

            lb_gen = np.percentile(params_gen, 3, axis=0)
            ub_gen = np.percentile(params_gen, 97, axis=0)
            params_gen = params_gen[((params_gen >= lb_gen) & (params_gen <= ub_gen)).all(axis=1)]

        # ----- Save params to cache -----
        os.makedirs(param_dir, exist_ok=True)
        np.save(data_param_path, params_data)
        np.save(gen_param_path, params_gen)

        print(f"[INFO] Saved Heston parameters to:")
        print(f"       {data_param_path}")
        print(f"       {gen_param_path}")

    # ---------- Plot ----------
    _heston_params = ["kappa", "theta", "xi", "rho", "r"]
    true_vals    = [kappa, theta, xi, rho, r]
    params_boxes = [params1, params2, params3, params4, params5]
    _kw = dict(title_fontsize=title_fontsize, label_fontsize=label_fontsize, tick_fontsize=tick_fontsize, m_train=N_data, m_gen=N_gen)

    plt.rcParams.update(_MLE_RCPARAMS)
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))

    for i in range(5):
        ax = axs[i // 3, i % 3]
        _mle_kdeplot(ax, params_data[:, i], params_gen[:, i], _heston_params[i],
                     fix=fix, true_val=true_vals[i], **_kw)

        _set_gt_title(ax, params_boxes[i], label_fontsize)
    

    # Dernier subplot vide
    axs[1, 2].axis('off')

    # --- Titre global optionnel ---
    # suptitle suppressed — context is given by the output folder and config

    plt.tight_layout()

    # ---------- Save plot ----------
    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"MLE_Heston_{data_label}_{stem}.png"
        full_plot_path = os.path.join(plot_dir, filename)
        fig.savefig(safe_filename(full_plot_path, ext=".png"), dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved Heston plot at: {full_plot_path}")

    # ---------- Show / close ----------
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_linreg_params_distrib(
    X_data,
    X_gen,
    fix=False,
    a_true=None,
    b_true=None,
    data_label="Data",
    gen_label="Gen",
    filter_outliers=True,
    save_img=True,
    plot_dir="plots",
    plot_path="linreg_params_distrib.png",
    param_dir="params",
    show=True,
    force=False,
    return_params=False,
    params1=None, params2=None,
    suptitle=None   # <<< NOUVEL ARGUMENT
):
    # === Style cohérent avec les autres fonctions ===
    title_fontsize = 18
    label_fontsize = 14
    tick_fontsize  = 14
    suptitle_size  = 20

    plt.rcParams.update({'font.size': label_fontsize})

    # ---------- Préparation des données ----------
    X_data = np.asarray(X_data)
    X_gen  = np.asarray(X_gen)

    params1 = params1 or {}
    params2 = params2 or {}

    # On squeeze une éventuelle dimension de taille 1 en tête (p.ex. (1, N, T))
    if X_data.ndim == 3 and X_data.shape[0] == 1:
        X_data = np.squeeze(X_data, axis=0)
    if X_gen.ndim == 3 and X_gen.shape[0] == 1:
        X_gen = np.squeeze(X_gen, axis=0)

    # Si 1D -> une seule série
    if X_data.ndim == 1:
        X_data = X_data[None, :]
    if X_gen.ndim == 1:
        X_gen = X_gen[None, :]

    N_data, T_data = X_data.shape
    N_gen,  T_gen  = X_gen.shape
    assert T_data == T_gen, "X_data et X_gen doivent avoir la même longueur temporelle."
    x = np.arange(T_data)

    # ---------- Gestion des chemins pour le cache ----------
    base = os.path.basename(plot_path)
    stem, _ = os.path.splitext(base)

    data_param_path = os.path.join(param_dir, f"{data_label}_{stem}.npy")
    gen_param_path  = os.path.join(param_dir, f"{gen_label}_{stem}.npy")

    X_data_fit = np.zeros_like(X_data, dtype=float)
    X_gen_fit  = np.zeros_like(X_gen, dtype=float)

    # ---------- Chargement éventuel du cache ----------
    if os.path.exists(data_param_path) and os.path.exists(gen_param_path) and not force:
        print("[INFO] Loading cached linreg parameters from:")
        print(f"       {data_param_path}")
        print(f"       {gen_param_path}")
        params_data = np.load(data_param_path)
        params_gen  = np.load(gen_param_path)

        params_data[:, 0] = params_data[:, 0] / T_data
        params_gen[:, 0]  = params_gen[:, 0]  / T_gen

        a_data = params_data[:, 0]
        b_data = params_data[:, 1]
        a_gen  = params_gen[:, 0]
        b_gen  = params_gen[:, 1]

        for m in range(len(params_data)):
            X_data_fit[m] = a_data[m] * x + b_data[m]
        for m in range(len(params_gen)):
            X_gen_fit[m] = a_gen[m] * x + b_gen[m]

    else:
        print("[INFO] No cached linreg parameters found. Estimating (a, b)...")

        # ----- Estimation (a, b) sur X_data -----
        params_data = np.zeros((N_data, 2))
        for m in range(N_data):
            y = X_data[m]
            a, b = np.polyfit(x, y, 1)  # a = pente, b = intercept
            params_data[m] = [a * T_data, b]
            X_data_fit[m] = a * x + b

        # ----- Estimation (a, b) sur X_gen -----
        params_gen = np.zeros((N_gen, 2))
        for m in range(N_gen):
            y = X_gen[m]
            a, b = np.polyfit(x, y, 1)
            params_gen[m] = [a * T_gen, b]
            X_gen_fit[m] = a * x + b

        # ----- Filtrage des outliers (optionnel) -----
        if filter_outliers:
            lb = np.percentile(params_data, 3, axis=0)
            ub = np.percentile(params_data, 97, axis=0)
            mask_data = ((params_data >= lb) & (params_data <= ub)).all(axis=1)
            params_data = params_data[mask_data]
            X_data_fit  = X_data_fit[mask_data]

            lb = np.percentile(params_gen, 3, axis=0)
            ub = np.percentile(params_gen, 97, axis=0)
            mask_gen = ((params_gen >= lb) & (params_gen <= ub)).all(axis=1)
            params_gen = params_gen[mask_gen]
            X_gen_fit  = X_gen_fit[mask_gen]

        # ----- Sauvegarde dans le cache -----
        os.makedirs(param_dir, exist_ok=True)
        np.save(data_param_path, params_data)
        np.save(gen_param_path, params_gen)

        print("[INFO] Saved linreg parameters to:")
        print(f"       {data_param_path}")
        print(f"       {gen_param_path}")
        

    # ---------- Plot des distributions ----------
    plt.rcParams.update(_MLE_RCPARAMS)
    fig, axs = plt.subplots(1, 2, figsize=(21, 6))

    _kw = dict(title_fontsize=title_fontsize, label_fontsize=label_fontsize, tick_fontsize=tick_fontsize, m_train=N_data, m_gen=N_gen)
    _mle_kdeplot(axs[0], params_data[:, 0], params_gen[:, 0], "a", fix=fix, true_val=a_true, **_kw)
    _mle_kdeplot(axs[1], params_data[:, 1], params_gen[:, 1], "b", fix=fix, true_val=b_true, **_kw)

    _set_gt_title(axs[0], params1, label_fontsize)
    _set_gt_title(axs[1], params2, label_fontsize)

    # --- Titre global optionnel ---
    # suptitle suppressed — context is given by the output folder and config

    plt.tight_layout()

    # ---------- Sauvegarde ----------
    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"linreg_{data_label}_{stem}.png"
        full_plot_path = os.path.join(plot_dir, filename)
        fig.savefig(full_plot_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved plot at: {full_plot_path}")

    # ---------- Affichage / fermeture ----------
    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_params:
        return params_data, params_gen, X_data_fit, X_gen_fit


def estimate_linreg_params(X, filter_outliers=True, scale_a_by_T=True):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim == 3 and X.shape[0] == 1:
        X = np.squeeze(X, axis=0)

    M, T = X.shape
    x = np.arange(T)

    # polyfit série par série
    params = np.zeros((M, 2), dtype=float)
    for m in range(M):
        a, b = np.polyfit(x, X[m], 1)
        if scale_a_by_T:
            a = a * T
        params[m] = [a, b]

    if filter_outliers and len(params) > 10:
        lb = np.percentile(params, 3, axis=0)
        ub = np.percentile(params, 97, axis=0)
        mask = ((params >= lb) & (params <= ub)).all(axis=1)
        params = params[mask]

    return params


def plot_linreg_params_distrib_multi(
    params_data,                 # (M,2)
    params_gen_dict,             # dict: label -> (M,2)
    fix=False,
    a_true=None,
    b_true=None,
    data_label="Data",
    filter_outliers=False,       # déjà fait avant en général
    save_img=True,
    plot_dir="plots",
    plot_path="linreg_params_multi.png",
    show=True,
    suptitle=None,
):
    title_fontsize = 18
    label_fontsize = 14
    tick_fontsize  = 14
    suptitle_size  = 20
    plt.rcParams.update({'font.size': label_fontsize})

    params_data = np.asarray(params_data)

    plt.rcParams.update(_MLE_RCPARAMS)
    fig, axs = plt.subplots(1, 2, figsize=(21, 6))

    _gen_colors = ["#2CA25F", "#CB181D", "#8856A7", "#F16913"]
    _line_params = [("a", a_true), ("b", b_true)]

    for col_i, (pname, true_val) in enumerate(_line_params):
        lat = _LATEX_PARAMS.get(pname, pname)
        ax  = axs[col_i]

        sns.kdeplot(ax=ax, data=params_data[:, col_i], fill=True,
                    color=_COLOR_TRAIN, alpha=0.5,
                    label=f"Train (N={len(params_data)})")
        for k, (lab, p) in enumerate(params_gen_dict.items()):
            p = np.asarray(p)
            c = _gen_colors[k % len(_gen_colors)]
            sns.kdeplot(ax=ax, data=p[:, col_i], fill=True, color=c, alpha=0.5,
                        label=f"Gen (N={len(p)})" if len(params_gen_dict) == 1 else f"{lab} (N={len(p)})")

        if fix and true_val is not None:
            ax.axvline(true_val, color="black", linestyle="--")

        w1_lines = []
        for lab, p in params_gen_dict.items():
            p = np.asarray(p)
            w1v     = wasserstein_1D(params_data[:, col_i], p[:, col_i])
            std_ref = float(np.std(params_data[:, col_i]))
            w1n     = (w1v / std_ref) if std_ref > 0 else float("nan")
            prefix  = "" if len(params_gen_dict) == 1 else f"{lab}: "
            w1_lines.append(f"{prefix}$W_1$={w1v:.2f}  $W_1^{{\\rm norm}}$={w1n:.2f}")
        ax.text(0.04, 0.96, "\n".join(w1_lines), transform=ax.transAxes,
                fontsize=label_fontsize - 2, va="top", ha="left", color="#333333",
                bbox=_BOX_STYLE)

        ax.set_title(f"$\\hat{{{lat}}}$",        fontsize=title_fontsize)
        ax.set_xlabel(f"${lat}$",                 fontsize=label_fontsize)
        ax.set_ylabel("Empirical density (KDE)",  fontsize=label_fontsize)
        from matplotlib.ticker import AutoMinorLocator
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize,
                       direction="out", top=True, right=True)
        ax.tick_params(axis="both", which="minor",
                       direction="out", top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(
            fontsize=label_fontsize - 2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False,
        )

    # suptitle suppressed — context is given by the output folder and config

    plt.tight_layout()

    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
        full_plot_path = os.path.join(plot_dir, plot_path)
        fig.savefig(full_plot_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved plot at: {full_plot_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


from scipy.optimize import minimize_scalar

def _fit_sine_1d(y, t):
    """
    Fit y(t) ~= A * sin(2*pi*f*t + phi) by least squares, de manière stable.
    Retourne (A, f, phi).

    Méthode:
      - pour un f donné: régression linéaire sur [sin(2πft), cos(2πft), 1]
      - puis recherche 1D de f (minimize_scalar) autour d'une initialisation FFT
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    # --- centrage pour une init FFT plus propre ---
    y_demean = y - y.mean()

    # --- pas de temps (supposé uniforme) ---
    dt = t[1] - t[0]
    if dt <= 0:
        raise ValueError("t doit être strictement croissant.")

    # --- 1) Initialisation f0 via FFT (avec zero-padding pour meilleure résolution) ---
    n = len(t)
    # zero-padding pour sur-échantillonner le spectre
    n_fft = 1
    while n_fft < 8 * n:
        n_fft *= 2

    freqs_fft = np.fft.rfftfreq(n_fft, d=dt)
    fft_vals = np.fft.rfft(y_demean, n=n_fft)

    # ignorer DC
    k_max = np.argmax(np.abs(fft_vals[1:])) + 1
    f0 = float(freqs_fft[k_max]) if freqs_fft[k_max] > 0 else 0.0

    # bornes raisonnables pour f
    f_nyq = 0.5 / dt  # Nyquist
    # si f0 tombe à 0 par accident, on prend une petite valeur positive
    if f0 <= 0:
        f0 = max(freqs_fft[1], 1e-6)

    # --- helper: solve LS pour un f donné ---
    def _ls_for_f(f):
        # design matrix: a*sin + b*cos + c
        w = 2.0 * np.pi * f
        s = np.sin(w * t)
        c = np.cos(w * t)
        X = np.column_stack([s, c, np.ones_like(t)])
        # moindres carrés
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)  # [a, b, offset]
        a, b, off = beta
        y_hat = X @ beta
        sse = float(np.sum((y_hat - y) ** 2))
        return sse, a, b, off

    # --- 2) objectif 1D sur f (SSE après projection LS) ---
    def objective_f(f):
        # contraindre numériquement
        if f <= 0 or f > f_nyq:
            return np.inf
        sse, _, _, _ = _ls_for_f(f)
        return sse

    # --- 3) intervalle de recherche autour de f0 ---
    # largeur: dépend de la résolution fréquentielle effective
    # ici: on prend une fenêtre proportionnelle (et bornée) autour de f0
    width = max(0.5 * f0, 2.0)   # assez large pour éviter de rater le bon bassin
    f_lo = max(1e-8, f0 - width)
    f_hi = min(f_nyq, f0 + width)

    # (sécurité) si intervalle dégénéré
    if f_hi <= f_lo + 1e-12:
        f_lo = max(1e-8, f0 * 0.5)
        f_hi = min(f_nyq, f0 * 1.5)

    # --- 4) minimisation scalaire de f ---
    res = minimize_scalar(
        objective_f,
        bounds=(f_lo, f_hi),
        method="bounded",
        options={"xatol": 1e-10}
    )
    f_hat = float(res.x)

    # --- 5) récupérer a,b -> A,phi ---
    _, a_hat, b_hat, _ = _ls_for_f(f_hat)
    A_hat = float(np.hypot(a_hat, b_hat))  # sqrt(a^2+b^2)

    # Convention: y ≈ a*sin(wt) + b*cos(wt)
    # Or A*sin(wt+phi) = A*cos(phi)*sin(wt) + A*sin(phi)*cos(wt)
    # donc a = A*cos(phi), b = A*sin(phi) => phi = atan2(b,a)
    phi_hat = float(np.arctan2(b_hat, a_hat))

    # --- 6) normalisation optionnelle de phi (cohérent avec tes plots) ---
    # Ici je le remets dans [-pi, pi], puis tu peux le remapper ailleurs si besoin.
    # Si tu préfères [0, 2pi): décommente la ligne suivante.
    # phi_hat = (phi_hat + 2*np.pi) % (2*np.pi)
    # Sinon, on laisse dans [-pi, pi] (stable pour l'optimisation et l'interprétation)
    # Pour coller à ton ancien bound [-2pi,2pi], aucun souci.

    # éviter A=0 exact (sinon phi non identifiable)
    if A_hat < 1e-12:
        A_hat = 0.0
        phi_hat = 0.0

    return np.array([A_hat, f_hat, phi_hat], dtype=float)


def plot_sine_params_distrib_1D(
    X_data,
    X_gen,
    fix=False,
    A_true=None,
    f_true=None,
    phi_true=None,
    data_label="Data",
    gen_label="Gen",
    filter_outliers=True,
    save_img=True,
    plot_dir="plots",
    plot_path="sine_params_distrib.png",
    param_dir="params",
    show=True,
    force=False,
    return_params=False,
    params1=None,
    params2=None,
    params3=None,
    suptitle=None,
):
    # === Style cohérent avec les autres fonctions ===
    title_fontsize = 2/3*18
    label_fontsize = 2/3*14
    tick_fontsize  = 2/3*14
    suptitle_size  = 2/3*20

    plt.rcParams.update({'font.size': label_fontsize})

    # ---------- Mise en forme des données ----------
    X_data = np.asarray(X_data)
    X_gen  = np.asarray(X_gen)

    params1 = params1 or {}
    params2 = params2 or {}
    params3 = params3 or {}

    # enlever une éventuelle dimension de taille 1 en tête
    if X_data.ndim == 3 and X_data.shape[0] == 1:
        X_data = np.squeeze(X_data, axis=0)
    if X_gen.ndim == 3 and X_gen.shape[0] == 1:
        X_gen = np.squeeze(X_gen, axis=0)

    if X_data.ndim == 1:
        X_data = X_data[None, :]
    if X_gen.ndim == 1:
        X_gen = X_gen[None, :]

    N_data, T_data = X_data.shape
    N_gen,  T_gen  = X_gen.shape
    assert T_data == T_gen, "X_data et X_gen doivent avoir la même longueur temporelle."

    # temps sur [0,1]
    t = np.linspace(0, 1, T_data)

    # ---------- Gestion des chemins de cache ----------
    base = os.path.basename(plot_path)
    stem, _ = os.path.splitext(base)

    data_param_path = os.path.join(param_dir, f"{data_label}_{stem}.npy")
    gen_param_path  = os.path.join(param_dir, f"{gen_label}_{stem}.npy")

    X_data_fit = np.zeros_like(X_data, dtype=float)
    X_gen_fit  = np.zeros_like(X_gen, dtype=float)

    # ---------- Chargement ou estimation des paramètres ----------
    if os.path.exists(data_param_path) and os.path.exists(gen_param_path) and not force:
        print("[INFO] Loading cached sine parameters from:")
        print(f"       {data_param_path}")
        print(f"       {gen_param_path}")
        params_data = np.load(data_param_path)
        params_gen  = np.load(gen_param_path)

        # Reconstruire les sinusoïdes à partir des paramètres
        def model(t, A, f, phi):
            return A * np.sin(2 * np.pi * f * t + phi)

        for m in range(len(params_data)):
            A, f, phi = params_data[m]
            X_data_fit[m] = model(t, A, f, phi)
        for m in range(len(params_gen)):
            A, f, phi = params_gen[m]
            X_gen_fit[m] = model(t, A, f, phi)

    else:
        print("[INFO] No cached sine parameters found. Estimating (A, f, phi)...")

        # ----- X_data -----
        params_data = np.zeros((N_data, 3))
        for m in range(N_data):
            A, f, phi = _fit_sine_1d(X_data[m], t)  # (A, f, phi)
            params_data[m] = [A, f, phi]
            X_data_fit[m] = A * np.sin(2 * np.pi * f * t + phi)

        # ----- X_gen -----
        params_gen = np.zeros((N_gen, 3))
        for m in range(N_gen):
            A, f, phi = _fit_sine_1d(X_gen[m], t)
            params_gen[m] = [A, f, phi]
            X_gen_fit[m] = A * np.sin(2 * np.pi * f * t + phi)

        # ----- Filtrage des outliers (optionnel) -----
        if filter_outliers:
            lb = np.percentile(params_data, 3, axis=0)
            ub = np.percentile(params_data, 97, axis=0)
            mask_data = ((params_data >= lb) & (params_data <= ub)).all(axis=1)
            params_data = params_data[mask_data]
            X_data_fit  = X_data_fit[mask_data]

            lb = np.percentile(params_gen, 3, axis=0)
            ub = np.percentile(params_gen, 97, axis=0)
            mask_gen = ((params_gen >= lb) & (params_gen <= ub)).all(axis=1)
            params_gen = params_gen[mask_gen]
            X_gen_fit  = X_gen_fit[mask_gen]

        # ----- Sauvegarde dans le cache -----
        os.makedirs(param_dir, exist_ok=True)
        np.save(data_param_path, params_data)
        np.save(gen_param_path, params_gen)

        print("[INFO] Saved sine parameters to:")
        print(f"       {data_param_path}")
        print(f"       {gen_param_path}")

    # ---------- Plots KDE ----------
    plt.rcParams.update(_MLE_RCPARAMS)
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))

    _kw = dict(title_fontsize=title_fontsize, label_fontsize=label_fontsize, tick_fontsize=tick_fontsize, m_train=N_data, m_gen=N_gen)
    _mle_kdeplot(axs[0], params_data[:, 0], params_gen[:, 0], "A",   fix=fix, true_val=A_true,   **_kw)
    _mle_kdeplot(axs[1], params_data[:, 1], params_gen[:, 1], "f",   fix=fix, true_val=f_true,   **_kw)
    _mle_kdeplot(axs[2], params_data[:, 2], params_gen[:, 2], "phi", fix=fix, true_val=phi_true, **_kw)

    _set_gt_title(axs[0], params1, label_fontsize)
    _set_gt_title(axs[1], params2, label_fontsize)
    _set_gt_title(axs[2], params3, label_fontsize)

    # --- Titre global optionnel ---
    # suptitle suppressed — context is given by the output folder and config

    plt.tight_layout()

    # ---------- Sauvegarde ----------
    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"sinefit_{data_label}_{stem}.png"
        full_plot_path = os.path.join(plot_dir, filename)
        fig.savefig(full_plot_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved plot at: {full_plot_path}")

    # ---------- Affichage / fermeture ----------
    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_params:
        return params_data, params_gen, X_data_fit, X_gen_fit


def simulate_linear_ode_from_params(t, dt, x0, a0, a1, b):
    N = len(t)
    x = np.zeros(N)
    x[0] = x0
    for n in range(N-1):
        x[n+1] = x[n] + dt*((a0 + a1*t[n])*x[n] + b)
    return x

def fit_linear_ode(y, t, dt):
    """
    Fit des paramètres (x0, a0, a1, b) par moindres carrés.
    """
    N = len(t)

    # Initial guesses
    x0_0 = y[0]
    a0_0 = 0.0
    a1_0 = 0.0
    b_0  = (y[1]-y[0]) / dt if dt>0 else 0.0

    p0 = np.array([x0_0, a0_0, a1_0, b_0])

    # Bound: x0 free, a0 free, a1 free, b free
    bounds = [(None,None),(None,None),(None,None),(None,None)]

    def objective(params):
        x0, a0, a1, b = params
        x_pred = simulate_linear_ode_from_params(t, dt, x0, a0, a1, b)
        return np.sum((x_pred - y)**2)

    res = minimize(objective, p0, bounds=bounds, method="L-BFGS-B")
    return res.x  # (x0, a0, a1, b)


def plot_ode_params_distrib_1D(
    X_data,
    X_gen,
    dt,
    fix=False,
    x0_true=None,
    a0_true=None,
    a1_true=None,
    b_true=None,
    data_label="Data",
    gen_label="Gen",
    filter_outliers=True,
    save_img=True,
    plot_dir="plots",
    plot_path="ode_params_distrib.png",
    param_dir="params",
    show=True,
    force=False,
    return_params=False,
    suptitle=None,   # <<< NOUVEL ARGUMENT
):
    # === Style cohérent avec les autres fonctions ===
    title_fontsize = 18
    label_fontsize = 14
    tick_fontsize  = 14
    suptitle_size  = 20

    plt.rcParams.update({'font.size': label_fontsize})

    # ---------- Formatage ----------
    X_data = np.asarray(X_data)
    X_gen  = np.asarray(X_gen)

    if X_data.ndim == 3 and X_data.shape[0] == 1:
        X_data = X_data[0]
    if X_gen.ndim == 3 and X_gen.shape[0] == 1:
        X_gen = X_gen[0]

    if X_data.ndim == 1:
        X_data = X_data[None, :]
    if X_gen.ndim == 1:
        X_gen = X_gen[None, :]

    N_data, T_data = X_data.shape
    N_gen,  T_gen  = X_gen.shape
    assert T_data == T_gen
    t = np.arange(T_data) * dt

    # ---------- Paths ----------
    base = os.path.basename(plot_path)
    stem, _ = os.path.splitext(base)
    data_param_path = os.path.join(param_dir, f"{data_label}_{stem}.npy")
    gen_param_path  = os.path.join(param_dir, f"{gen_label}_{stem}.npy")

    X_data_fit = np.zeros_like(X_data)
    X_gen_fit  = np.zeros_like(X_gen)

    # ---------- Load cache ----------
    if os.path.exists(data_param_path) and os.path.exists(gen_param_path) and not force:
        print("[INFO] Loading cached ODE parameters")

        params_data = np.load(data_param_path)
        params_gen  = np.load(gen_param_path)

        # Reconstruire les séries lissées
        for i, p in enumerate(params_data):
            x0, a0, a1, b = p
            X_data_fit[i] = simulate_linear_ode_from_params(t, dt, x0, a0, a1, b)

        for i, p in enumerate(params_gen):
            x0, a0, a1, b = p
            X_gen_fit[i] = simulate_linear_ode_from_params(t, dt, x0, a0, a1, b)

    else:
        print("[INFO] Fitting ODE parameters...")

        # ------ Fit Data ------
        params_data = np.zeros((N_data, 4))
        for i in range(N_data):
            params_data[i] = fit_linear_ode(X_data[i], t, dt)
            x0, a0, a1, b = params_data[i]
            X_data_fit[i] = simulate_linear_ode_from_params(t, dt, x0, a0, a1, b)

        # ------ Fit Gen ------
        params_gen = np.zeros((N_gen, 4))
        for i in range(N_gen):
            params_gen[i] = fit_linear_ode(X_gen[i], t, dt)
            x0, a0, a1, b = params_gen[i]
            X_gen_fit[i] = simulate_linear_ode_from_params(t, dt, x0, a0, a1, b)

        # ------ Filter outliers ------
        if filter_outliers:
            lb = np.percentile(params_data, 3, axis=0)
            ub = np.percentile(params_data, 97, axis=0)
            m = ((params_data >= lb) & (params_data <= ub)).all(axis=1)
            params_data = params_data[m]
            X_data_fit  = X_data_fit[m]

            lb = np.percentile(params_gen, 3, axis=0)
            ub = np.percentile(params_gen, 97, axis=0)
            m = ((params_gen >= lb) & (params_gen <= ub)).all(axis=1)
            params_gen = params_gen[m]
            X_gen_fit  = X_gen_fit[m]

        # Save
        os.makedirs(param_dir, exist_ok=True)
        np.save(data_param_path, params_data)
        np.save(gen_param_path, params_gen)

    # ---------- Plots ----------
    plt.rcParams.update(_MLE_RCPARAMS)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    names  = ["x0", "a0", "a1", "b"]
    truths = [x0_true, a0_true, a1_true, b_true]
    _kw = dict(title_fontsize=title_fontsize, label_fontsize=label_fontsize, tick_fontsize=tick_fontsize, m_train=N_data, m_gen=N_gen)

    for j in range(4):
        _mle_kdeplot(axs[j], params_data[:, j], params_gen[:, j], names[j],
                     fix=fix, true_val=truths[j], **_kw)

    # --- Titre global optionnel ---
    # suptitle suppressed — context is given by the output folder and config

    plt.tight_layout()

    # Save
    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
        out = os.path.join(plot_dir, f"ODE_{data_label}_{stem}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("[INFO] Saved plot to", out)

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_params:
        return params_data, params_gen, X_data_fit, X_gen_fit


def _mle_gbm_mu_sigma(x, dt):
    """
    MLE de (mu, sigma) pour un GBM à partir d'une série x(t).
    x doit être strictement positive.
    """
    x = np.asarray(x)
    if np.any(x <= 0):
        raise ValueError("GBM MLE: all values must be > 0.")

    logx = np.log(x)
    r = np.diff(logx)          # log-returns
    m = np.mean(r)
    v = np.mean((r - m) ** 2)  # variance MLE (1/n)

    sigma2_hat = v / dt
    sigma_hat = np.sqrt(sigma2_hat)
    mu_hat = m / dt + 0.5 * sigma2_hat

    return np.array([mu_hat, sigma_hat])



def plot_params_distrib_GBM(
    X_data,
    X_gen,
    dt,
    fix=False,
    mu=None,
    sigma=None,
    data_label="Data",
    gen_label="Gen",
    filter_outliers=True,
    save_img=True,
    plot_dir="plots",
    plot_path="GBM_params_distrib.png",
    param_dir="params",
    show=True,
    force=False,
    return_params=False,
    suptitle=None,   # <<< NOUVEL ARGUMENT
):
    """
    Estime (mu, sigma) d'un GBM pour chaque série de X_data et X_gen
    via MLE sur les log-retours, puis trace les distributions KDE.
    """

    # === Style cohérent avec les autres fonctions ===
    title_fontsize = 18
    label_fontsize = 14
    tick_fontsize  = 14
    suptitle_size  = 20

    plt.rcParams.update({'font.size': label_fontsize})

    # --------- Mise en forme des données ----------
    X_data = np.asarray(X_data)
    X_gen  = np.asarray(X_gen)

    # supprimer éventuelle dimension de tête de taille 1
    if X_data.ndim == 3 and X_data.shape[0] == 1:
        X_data = X_data[0]
    if X_gen.ndim == 3 and X_gen.shape[0] == 1:
        X_gen = X_gen[0]

    if X_data.ndim == 1:
        X_data = X_data[None, :]
    if X_gen.ndim == 1:
        X_gen = X_gen[None, :]

    N_data, T_data = X_data.shape
    N_gen,  T_gen  = X_gen.shape
    assert T_data == T_gen, "X_data et X_gen doivent avoir la même longueur."

    # --------- Gestion des chemins de cache ----------
    base = os.path.basename(plot_path)
    stem, _ = os.path.splitext(base)

    if not os.path.isdir(param_dir):
        os.makedirs(param_dir, exist_ok=True)

    data_param_path = os.path.join(param_dir, f"{data_label}_{stem}.npy")
    gen_param_path  = os.path.join(param_dir,  f"{gen_label}_{stem}.npy")

    # --------- Chargement éventuel du cache ----------
    if os.path.exists(data_param_path) and os.path.exists(gen_param_path) and not force:
        print("[INFO] Loading cached GBM parameters from:")
        print(f"       {data_param_path}")
        print(f"       {gen_param_path}")
        params_data = np.load(data_param_path)
        params_gen  = np.load(gen_param_path)

    else:
        print("[INFO] No cached GBM parameters found. Estimating (mu, sigma)...")

        # ----- Estimation (mu, sigma) sur X_data -----
        params_data = np.zeros((N_data, 2))
        for m in range(N_data):
            params_data[m] = _mle_gbm_mu_sigma(X_data[m], dt)
        
        # ----- Estimation (mu, sigma) sur X_gen -----

        # garder seulement les trajectoires strictement positives
        mask_pos = np.all(X_gen > 0, axis=1)
        X_gen_pos = X_gen[mask_pos]
        
        print(f"[WARN] GBM MLE: removed {np.sum(~mask_pos)}/{len(mask_pos)} gen paths with nonpositive values.")
        
        params_gen = np.zeros((X_gen_pos.shape[0], 2))
        for m in range(X_gen_pos.shape[0]):
            params_gen[m] = _mle_gbm_mu_sigma(X_gen_pos[m], dt)

        # ----- Filtrage des outliers (optionnel) -----
        if filter_outliers:
            lb = np.percentile(params_data, 3, axis=0)
            ub = np.percentile(params_data, 97, axis=0)
            mask = ((params_data >= lb) & (params_data <= ub)).all(axis=1)
            params_data = params_data[mask]

            lb = np.percentile(params_gen, 3, axis=0)
            ub = np.percentile(params_gen, 97, axis=0)
            mask = ((params_gen >= lb) & (params_gen <= ub)).all(axis=1)
            params_gen = params_gen[mask]

        # ----- Sauvegarde -----
        np.save(data_param_path, params_data)
        np.save(gen_param_path,  params_gen)

        print("[INFO] Saved GBM parameters to:")
        print(f"       {data_param_path}")
        print(f"       {gen_param_path}")

    # --------- Plot des distributions ----------
    plt.rcParams.update(_MLE_RCPARAMS)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    _kw = dict(title_fontsize=title_fontsize, label_fontsize=label_fontsize, tick_fontsize=tick_fontsize, m_train=N_data, m_gen=N_gen)
    _mle_kdeplot(axs[0], params_data[:, 0], params_gen[:, 0], "mu",    fix=fix, true_val=mu,    **_kw)
    _mle_kdeplot(axs[1], params_data[:, 1], params_gen[:, 1], "sigma", fix=fix, true_val=sigma, **_kw)

    # --- Titre global optionnel ---
    # suptitle suppressed — context is given by the output folder and config

    plt.tight_layout()

    # --------- Sauvegarde figure ----------
    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"GBM_{data_label}_{stem}.png"
        full_plot_path = os.path.join(plot_dir, filename)
        fig.savefig(full_plot_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved plot at: {full_plot_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_params:
        return params_data, params_gen



def _mle_bm_sigma(x, dt):
    """
    MLE de sigma pour un mouvement brownien :
        X_{k+1} - X_k ~ N(0, sigma^2 dt)
    retourne sigma_hat (scalaire).
    """
    x = np.asarray(x)
    dx = np.diff(x)          # increments
    n = len(dx)

    # variance MLE : (1/n) * sum dx^2
    v = np.mean(dx**2)
    sigma2_hat = v / dt
    sigma_hat = np.sqrt(sigma2_hat)
    return sigma_hat


def plot_params_distrib_BM(
    X_data,
    X_gen,
    dt,
    fix=False,
    sigma=None,
    data_label="Data",
    gen_label="Gen",
    filter_outliers=True,
    save_img=True,
    plot_dir="plots",
    plot_path="BM_params_distrib.png",
    param_dir="params",
    show=True,
    force=False,
    return_params=False,
    suptitle=None,   # <<< NOUVEL ARGUMENT
):
    """
    Estime sigma pour des trajectoires de BM dans X_data et X_gen via MLE
    et trace la distribution (KDE + histogrammes lissés).

    X_data, X_gen : array-like
        - shape (M, N)   ou
        - shape (1, M, N)

    dt : pas de temps utilisé dans simulate_BM.
    """

    # === Style cohérent avec les autres fonctions ===
    title_fontsize = 18
    label_fontsize = 14
    tick_fontsize  = 14
    suptitle_size  = 20

    plt.rcParams.update({'font.size': label_fontsize})

    # --------- Mise en forme des données ----------
    X_data = np.asarray(X_data)
    X_gen  = np.asarray(X_gen)

    # enlever éventuelle dimension (1, M, N)
    if X_data.ndim == 3 and X_data.shape[0] == 1:
        X_data = X_data[0]
    if X_gen.ndim == 3 and X_gen.shape[0] == 1:
        X_gen = X_gen[0]

    if X_data.ndim == 1:
        X_data = X_data[None, :]
    if X_gen.ndim == 1:
        X_gen = X_gen[None, :]

    N_data, T_data = X_data.shape
    N_gen,  T_gen  = X_gen.shape
    assert T_data == T_gen, "X_data et X_gen doivent avoir la même longueur temporelle."

    # --------- Gestion du cache ----------
    base = os.path.basename(plot_path)
    stem, _ = os.path.splitext(base)

    os.makedirs(param_dir, exist_ok=True)
    data_param_path = os.path.join(param_dir, f"{data_label}_{stem}.npy")
    gen_param_path  = os.path.join(param_dir,  f"{gen_label}_{stem}.npy")

    # --------- Chargement éventuel ----------
    if os.path.exists(data_param_path) and os.path.exists(gen_param_path) and not force:
        print("[INFO] Loading cached BM parameters from:")
        print(f"       {data_param_path}")
        print(f"       {gen_param_path}")
        sigma_data = np.load(data_param_path)
        sigma_gen  = np.load(gen_param_path)

    else:
        print("[INFO] No cached BM parameters found. Estimating sigma...")

        # ----- Estimation sur X_data -----
        sigma_data = np.zeros(N_data)
        for m in range(N_data):
            sigma_data[m] = _mle_bm_sigma(X_data[m], dt)

        # ----- Estimation sur X_gen -----
        sigma_gen = np.zeros(N_gen)
        for m in range(N_gen):
            sigma_gen[m] = _mle_bm_sigma(X_gen[m], dt)

        # ----- Filtrage des outliers -----
        if filter_outliers:
            lb = np.percentile(sigma_data, 3)
            ub = np.percentile(sigma_data, 97)
            mask_data = (sigma_data >= lb) & (sigma_data <= ub)
            sigma_data = sigma_data[mask_data]

            lb = np.percentile(sigma_gen, 3)
            ub = np.percentile(sigma_gen, 97)
            mask_gen = (sigma_gen >= lb) & (sigma_gen <= ub)
            sigma_gen = sigma_gen[mask_gen]

        # ----- Sauvegarde -----
        np.save(data_param_path, sigma_data)
        np.save(gen_param_path,  sigma_gen)

        print("[INFO] Saved BM parameters to:")
        print(f"       {data_param_path}")
        print(f"       {gen_param_path}")

    # --------- Plot ----------
    plt.rcParams.update(_MLE_RCPARAMS)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    _kw = dict(title_fontsize=title_fontsize, label_fontsize=label_fontsize, tick_fontsize=tick_fontsize, m_train=N_data, m_gen=N_gen)
    _mle_kdeplot(ax, sigma_data, sigma_gen, "sigma", fix=fix, true_val=sigma, **_kw)

    # --- Titre global optionnel ---
    # suptitle suppressed — context is given by the output folder and config

    plt.tight_layout()

    # --------- Sauvegarde figure ----------
    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"BM_{data_label}_{stem}.png"
        full_plot_path = os.path.join(plot_dir, filename)
        fig.savefig(full_plot_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved plot at: {full_plot_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_params:
        # on renvoie juste les tableaux de sigma estimés
        return sigma_data, sigma_gen


def plot_sine_params_distrib_multi_1D(
    X_data, X_gen_dict,
    fix=False, A_true=None, f_true=None, phi_true=None,
    data_label="Data",
    filter_outliers=True,
    save_img=True,
    plot_dir="plots",
    plot_path="sine_params_multi.png",
    param_dir="params",
    show=True,
    force=False,
    suptitle=None,
    compute_w1=True,
    save_w1=True,
    w1_dir="metrics",
):
    """
    Version MULTI identique au pattern OU_multi :
      - entrée: X_data + dict {label: X_gen}
      - estimation + cache des params (A,f,phi) pour data et chaque gen
      - KDE superposées sur 3 subplots

    Requiert: _fit_sine_1d, wasserstein_1D
    """
    # === Style cohérent avec plot_sine_params_distrib_1D / OU_multi ===
    title_fontsize = 2/3 * 18
    label_fontsize = 2/3 * 14
    tick_fontsize  = 2/3 * 14
    suptitle_size  = 2/3 * 20
    plt.rcParams.update({'font.size': label_fontsize})

    # ---------- format data ----------
    X_data = np.asarray(X_data)
    if X_data.ndim == 3 and X_data.shape[0] == 1:
        X_data = np.squeeze(X_data, axis=0)
    if X_data.ndim == 1:
        X_data = X_data[None, :]

    M_data, T = X_data.shape
    t = np.linspace(0, 1, T)

    # ---------- cache paths ----------
    base = os.path.basename(plot_path)
    stem, _ = os.path.splitext(base)

    os.makedirs(param_dir, exist_ok=True)

    data_param_path = os.path.join(param_dir, f"{data_label}_{stem}.npy")

    # ---------- estimate/load params_data ----------
    if os.path.exists(data_param_path) and not force:
        params_data = np.load(data_param_path)
    else:
        params_data = np.zeros((len(X_data), 3), dtype=float)
        for m in range(len(X_data)):
            params_data[m] = _fit_sine_1d(X_data[m], t)

        # sécurité
        params_data = params_data[np.isfinite(params_data).all(axis=1)]

        if filter_outliers and len(params_data) > 10:
            lb = np.percentile(params_data, 3, axis=0)
            ub = np.percentile(params_data, 97, axis=0)
            params_data = params_data[((params_data >= lb) & (params_data <= ub)).all(axis=1)]

        np.save(data_param_path, params_data)

    # ---------- estimate/load params_gen_map ----------
    params_gen_map = {}
    for gen_label, X_gen in X_gen_dict.items():
        X_gen = np.asarray(X_gen)
        if X_gen.ndim == 3 and X_gen.shape[0] == 1:
            X_gen = np.squeeze(X_gen, axis=0)
        if X_gen.ndim == 1:
            X_gen = X_gen[None, :]

        gen_param_path = os.path.join(param_dir, f"{gen_label}_{stem}.npy")

        if os.path.exists(gen_param_path) and not force:
            params_gen = np.load(gen_param_path)
        else:
            params_gen = np.zeros((len(X_gen), 3), dtype=float)
            for m in range(len(X_gen)):
                params_gen[m] = _fit_sine_1d(X_gen[m], t)

            # sécurité
            params_gen = params_gen[np.isfinite(params_gen).all(axis=1)]

            if filter_outliers and len(params_gen) > 10:
                lb = np.percentile(params_gen, 3, axis=0)
                ub = np.percentile(params_gen, 97, axis=0)
                params_gen = params_gen[((params_gen >= lb) & (params_gen <= ub)).all(axis=1)]

            np.save(gen_param_path, params_gen)

        params_gen_map[gen_label] = params_gen

    # ---------- compute W1 (comme OU_multi) ----------
    w1 = None
    if compute_w1:
        w1 = {}
        for gen_label, params_gen in params_gen_map.items():
            w1[gen_label] = {
                "A":   wasserstein_1D(params_data[:, 0], params_gen[:, 0]),
                "f":   wasserstein_1D(params_data[:, 1], params_gen[:, 1]),
                "phi": wasserstein_1D(params_data[:, 2], params_gen[:, 2]),
            }

        if save_w1:
            os.makedirs(w1_dir, exist_ok=True)
            w1_path = os.path.join(w1_dir, f"w1_multi_{data_label}_{stem}.json")
            with open(w1_path, "w", encoding="utf-8") as f:
                json.dump(w1, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Saved W1 metrics at: {w1_path}")

    # ---------- plot (même logique que OU_multi) ----------
    _gen_colors = ["#2CA25F", "#CB181D", "#8856A7", "#F16913"]
    _sine_params = [("A", A_true), ("f", f_true), ("phi", phi_true)]

    plt.rcParams.update(_MLE_RCPARAMS)
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))

    for col_i, (pname, true_val) in enumerate(_sine_params):
        lat = _LATEX_PARAMS.get(pname, pname)
        ax  = axs[col_i]

        sns.kdeplot(ax=ax, data=params_data[:, col_i], fill=True,
                    color=_COLOR_TRAIN, alpha=0.5,
                    label=f"Train (N={len(params_data)})")
        for k, (glabel, pg) in enumerate(params_gen_map.items()):
            c = _gen_colors[k % len(_gen_colors)]
            sns.kdeplot(ax=ax, data=pg[:, col_i], fill=True, color=c, alpha=0.5,
                        label=f"Gen (N={len(pg)})" if len(params_gen_map) == 1 else f"{glabel} (N={len(pg)})")

        if fix and true_val is not None:
            ax.axvline(true_val, color="black", linestyle="--")

        w1_lines = []
        for glabel, pg in params_gen_map.items():
            w1v     = wasserstein_1D(params_data[:, col_i], pg[:, col_i])
            std_ref = float(np.std(params_data[:, col_i]))
            w1n     = (w1v / std_ref) if std_ref > 0 else float("nan")
            prefix  = "" if len(params_gen_map) == 1 else f"{glabel}: "
            w1_lines.append(f"{prefix}$W_1$={w1v:.2f}  $W_1^{{\\rm norm}}$={w1n:.2f}")
        ax.text(0.04, 0.96, "\n".join(w1_lines), transform=ax.transAxes,
                fontsize=label_fontsize - 2, va="top", ha="left", color="#333333",
                bbox=_BOX_STYLE)

        ax.set_title(f"$\\hat{{{lat}}}$",        fontsize=title_fontsize)
        ax.set_xlabel(f"${lat}$",                 fontsize=label_fontsize)
        ax.set_ylabel("Empirical density (KDE)",  fontsize=label_fontsize)
        from matplotlib.ticker import AutoMinorLocator
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize,
                       direction="out", top=True, right=True)
        ax.tick_params(axis="both", which="minor",
                       direction="out", top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(
            fontsize=label_fontsize - 2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False,
        )

    # suptitle suppressed — context is given by the output folder and config

    plt.tight_layout()

    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
        full_plot_path = os.path.join(plot_dir, plot_path if plot_path.endswith(".png") else plot_path + ".png")
        fig.savefig(full_plot_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved plot at: {full_plot_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return w1



import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import gaussian_kde
import numpy as np

def _kde_1d(x, n_grid=200, clip=None, jitter=1e-8, max_tries=3):
    x = np.asarray(x).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.array([]), np.array([])

    # clip optionnel
    if clip is not None:
        lo, hi = clip
        x = x[(x >= lo) & (x <= hi)]
        if x.size < 2:
            return np.array([]), np.array([])

    xmin, xmax = float(np.min(x)), float(np.max(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return np.array([]), np.array([])

    # cas dégénéré: variance ~ 0 ou min==max
    if (xmax - xmin) <= 0.0 or np.std(x) == 0.0:
        # on fabrique une petite bosse gaussienne centrée sur la valeur unique
        m = float(np.median(x))
        s = max(jitter, 1e-6 * (abs(m) + 1.0))
        xs = np.linspace(m - 5*s, m + 5*s, n_grid)
        ys = np.exp(-0.5 * ((xs - m)/s)**2)
        ys /= np.trapz(ys, xs)
        return xs, ys

    # grille
    pad = 1e-6 * (abs(xmin) + abs(xmax) + 1.0)
    xs = np.linspace(xmin - pad, xmax + pad, n_grid)

    # KDE avec fallback jitter
    for t in range(max_tries):
        try:
            kde = gaussian_kde(x)
            ys = kde(xs)
            return xs, ys
        except Exception:
            # jitter proportionnel à l'échelle
            scale = max(np.std(x), 1e-6 * (abs(np.median(x)) + 1.0))
            x = x + np.random.normal(0.0, jitter * scale, size=x.shape)

    # dernier recours: hist lissé
    hist, edges = np.histogram(x, bins=max(10, n_grid//10), density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    # interpolation sur xs
    ys = np.interp(xs, centers, hist, left=0.0, right=0.0)
    return xs, ys



def _filter_outliers_percentile(params, lo=3, hi=97):
    params = np.asarray(params)
    lb = np.percentile(params, lo, axis=0)
    ub = np.percentile(params, hi, axis=0)
    mask = ((params >= lb) & (params <= ub)).all(axis=1)
    return params[mask]


def _format_params_html(params_dict):
    lines = if_lines = []
    for _, (lab, val) in (params_dict or {}).items():
        if isinstance(val, (int, float, np.integer, np.floating)):
            v_str = f"{float(val):.2f}"
        else:
            v_str = str(val)
        if_lines.append(f"{lab} = {v_str}")
    return "<br>".join(if_lines)


def _safe_label(s):
    # filenames: remove separators likely to break paths
    return str(s).replace("/", "_").replace("\\", "_")


# ---------------------------------------------------------------------
# Single OU distrib Plotly
# ---------------------------------------------------------------------
def plot_params_distrib_OU_plotly(
    X_data, X_gen, dt,
    fix=False,
    theta=None, mu=None, sigma=None,
    data_label="Data", gen_label="Gen",
    filter_outliers=True,
    save_img=True,
    plot_dir="plots",
    plot_path="params_distrib.html",
    param_dir="params",
    show=True,
    force=False,
    vectorized=True,
    returns=False,
    returns_type="simple",
    suptitle=None,
    params1=None,
    params2=None,
    params3=None,
    compute_w1=True,
    save_w1=True,
    w1_dir="metrics",
    n_grid=400,
):
    """
    Plotly interactive KDE plots for OU params (theta, mu, sigma).
    - Legend click to show/hide
    - Double-click to isolate
    - Scroll to zoom
    - Legend outside (no overlap)
    Saves HTML by default. PNG requires kaleido.
    Returns w1 dict (or None) like your matplotlib version.
    """
    params1 = params1 or {}
    params2 = params2 or {}
    params3 = params3 or {}

    # ---------- Paths for parameter cache ----------
    base = os.path.basename(plot_path)
    stem, _ = os.path.splitext(base)

    if returns:
        if returns_type == "log":
            data_label_eff = f"logr{data_label}"
            gen_label_eff  = f"logr{gen_label}"
        else:
            data_label_eff = f"r{data_label}"
            gen_label_eff  = f"r{gen_label}"
    else:
        data_label_eff = data_label
        gen_label_eff  = gen_label

    os.makedirs(param_dir, exist_ok=True)
    data_param_path = os.path.join(param_dir, f"{_safe_label(data_label_eff)}_{stem}.npy")
    gen_param_path  = os.path.join(param_dir,  f"{_safe_label(gen_label_eff)}_{stem}.npy")

    # ---------- Load or compute parameters ----------
    if os.path.exists(data_param_path) and os.path.exists(gen_param_path) and not force:
        params_data = np.load(data_param_path)
        params_gen  = np.load(gen_param_path)
    else:
        from scipy.optimize import minimize

        params_data = np.zeros((len(X_data), 3))
        for m in range(len(X_data)):
            params_init = [1, np.mean(X_data[m]), np.std(X_data[m])]
            fun = MLE_OU_robust_vect if vectorized else MLE_OU_robust
            res = minimize(
                fun, np.array(params_init), args=(X_data[m], dt),
                bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                method="L-BFGS-B",
            )
            params_data[m] = res.x

        params_gen = np.zeros((len(X_gen), 3))
        for m in range(len(X_gen)):
            params_init = [1, np.mean(X_gen[m]), np.std(X_gen[m])]
            fun = MLE_OU_robust_vect if vectorized else MLE_OU_robust
            res = minimize(
                fun, np.array(params_init), args=(X_gen[m], dt),
                bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                method="L-BFGS-B",
            )
            params_gen[m] = res.x

        if filter_outliers:
            params_data = _filter_outliers_percentile(params_data, 3, 97)
            params_gen  = _filter_outliers_percentile(params_gen, 3, 97)

        np.save(data_param_path, params_data)
        np.save(gen_param_path, params_gen)

    # ---------- Wasserstein ----------
    w1 = None
    if compute_w1:
        w1 = {
            "theta": wasserstein_1D(params_data[:, 0], params_gen[:, 0]),
            "mu":    wasserstein_1D(params_data[:, 1], params_gen[:, 1]),
            "sigma": wasserstein_1D(params_data[:, 2], params_gen[:, 2]),
        }
        if save_w1:
            os.makedirs(w1_dir, exist_ok=True)
            w1_path = os.path.join(w1_dir, f"w1_{_safe_label(data_label_eff)}_vs_{_safe_label(gen_label_eff)}_{stem}.json")
            with open(w1_path, "w", encoding="utf-8") as f:
                json.dump(w1, f, indent=2, ensure_ascii=False)

    # ---------- Build figure (3 panels) ----------
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(r"Distribution of θ", r"Distribution of μ", r"Distribution of σ"),
        horizontal_spacing=0.06
    )

    # shared x-range per param for consistent KDE grid
    clips = []
    for k in range(3):
        x_all = np.concatenate([params_data[:, k], params_gen[:, k]])
        x_all = x_all[np.isfinite(x_all)]
        clips.append((float(np.min(x_all)), float(np.max(x_all))) if x_all.size else None)

    # Add KDE traces with filled areas via "tozeroy"
    labels = [r"$\theta$", r"$\mu$", r"$\sigma$"]
    for col, (k, lab) in enumerate(zip([0, 1, 2], labels), start=1):
        xs_d, ys_d = _kde_1d(params_data[:, k], n_grid=n_grid, clip=clips[k])
        xs_g, ys_g = _kde_1d(params_gen[:, k],  n_grid=n_grid, clip=clips[k])

        if xs_d.size:
            fig.add_trace(
                go.Scatter(
                    x=xs_d, y=ys_d, mode="lines",
                    name=data_label_eff,
                    fill="tozeroy",
                    opacity=0.5,
                    showlegend=(col == 1),
                    legendgroup="data",
                ),
                row=1, col=col
            )
        if xs_g.size:
            fig.add_trace(
                go.Scatter(
                    x=xs_g, y=ys_g, mode="lines",
                    name=gen_label_eff,
                    fill="tozeroy",
                    opacity=0.5,
                    showlegend=(col == 1),
                    legendgroup="gen",
                ),
                row=1, col=col
            )

        # True value vertical line
        true_val = [theta, mu, sigma][k]
        if fix and (true_val is not None):
            fig.add_trace(
                go.Scatter(
                    x=[true_val, true_val],
                    y=[0, 1],  # will be rescaled by "yref"
                    mode="lines",
                    name="true" if col == 1 else "true ",
                    showlegend=(col == 1),
                    legendgroup="true",
                ),
                row=1, col=col
            )
            # Make that line span the plot height by updating after we know y-range:
            # We'll use shapes instead (cleaner) below.

        fig.update_xaxes(title_text=lab, row=1, col=col)
        fig.update_yaxes(title_text="density" if col == 1 else "", row=1, col=col)

    # Replace true-value traces by shapes (full height) for correctness
    shapes = []
    for col, val in enumerate([theta, mu, sigma], start=1):
        if fix and (val is not None):
            # xref depends on subplot
            xref = "x" if col == 1 else f"x{col}"
            yref = "paper"
            shapes.append(dict(
                type="line",
                xref=xref, yref=yref,
                x0=val, x1=val,
                y0=0.0, y1=1.0,
                line=dict(dash="dash", width=2),
            ))
    if shapes:
        fig.update_layout(shapes=shapes)

    # Annotations: params boxes + W1 per panel
    ann = []
    if params1:
        ann.append(dict(text=_format_params_html(params1), xref="x domain", yref="y domain",
                        x=0.98, y=0.98, xanchor="right", yanchor="top",
                        showarrow=False, align="right"))
    if params2:
        ann.append(dict(text=_format_params_html(params2), xref="x2 domain", yref="y2 domain",
                        x=0.98, y=0.98, xanchor="right", yanchor="top",
                        showarrow=False, align="right"))
    if params3:
        ann.append(dict(text=_format_params_html(params3), xref="x3 domain", yref="y3 domain",
                        x=0.98, y=0.98, xanchor="right", yanchor="top",
                        showarrow=False, align="right"))

    if w1 is not None:
        ann.append(dict(text=f"W1 = {w1['theta']:.4g}", xref="x domain",  yref="y domain",
                        x=0.02, y=0.98, xanchor="left", yanchor="top", showarrow=False))
        ann.append(dict(text=f"W1 = {w1['mu']:.4g}",    xref="x2 domain", yref="y2 domain",
                        x=0.02, y=0.98, xanchor="left", yanchor="top", showarrow=False))
        ann.append(dict(text=f"W1 = {w1['sigma']:.4g}", xref="x3 domain", yref="y3 domain",
                        x=0.02, y=0.98, xanchor="left", yanchor="top", showarrow=False))

    # Layout: legend outside, scroll zoom enabled via config
    title = ""  # suptitle suppressed — context given by output folder and config
    fig.update_layout(
        title=dict(text=title, x=0.5),
        legend=dict(
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        margin=dict(l=60, r=260, t=80, b=60),
        annotations=ann
    )

    config = {"scrollZoom": True, "displaylogo": False}

    # Save
    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
        base = os.path.basename(plot_path)
        stem2, ext = os.path.splitext(base)
        ext = ext.lower()

        if ext == "":
            ext = ".html"

        out_path = os.path.join(plot_dir, f"MLE_{_safe_label(data_label_eff)}_{stem2}{ext}")

        if ext == ".html":
            fig.write_html(out_path, include_plotlyjs="cdn")
        else:
            # requires kaleido for static export
            fig.write_image(out_path, scale=2)

        print(f"[INFO] Saved plot at: {out_path}")

    if show:
        fig.show(config=config)

    return w1


# ---------------------------------------------------------------------
# Multi OU distrib Plotly
# ---------------------------------------------------------------------
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_params_distrib_OU_multi_plotly(
    X_data, X_gen_dict, dt,
    fix=False, theta=None, mu=None, sigma=None,
    data_label="Data",
    filter_outliers=True,
    save_img=True,
    plot_dir="plots",
    plot_path="params_distrib_multi.html",
    param_dir="params",
    show=True,
    force=False,
    vectorized=True,
    suptitle=None,
    params1=None, params2=None, params3=None,
    compute_w1=True,
    save_w1=True,
    w1_dir="metrics",
    n_grid=400,
    include_w1_in_name=False,
    # NEW: layout / colors
    width=1200,
    height=500,
    legend_position="top",   # "top" or "right"
    opacity_fill=0.30,
    opacity_line=1.0,
):
    """
    Plotly multi comparison with consistent colors across subplots:
      - Same model -> same color for theta/mu/sigma traces.
      - Wider figure to use full width.
      - Legend placement configurable (top is best to avoid narrow subplots).

    Requires helpers:
      - _kde_1d
      - _filter_outliers_percentile
      - _safe_label
      - wasserstein_1D
      - MLE_OU_robust_vect / MLE_OU_robust
    """

    params1 = params1 or {}
    params2 = params2 or {}
    params3 = params3 or {}

    base = os.path.basename(plot_path)
    stem, _ = os.path.splitext(base)

    os.makedirs(param_dir, exist_ok=True)
    data_param_path = os.path.join(param_dir, f"{_safe_label(data_label)}_{stem}.npy")

    # ---------- Data params cache ----------
    if os.path.exists(data_param_path) and not force:
        params_data = np.load(data_param_path)
    else:
        from scipy.optimize import minimize

        params_data = np.zeros((len(X_data), 3))
        for m in range(len(X_data)):
            params_init = [1, np.mean(X_data[m]), np.std(X_data[m])]
            fun = MLE_OU_robust_vect if vectorized else MLE_OU_robust
            res = minimize(
                fun, np.array(params_init), args=(X_data[m], dt),
                bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                method="L-BFGS-B",
            )
            params_data[m] = res.x

        if filter_outliers:
            params_data = _filter_outliers_percentile(params_data, 3, 97)

        np.save(data_param_path, params_data)

    # ---------- Gen params cache ----------
    params_gen_map = {}
    from scipy.optimize import minimize

    for gen_label, X_gen in X_gen_dict.items():
        gen_param_path = os.path.join(param_dir, f"{_safe_label(gen_label)}_{stem}.npy")
        if os.path.exists(gen_param_path) and not force:
            params_gen = np.load(gen_param_path)
        else:
            params_gen = np.zeros((len(X_gen), 3))
            for m in range(len(X_gen)):
                params_init = [1, np.mean(X_gen[m]), np.std(X_gen[m])]
                fun = MLE_OU_robust_vect if vectorized else MLE_OU_robust
                res = minimize(
                    fun, np.array(params_init), args=(X_gen[m], dt),
                    bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                    method="L-BFGS-B",
                )
                params_gen[m] = res.x

            if filter_outliers:
                params_gen = _filter_outliers_percentile(params_gen, 3, 97)

            np.save(gen_param_path, params_gen)

        params_gen_map[gen_label] = params_gen

    # ---------- W1 ----------
    w1 = None
    if compute_w1:
        w1 = {}
        for gen_label, params_gen in params_gen_map.items():
            w1[gen_label] = {
                "theta": wasserstein_1D(params_data[:, 0], params_gen[:, 0]),
                "mu":    wasserstein_1D(params_data[:, 1], params_gen[:, 1]),
                "sigma": wasserstein_1D(params_data[:, 2], params_gen[:, 2]),
            }

        if save_w1:
            os.makedirs(w1_dir, exist_ok=True)
            w1_path = os.path.join(w1_dir, f"w1_multi_{_safe_label(data_label)}_{stem}.json")
            with open(w1_path, "w", encoding="utf-8") as f:
                json.dump(w1, f, indent=2, ensure_ascii=False)

    # ---------- Color map: SAME label -> SAME color across panels ----------
    try:
        import plotly.express as px
        palette = px.colors.qualitative.Plotly
    except Exception:
        # fallback simple palette
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    def hex_to_rgba(hex_color, a):
        h = hex_color.lstrip("#")
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return f"rgba({r},{g},{b},{a})"

    labels_all = [data_label] + list(params_gen_map.keys())
    color_map = {}
    for i, lab in enumerate(labels_all):
        color_map[lab] = palette[i % len(palette)]

    # ---------- Figure (wider) ----------
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Distribution of θ", "Distribution of μ", "Distribution of σ"),
        horizontal_spacing=0.04
    )

    # shared x-range per param based on all sets (data + all gens)
    clips = []
    for k in range(3):
        arrays = [params_data[:, k]] + [pg[:, k] for pg in params_gen_map.values()]
        x_all = np.concatenate(arrays) if arrays else params_data[:, k]
        x_all = x_all[np.isfinite(x_all)]
        clips.append((float(np.min(x_all)), float(np.max(x_all))) if x_all.size else None)

    # ---------- Add DATA traces first ----------
    axis_labels = [r"$\theta$", r"$\mu$", r"$\sigma$"]
    data_line = color_map[data_label]
    data_fill = hex_to_rgba(data_line, opacity_fill)

    for col, k in enumerate([0, 1, 2], start=1):
        xs_d, ys_d = _kde_1d(params_data[:, k], n_grid=n_grid, clip=clips[k])
        if xs_d.size:
            fig.add_trace(
                go.Scatter(
                    x=xs_d, y=ys_d, mode="lines",
                    name=data_label,
                    showlegend=(col == 1),
                    legendgroup=data_label,
                    line=dict(color=data_line, width=3),
                    fill="tozeroy",
                    fillcolor=data_fill,
                    opacity=opacity_line,
                ),
                row=1, col=col
            )
        fig.update_xaxes(title_text=axis_labels[col-1], row=1, col=col)
        fig.update_yaxes(title_text="density" if col == 1 else "", row=1, col=col)

    # ---------- Add GEN traces (same color across θ/μ/σ for each model) ----------
    for gen_label, params_gen in params_gen_map.items():
        line_color = color_map[gen_label]
        fill_color = hex_to_rgba(line_color, opacity_fill)

        name = gen_label
        if include_w1_in_name and (w1 is not None) and (gen_label in w1):
            ww = w1[gen_label]
            name = f"{gen_label} (W1θ={ww['theta']:.3g}, W1μ={ww['mu']:.3g}, W1σ={ww['sigma']:.3g})"

        if (w1 is not None) and (gen_label in w1):
            ww = w1[gen_label]
            hover = f"{gen_label}<br>W1θ={ww['theta']:.4g}<br>W1μ={ww['mu']:.4g}<br>W1σ={ww['sigma']:.4g}"
        else:
            hover = gen_label

        for col, k in enumerate([0, 1, 2], start=1):
            xs_g, ys_g = _kde_1d(params_gen[:, k], n_grid=n_grid, clip=clips[k])
            if not xs_g.size:
                continue

            fig.add_trace(
                go.Scatter(
                    x=xs_g, y=ys_g, mode="lines",
                    name=name,
                    showlegend=(col == 1),
                    legendgroup=name,  # toggles all 3 panels at once
                    line=dict(color=line_color, width=3),
                    fill="tozeroy",
                    fillcolor=fill_color,
                    opacity=opacity_line,
                    hovertext=hover,
                    hoverinfo="text",
                ),
                row=1, col=col
            )

    # ---------- True lines (full height) ----------
    shapes = []
    for col, val in enumerate([theta, mu, sigma], start=1):
        if fix and (val is not None):
            xref = "x" if col == 1 else f"x{col}"
            shapes.append(dict(
                type="line",
                xref=xref, yref="paper",
                x0=val, x1=val, y0=0.0, y1=1.0,
                line=dict(dash="dash", width=2, color="black"),
            ))
    if shapes:
        fig.update_layout(shapes=shapes)

    # ---------- Layout: make panels wide by avoiding a right-side legend ----------
    title = ""  # suptitle suppressed — context given by output folder and config
    if legend_position == "top":
        fig.update_layout(
            width=width,
            height=height,
            title=dict(text=title, x=0.5),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.08,
                xanchor="left", x=0.0,
            ),
            margin=dict(l=60, r=40, t=120, b=60),
        )
    else:
        # right legend eats width; increase width if you insist
        fig.update_layout(
            width=max(width, 2000),
            height=height,
            title=dict(text=title, x=0.5),
            legend=dict(
                yanchor="top", y=1.0,
                xanchor="left", x=1.02,
                bgcolor="rgba(255,255,255,0.7)",
            ),
            margin=dict(l=60, r=320, t=90, b=60),
        )

    config = {"scrollZoom": True, "displaylogo": False}

    # ---------- Save (HTML only, no kaleido) ----------
    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
        base = os.path.basename(plot_path)
        stem2, ext = os.path.splitext(base)
        ext = ext.lower()
        if ext != ".html":
            ext = ".html"

        out_path = os.path.join(plot_dir, f"MLE_MULTI_{_safe_label(data_label)}_{stem2}{ext}")
        fig.write_html(out_path, include_plotlyjs="cdn")
        print(f"[INFO] Saved interactive plot at: {out_path}")

    if show:
        fig.show(config=config)

    return w1


import os, json, hashlib
import numpy as np

def _hash_array_light(x: np.ndarray, n=1024) -> str:
    """Hash léger pour invalider le cache si contenu change (sampled)."""
    x = np.asarray(x)
    if x.size == 0:
        return "empty"
    xf = x.ravel()
    step = max(1, xf.size // n)
    xs = xf[::step][:n]
    xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
    h = hashlib.sha1(xs.tobytes()).hexdigest()
    return h[:12]


def _ensure_heston_shape(X: np.ndarray) -> np.ndarray:
    """Force (B, L, 2). Accepte (B,2,L) et filtre séries non finies."""
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Heston: X doit être 3D, reçu {X.shape}")
    # (B,2,L) -> (B,L,2)
    if X.shape[1] == 2 and X.shape[2] != 2:
        X = np.transpose(X, (0, 2, 1))
    if X.shape[-1] != 2:
        raise ValueError(f"Heston: dernière dim doit être 2 (S,v), reçu {X.shape}")
    # filtrage NaN/inf par trajectoire
    mask = np.isfinite(X).all(axis=(1, 2))
    return X[mask]


def _compute_heston_mle_params_cached(
    X: np.ndarray,
    dt: float,
    label: str,
    stem: str,
    param_dir: str,
    force: bool = False,
    vectorized: bool = True,
    filter_outliers: bool = True,
):
    """
    Calcule params MLE (kappa, theta, xi, rho, r) pour chaque trajectoire de X.
    Cache sur disque: param_dir/<safe_label>_<stem>__dt=...__h=....npy
    """
    from scipy.optimize import minimize

    X = _ensure_heston_shape(X)

    MLE_fn = MLE_Heston_robust_vect if vectorized else MLE_Heston_robust

    bounds = [
        (1e-6, None),      # kappa
        (1e-6, None),      # theta
        (1e-6, None),      # xi
        (-0.999, 0.999),   # rho
        (None, None),      # r
    ]

    os.makedirs(param_dir, exist_ok=True)
    h = _hash_array_light(X)
    fname = f"{_safe_label(label)}_{stem}__dt={dt:.12g}__h={h}.npy"
    path = os.path.join(param_dir, fname)

    if os.path.exists(path) and not force:
        params = np.load(path)
        return params

    params = np.zeros((len(X), 5), dtype=float)
    x0 = np.array([3.0, 0.5, 0.7, 0.0, 0.02], dtype=float)

    for m in range(len(X)):
        res = minimize(
            MLE_fn,
            x0=x0,
            args=(X[m], dt),
            bounds=bounds,
            method="L-BFGS-B",
        )
        params[m] = res.x

    if filter_outliers and len(params) > 10:
        params = _filter_outliers_percentile(params, 3, 97)

    np.save(path, params)
    return params


import os, json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_params_distrib_Heston_multi_plotly(
    X_data, X_gen_dict, dt,
    fix=False,
    kappa=None, theta=None, xi=None, rho=None, r=None,
    data_label="Data",
    filter_outliers=True,
    save_img=True,
    plot_dir="plots",
    plot_path="params_distrib_heston_multi.html",
    param_dir="params_heston",
    show=True,
    force=False,
    vectorized=True,
    suptitle=None,
    compute_w1=True,
    save_w1=True,
    w1_dir="metrics_heston",
    n_grid=400,
    include_w1_in_name=False,
    width=1700,
    height=650,
    legend_position="top",   # "top" or "right"
    opacity_fill=0.30,
    opacity_line=1.0,
):
    """
    IMPORTANT: X_data et tous les X_gen doivent être EN ESPACE ORIGINAL et en shape (B,L,2).
    """

    base = os.path.basename(plot_path)
    stem, _ = os.path.splitext(base)

    # --- compute cached params (réutilise logique non-multi) ---
    params_data = _compute_heston_mle_params_cached(
        X=X_data, dt=dt, label=data_label, stem=stem, param_dir=param_dir,
        force=force, vectorized=vectorized, filter_outliers=filter_outliers
    )

    params_gen_map = {}
    for gen_label, X_gen in X_gen_dict.items():
        params_gen_map[gen_label] = _compute_heston_mle_params_cached(
            X=X_gen, dt=dt, label=gen_label, stem=stem, param_dir=param_dir,
            force=force, vectorized=vectorized, filter_outliers=filter_outliers
        )

    # --- W1 ---
    w1 = None
    if compute_w1:
        w1 = {}
        for gen_label, pg in params_gen_map.items():
            w1[gen_label] = {
                "kappa": wasserstein_1D(params_data[:, 0], pg[:, 0]),
                "theta": wasserstein_1D(params_data[:, 1], pg[:, 1]),
                "xi":    wasserstein_1D(params_data[:, 2], pg[:, 2]),
                "rho":   wasserstein_1D(params_data[:, 3], pg[:, 3]),
                "r":     wasserstein_1D(params_data[:, 4], pg[:, 4]),
            }

        if save_w1:
            os.makedirs(w1_dir, exist_ok=True)
            w1_path = os.path.join(w1_dir, f"w1_heston_multi_{_safe_label(data_label)}_{stem}.json")
            with open(w1_path, "w", encoding="utf-8") as f:
                json.dump(w1, f, indent=2, ensure_ascii=False)

    # --- plotly colors ---
    try:
        import plotly.express as px
        palette = px.colors.qualitative.Plotly
    except Exception:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    def hex_to_rgba(hex_color, a):
        h = hex_color.lstrip("#")
        rr = int(h[0:2], 16); gg = int(h[2:4], 16); bb = int(h[4:6], 16)
        return f"rgba({rr},{gg},{bb},{a})"

    labels_all = [data_label] + list(params_gen_map.keys())
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(labels_all)}

    # --- figure layout ---
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=("Distribution of κ", "Distribution of θ", "Distribution of ξ",
                        "Distribution of ρ", "Distribution of r", ""),
        horizontal_spacing=0.05,
        vertical_spacing=0.12,
    )

    # ranges robustes (évite min/max extrêmes si bug) : percentiles 0.5-99.5
    clips = []
    for k in range(5):
        arrays = [params_data[:, k]] + [pg[:, k] for pg in params_gen_map.values()]
        x_all = np.concatenate(arrays)
        x_all = x_all[np.isfinite(x_all)]
        if x_all.size == 0:
            clips.append(None)
            continue
        lo = float(np.quantile(x_all, 0.005))
        hi = float(np.quantile(x_all, 0.995))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.min(x_all)), float(np.max(x_all))
            if lo == hi:
                hi = lo + 1e-6
        clips.append((lo, hi))

    pos = {0:(1,1), 1:(1,2), 2:(1,3), 3:(2,1), 4:(2,2)}
    axis_labels = [r"$\kappa$", r"$\theta$", r"$\xi$", r"$\rho$", r"$r$"]

    # --- DATA traces ---
    data_line = color_map[data_label]
    data_fill = hex_to_rgba(data_line, opacity_fill)

    for k in range(5):
        row, col = pos[k]
        xs_d, ys_d = _kde_1d(params_data[:, k], n_grid=n_grid, clip=clips[k])
        if xs_d.size:
            fig.add_trace(
                go.Scatter(
                    x=xs_d, y=ys_d, mode="lines",
                    name=data_label,
                    showlegend=(k == 0),
                    legendgroup=data_label,
                    line=dict(color=data_line, width=3),
                    fill="tozeroy",
                    fillcolor=data_fill,
                    opacity=opacity_line,
                ),
                row=row, col=col
            )
        fig.update_xaxes(title_text=axis_labels[k], row=row, col=col)
        fig.update_yaxes(title_text="density" if (row, col) == (1, 1) else "", row=row, col=col)

    # --- GEN traces ---
    for gen_label, pg in params_gen_map.items():
        line_color = color_map[gen_label]
        fill_color = hex_to_rgba(line_color, opacity_fill)

        name = gen_label
        hover = gen_label
        if (w1 is not None) and (gen_label in w1):
            ww = w1[gen_label]
            hover = (f"{gen_label}<br>"
                     f"W1κ={ww['kappa']:.4g}<br>"
                     f"W1θ={ww['theta']:.4g}<br>"
                     f"W1ξ={ww['xi']:.4g}<br>"
                     f"W1ρ={ww['rho']:.4g}<br>"
                     f"W1r={ww['r']:.4g}")
            if include_w1_in_name:
                name = (f"{gen_label} "
                        f"(W1κ={ww['kappa']:.3g}, W1θ={ww['theta']:.3g}, W1ξ={ww['xi']:.3g}, "
                        f"W1ρ={ww['rho']:.3g}, W1r={ww['r']:.3g})")

        for k in range(5):
            row, col = pos[k]
            xs_g, ys_g = _kde_1d(pg[:, k], n_grid=n_grid, clip=clips[k])
            if not xs_g.size:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs_g, y=ys_g, mode="lines",
                    name=name,
                    showlegend=(k == 0),
                    legendgroup=name,
                    line=dict(color=line_color, width=3),
                    fill="tozeroy",
                    fillcolor=fill_color,
                    opacity=opacity_line,
                    hovertext=hover,
                    hoverinfo="text",
                ),
                row=row, col=col
            )

    # --- True values lines (par subplot, pas yref="paper" global) ---
    true_vals = [kappa, theta, xi, rho, r]
    for k in range(5):
        val = true_vals[k]
        if fix and (val is not None):
            row, col = pos[k]
            fig.add_vline(x=val, line_dash="dash", line_width=2, line_color="black", row=row, col=col)

    # layout
    title = ""  # suptitle suppressed — context given by output folder and config
    if legend_position == "top":
        fig.update_layout(
            width=width, height=height,
            title=dict(text=title, x=0.5),
            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0.0),
            margin=dict(l=60, r=40, t=130, b=60),
        )
    else:
        fig.update_layout(
            width=max(width, 2200), height=height,
            title=dict(text=title, x=0.5),
            legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.02, bgcolor="rgba(255,255,255,0.7)"),
            margin=dict(l=60, r=360, t=100, b=60),
        )

    fig.update_xaxes(visible=False, row=2, col=3)
    fig.update_yaxes(visible=False, row=2, col=3)

    config = {"scrollZoom": True, "displaylogo": False}

    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
        stem2, ext = os.path.splitext(os.path.basename(plot_path))
        if ext.lower() != ".html":
            ext = ".html"
        out_path = os.path.join(plot_dir, f"MLE_MULTI_{_safe_label(data_label)}_{stem2}{ext}")
        fig.write_html(out_path, include_plotlyjs="cdn")
        print(f"[INFO] Saved interactive Heston multi plot at: {out_path}")

    if show:
        fig.show(config=config)

    return w1


def plot_Rt_chi2_test_heston(
    X_data, X_gen,
    dt,
    data_label="Simulated",
    gen_label="Generated",
    plot_dir="figures",
    plot_path="Rt_chi2.png",
    results_path="Rt_chi2_results.json",
    nbins=120,
    max_points=300_000,
    eps_v=1e-12,
    clip_R=(0.0, 30.0),
    show=True,
):
    # Lazy import for stats (scipy is usually available)
    try:
        from scipy import stats
        have_scipy = True
    except Exception:
        have_scipy = False

    os.makedirs(plot_dir, exist_ok=True)

    def _ensure_BLC2(X):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (B,L,2). Got {X.shape}")
        # Accept (B,2,L) -> (B,L,2)
        if X.shape[1] == 2 and X.shape[2] != 2:
            X = np.transpose(X, (0, 2, 1))
        if X.shape[-1] != 2:
            raise ValueError(f"Expected last dim=2. Got {X.shape}")
        return X

    def _compute_R(X):
        X = _ensure_BLC2(X)
        S = X[:, :, 0]
        v = X[:, :, 1]

        # ΔlogS_t on t=0..L-2
        # log requires S>0
        mask_S = np.isfinite(S) & (S > 0)
        mask_v = np.isfinite(v) & (v > eps_v)

        # Need both at t and t+1 for log increment, and v at t
        mask_pair = mask_S[:, :-1] & mask_S[:, 1:] & mask_v[:, :-1]
        if not np.any(mask_pair):
            return np.array([], dtype=float)

        logS = np.log(S, where=(S > 0), out=np.full_like(S, np.nan))
        dlogS = logS[:, 1:] - logS[:, :-1]

        num = dlogS**2
        den = v[:, :-1] * dt

        R = num / den
        R = R[mask_pair]

        # finite only
        R = R[np.isfinite(R)]
        if clip_R is not None:
            lo, hi = clip_R
            R = R[(R >= lo) & (R <= hi)]
        # subsample for speed/robustness
        if (max_points is not None) and (R.size > max_points):
            idx = np.random.choice(R.size, size=max_points, replace=False)
            R = R[idx]
        return R

    R_data = _compute_R(X_data)
    R_gen  = _compute_R(X_gen)

    out = {
        "dt": float(dt),
        "n_R_data": int(R_data.size),
        "n_R_gen": int(R_gen.size),
    }

    def _moments(R):
        if R.size == 0:
            return {"mean": None, "var": None}
        return {"mean": float(np.mean(R)), "var": float(np.var(R, ddof=1)) if R.size > 1 else 0.0}

    out["moments_data"] = _moments(R_data)
    out["moments_gen"]  = _moments(R_gen)

    # KS test vs Chi2(1)
    if have_scipy:
        # Use exact chi2 cdf
        if R_data.size > 0:
            ks_data = stats.kstest(R_data, 'chi2', args=(1,))
            out["ks_data"] = {"stat": float(ks_data.statistic), "pvalue": float(ks_data.pvalue)}
        else:
            out["ks_data"] = {"stat": None, "pvalue": None}

        if R_gen.size > 0:
            ks_gen = stats.kstest(R_gen, 'chi2', args=(1,))
            out["ks_gen"] = {"stat": float(ks_gen.statistic), "pvalue": float(ks_gen.pvalue)}
        else:
            out["ks_gen"] = {"stat": None, "pvalue": None}
    else:
        out["ks_data"] = {"stat": None, "pvalue": None, "note": "scipy not available"}
        out["ks_gen"]  = {"stat": None, "pvalue": None, "note": "scipy not available"}

    # Plot histogram + chi2 pdf overlay
    fig_path = os.path.join(plot_dir, plot_path)
    res_path = os.path.join(plot_dir, results_path)

    plt.figure(figsize=(9, 5))

    # binning on [lo,hi]
    if clip_R is not None:
        lo, hi = clip_R
    else:
        lo, hi = 0.0, max(
            np.max(R_data) if R_data.size else 1.0,
            np.max(R_gen) if R_gen.size else 1.0,
        )

    bins = np.linspace(lo, hi, nbins + 1)

    if R_data.size:
        plt.hist(R_data, bins=bins, density=True, alpha=0.45, label=f"{data_label} (R)")
    if R_gen.size:
        plt.hist(R_gen,  bins=bins, density=True, alpha=0.45, label=f"{gen_label} (R)")

    # chi2 pdf
    if have_scipy:
        xs = np.linspace(lo, hi, 400)
        pdf = stats.chi2.pdf(xs, df=1)
        plt.plot(xs, pdf, linewidth=2.0, label=r"$\chi^2(1)$ pdf")

    title = r"Diagnostic $R_t=(\Delta\log S_t)^2/(v_t\Delta t)$"
    if have_scipy:
        p_d = out["ks_data"]["pvalue"]
        p_g = out["ks_gen"]["pvalue"]
        title += "\n"
        title += f"KS p-values: data={p_d:.3g}" if p_d is not None else "KS p-values: data=None"
        title += f", gen={p_g:.3g}" if p_g is not None else ", gen=None"

    plt.title(title)
    plt.xlabel(r"$R_t$")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(fig_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

    # Save stats json
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[Rt-chi2] Saved figure: {fig_path}")
    print(f"[Rt-chi2] Saved stats : {res_path}")
    print("[Rt-chi2] Summary:", out)

    return out


# ===========================================================================
# Wasserstein W1 distance between MLE parameter distributions (SIM vs GEN)
# ===========================================================================

def compute_wasserstein_mle(
    paramsMLE_dir: str,
    param_names,
    data_label: str,
    gen_label: str,
    gen_plot_path: str,
) -> dict:
    """
    Compute the Wasserstein-1 distance between the empirical MLE parameter
    distributions of the simulation data and the generated data.

    Reads pre-computed .npy files from *paramsMLE_dir*:
        {data_label}_{stem}.npy  — shape (N, num_params)
        {gen_label}_{stem}.npy   — shape (M, num_params)

    where *stem* = basename of *gen_plot_path* without extension.

    Parameters
    ----------
    paramsMLE_dir : str
        Directory where the MLE .npy files are cached.
    param_names : sequence of str
        Ordered list of parameter names, e.g. ("theta", "mu", "sigma").
        Comes from METRICS_CONFIG[raw_label]["true_params"].
    data_label : str
        Label used when saving the simulation data .npy file.
    gen_label : str
        Label used when saving the generated data .npy file.
    gen_plot_path : str
        Path used as key in the MLE functions (stem determines the filename).

    Returns
    -------
    dict with two families of keys:
        "mle_w1/<param_name>"      — raw W1 distance (same units as the parameter)
        "mle_w1_norm/<param_name>" — W1 normalised by std(data), dimensionless
                                     (comparable across parameters)
    Empty dict if the .npy files are not found or param_names is empty.
    """
    if not param_names:
        return {}

    stem = os.path.splitext(os.path.basename(gen_plot_path))[0]
    data_path = os.path.join(paramsMLE_dir, f"{data_label}_{stem}.npy")
    gen_path  = os.path.join(paramsMLE_dir, f"{gen_label}_{stem}.npy")

    if not os.path.exists(data_path) or not os.path.exists(gen_path):
        print(
            f"[W1-MLE] .npy files not found — skipping.\n"
            f"  expected: {data_path}\n"
            f"            {gen_path}"
        )
        return {}

    params_data = np.load(data_path)
    params_gen  = np.load(gen_path)

    # Ensure 2-D (N, num_params)
    if params_data.ndim == 1:
        params_data = params_data[:, None]
    if params_gen.ndim == 1:
        params_gen = params_gen[:, None]

    result = {}
    for i, name in enumerate(param_names):
        if i < params_data.shape[1] and i < params_gen.shape[1]:
            w1 = wasserstein_1D(params_data[:, i], params_gen[:, i])
            result[f"mle_w1/{name}"] = w1

            # Normalise by std of the reference (SIM) distribution so that
            # values are dimensionless and comparable across parameters.
            std_ref = float(np.std(params_data[:, i]))
            w1_norm = (w1 / std_ref) if std_ref > 0 else float("nan")
            result[f"mle_w1_norm/{name}"] = w1_norm

            print(f"[W1-MLE] {name}: W1 = {w1:.6f}  W1_norm = {w1_norm:.4f}")

    # ── Aggregate scalar: mean W1_norm over non-degenerate parameters ──────────
    # A parameter is degenerate when its training distribution has zero variance
    # (e.g. a fixed parameter), making normalisation undefined (NaN).  Such
    # parameters are excluded from the mean so the aggregate remains meaningful.
    #
    # Interpretation:
    #   mle_w1_norm_mean = 0  → perfect overlap on all parameters
    #   mle_w1_norm_mean ≈ 0.1 → distributions are very similar (< 0.1 σ_train apart)
    #   mle_w1_norm_mean ≈ 1   → distributions separated by ~1 σ_train on average
    w1_norm_valid = [
        v for k, v in result.items()
        if k.startswith("mle_w1_norm/") and not np.isnan(v)
    ]
    n_total   = len(param_names)
    n_valid   = len(w1_norm_valid)
    if w1_norm_valid:
        mean_w1_norm = float(np.mean(w1_norm_valid))
        result["mle_w1_norm_mean"] = mean_w1_norm
        excl = f"  ({n_total - n_valid} degenerate param(s) excluded)" if n_valid < n_total else ""
        print(f"[W1-MLE] mean W1_norm = {mean_w1_norm:.4f}{excl}")
    else:
        result["mle_w1_norm_mean"] = float("nan")
        print("[W1-MLE] mean W1_norm = nan (all parameters degenerate)")

    return result






