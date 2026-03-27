from utils.imports_statiques import *
import plotly.io as pio
from IPython.display import HTML, display


def rescale(x):
    x = np.asarray(x)
    minx = np.min(x)
    maxx = np.max(x)
    if maxx - minx < 1e-12:
        return x
    return (x - minx) / (maxx - minx)

def plot_random_time_series(
    X_data, X_gen=None, params1=None, params2=None, 
    data_label="", xlabel="Time", ylabel="Value", 
    N=None, dt=None, plot_dir="plots", plot_path="ts.png", 
    save_img=False, show=True, figsize=(14, 6), n_plot=5, 
    suptitle=None, same_scale=True
):
    # === Style cohérent avec plot_params_distrib_BM ===
    title_fontsize = 18
    label_fontsize = 14
    tick_fontsize  = 14
    suptitle_size  = 20

    plt.rcParams.update({'font.size': label_fontsize})

    # --- Initialisation params/labels ---
    params1 = params1 or {}
    params2 = params2 or {}
    
    if dt is None:
        t = np.arange(N)
        xlabel = "Number of points (L)"
    else:
        t = np.arange(N) * dt
        xlabel = r"Time units (L * $d_{\tau}$)"

    # --- Prepare figure ---
    if X_gen is not None:
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))
        axs = [axes[0], axes[1]]
        titles = [f"Simulated {data_label}", f"Generated {data_label}"]
        datasets = [X_data, X_gen]

        # === Même échelle que X_data (optionnel) ===
        if same_scale:
            y_min = np.min(X_data)
            y_max = np.max(X_data)
            x_min, x_max = t[0], t[-1]

    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axs = [ax]
        titles = [f"Simulated {data_label}"]
        datasets = [X_data]

    # --- Plot ---
    for i, (ax, X, title) in enumerate(zip(axs, datasets, titles)):
    
        n_series = len(X)

        if n_series <= n_plot:
            indices = np.arange(n_series)  # toutes les séries
        else:
            indices = np.random.choice(n_series, size=n_plot, replace=False)  # sans remise
        
        for j in indices:
            y = np.asarray(X[j]).reshape(-1)
            ax.plot(t, y, linewidth=1.5)

    
        # --- Style cohérent ---
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.grid(True, linestyle="--", alpha=0.5)
    
        # --- Même échelle (si demandé) ---
        if X_gen is not None and same_scale:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        # --- Affichage des paramètres ---
        if i == 0 and params1:
            lines = []
            for key, (label, value) in params1.items():
                v_str = f"{float(value):.3f}" if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
                lines.append(f"{label} = {v_str}")
            if lines:
                ax.text(0.98, 0.95, "\n".join(lines),
                        transform=ax.transAxes,
                        fontsize=label_fontsize,
                        va='top', ha='right', color='black')

        if i == 1 and params2:
            lines = []
            for key, (label, value) in params2.items():
                v_str = f"{float(value):.3f}" if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
                lines.append(f"{label} = {v_str}")
            if lines:
                ax.text(0.98, 0.95, "\n".join(lines),
                        transform=ax.transAxes,
                        fontsize=label_fontsize,
                        va='top', ha='right', color='black')
    
    # --- Titre global ---
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=suptitle_size)

    plt.tight_layout()  # laisse de la place pour le suptitle


    # --- Sauvegarde image ---
    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
        base = os.path.basename(plot_path)
        stem, _ = os.path.splitext(base)
        filename = f"{data_label}_{stem}.png"
        full_path = os.path.join(plot_dir, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        return full_path

    if show:
        plt.show()
    else:
        plt.close()

    return None


def plot_random_time_series_plotly(
    X_data, X_gen=None, params1=None, params2=None,
    data_label="", xlabel="Time", ylabel="Value",
    N=None, dt=None, plot_dir="plots", plot_path="ts.html",
    save_img=False, show=True, figsize=(14, 6), n_plot=5,
    suptitle=None, same_scale=True
):
    # --- Defaults ---
    params1 = params1 or {}
    params2 = params2 or {}

    if N is None:
        N = X_data.shape[1]

    if dt is None:
        t = np.arange(N)
        xlabel = "Number of points (L)"
    else:
        t = np.arange(N) * dt
        xlabel = r"Time units (L * d_tau)"

    # --- Subplots ---
    if X_gen is not None:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"Simulated {data_label}", f"Generated {data_label}")
        )
        datasets = [(X_data, 1, 1), (X_gen, 1, 2)]
    else:
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=(f"Simulated {data_label}",)
        )
        datasets = [(X_data, 1, 1)]

    # --- Same scale (optional, based on X_data only like your matplotlib code) ---
    y_min = y_max = None
    if X_gen is not None and same_scale:
        y_min = float(np.min(X_data))
        y_max = float(np.max(X_data))

    # --- Add traces ---
    rng = np.random.default_rng()
    for X, r, c in datasets:
        m = len(X)
        if m == 0:
            continue

        # pick indices (with replacement)
        idxs = rng.integers(0, m, size=n_plot)

        for k, j in enumerate(idxs):
            y = np.asarray(X[j]).reshape(-1)
            name = f"{'Sim' if c == 1 else 'Gen'} #{k+1}"

            # Use legend groups so each panel has its own legend group (cleaner toggling)
            legendgroup = "sim" if c == 1 else "gen"
            showlegend = True  # show each trace; user can toggle individually

            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=y,
                    mode="lines",
                    name=name,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                ),
                row=r, col=c
            )

    # --- Axis labels ---
    # x labels per subplot (Plotly shares layout, but we set per axis)
    if X_gen is not None:
        fig.update_xaxes(title_text=xlabel, row=1, col=1)
        fig.update_xaxes(title_text=xlabel, row=1, col=2)
        fig.update_yaxes(title_text=ylabel, row=1, col=1)
        fig.update_yaxes(title_text=ylabel, row=1, col=2)
    else:
        fig.update_xaxes(title_text=xlabel, row=1, col=1)
        fig.update_yaxes(title_text=ylabel, row=1, col=1)

    # --- Same scale limits ---
    if X_gen is not None and same_scale and y_min is not None and y_max is not None:
        fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
        fig.update_yaxes(range=[y_min, y_max], row=1, col=2)

    # --- Params as annotations (top-right inside each panel) ---
    def _params_text(pdict):
        lines = []
        for _, (lab, val) in pdict.items():
            if isinstance(val, (int, float, np.integer, np.floating)):
                v_str = f"{float(val):.3f}"
            else:
                v_str = str(val)
            lines.append(f"{lab} = {v_str}")
        return "<br>".join(lines)

    # position inside each subplot domain
    annotations = []
    if params1:
        annotations.append(dict(
            text=_params_text(params1),
            xref="x domain" if X_gen is None else "x domain",
            yref="y domain" if X_gen is None else "y domain",
            x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            align="right",
            showarrow=False
        ))
    if X_gen is not None and params2:
        # second subplot uses x2/y2 domains
        annotations.append(dict(
            text=_params_text(params2),
            xref="x2 domain",
            yref="y2 domain",
            x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            align="right",
            showarrow=False
        ))

    # --- Layout: legend outside, scroll zoom on, clean margins ---
    title = suptitle if suptitle is not None else ""
    width_px = int(figsize[0] * 90)
    height_px = int(figsize[1] * 90)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=width_px,
        height=height_px,
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,  # outside right
            bgcolor="rgba(255,255,255,0.7)",
            borderwidth=0
        ),
        margin=dict(l=60, r=260 if True else 60, t=80, b=60),
        annotations=annotations,
    )

    # This enables scroll-to-zoom in the rendered figure in notebooks/browsers
    # If you use fig.show(config=...), this is guaranteed.
    config = {
        "scrollZoom": True,
        "displaylogo": False,
    }

    # save
    if save_img:
        os.makedirs(plot_dir, exist_ok=True)
    
        base = os.path.basename(plot_path)
        stem2, ext = os.path.splitext(base)
        ext = ext.lower()
    
        # Sans kaleido: on force HTML
        if ext in [".png", ".pdf", ".svg", ".jpeg", ".jpg", ".webp"]:
            ext = ".html"
    
        out_path = os.path.join(plot_dir, f"{stem2}{ext}")
    
        if ext == ".html":
            fig.write_html(out_path, include_plotlyjs="cdn")
            print(f"[INFO] Saved interactive plot at: {out_path}")
        else:
            # au cas où, fallback HTML
            out_path = os.path.join(plot_dir, f"{stem2}.html")
            fig.write_html(out_path, include_plotlyjs=True)
            print(f"[INFO] Saved interactive plot at: {out_path}")

    # --- Show / close ---
    if show:
        html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,  # IMPORTANT : plotly.js déjà chargé par la cellule init
            config=config
        )
        display(HTML(html))

    return fig


def plot_generated_vs_nearest_simulated(X_gen, X_sim, n_gen=5, k=3, title_suffix=None, save_dir=None, plot_path="memorization"):
    X_gen = np.asarray(X_gen)
    X_sim = np.asarray(X_sim)

    assert X_gen.ndim == 2
    assert X_sim.ndim == 2

    M_gen = X_gen.shape[0]
    n_gen = min(n_gen, M_gen)

    # sélection aléatoire sans remise
    gen_idx = np.random.choice(M_gen, size=n_gen, replace=False)

    fig, axes = plt.subplots(
        nrows=n_gen,
        ncols=1,
        figsize=(14, 2.8 * n_gen),
        sharex=True
    )

    if n_gen == 1:
        axes = [axes]

    for ax, gi in zip(axes, gen_idx):
        xg = X_gen[gi]

        # distance L2 pointwise
        dists = np.sqrt(np.sum((X_sim - xg)**2, axis=1))
        nn_idx = np.argsort(dists)[:k]

        ax.plot(xg, linewidth=2, label=f"GEN[{gi}]")

        for rank, j in enumerate(nn_idx, start=1):
            ax.plot(
                X_sim[j],
                linewidth=1.2,
                label=f"SIM nn{rank}"
            )

        ax.legend(loc="upper right")
        ax.set_ylabel("value")

    if title_suffix is None:
        axes[0].set_title("Memorization check (L2)")
    else:
        axes[0].set_title(f"Memorization check (L2) — {title_suffix}")

    axes[-1].set_xlabel("time")

    plt.tight_layout()

    saved_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(plot_path))[0]
        suffix = f"_{title_suffix}" if title_suffix else ""
        saved_path = os.path.join(save_dir, f"{stem}_memorization{suffix}.png")
        plt.savefig(saved_path, dpi=300, bbox_inches='tight')

    plt.close(fig)
    return saved_path


def plot_fmem_over_epochs(fmem_values, epochs, save_dir=None, plot_path="fmem", title_suffix=None):
    """
    Plot f_mem (memorisation fraction) as a function of training epochs.

    Parameters
    ----------
    fmem_values : list of float — f_mem values (one per checkpoint)
    epochs      : list of int   — corresponding epoch numbers
    save_dir    : str or None   — directory to save the figure
    plot_path   : str           — base name for the output file
    title_suffix: str or None   — optional suffix for the figure title

    Returns
    -------
    saved_path : str or None
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(epochs, fmem_values, marker="o", markersize=3, linewidth=1.5, color="tab:red")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$f_{\mathrm{mem}}$")

    title = r"Memorisation fraction $f_{\mathrm{mem}}$ over training"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)

    ax.set_ylim(-0.02, max(max(fmem_values) * 1.1, 0.05))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    saved_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        stem   = os.path.splitext(os.path.basename(plot_path))[0]
        suffix = f"_{title_suffix}" if title_suffix else ""
        saved_path = os.path.join(save_dir, f"{stem}_fmem{suffix}.png")
        plt.savefig(saved_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return saved_path


# ---------------------------------------------------------------------------
# Publication-quality rcParams for correlation-matrix plots
# ---------------------------------------------------------------------------
_CORR_RCPARAMS = {
    "font.family":                  "STIXGeneral",
    "font.weight":                  "normal",
    "mathtext.fontset":             "cm",
    "font.size":                    16,
    "axes.labelsize":               18,
    "xtick.labelsize":              13,
    "ytick.labelsize":              13,
    "axes.titlesize":               18,
    "figure.titlesize":             20,
    "axes.formatter.use_mathtext":  True,
    "axes.linewidth":               0.8,
    "axes.facecolor":               "white",
    "figure.facecolor":             "white",
    "axes.grid":                    False,
    "figure.dpi":                   150,
    "savefig.dpi":                  300,
    "pdf.fonttype":                 42,
    "ps.fonttype":                  42,
    "savefig.bbox":                 "tight",
}


def _add_time_ticks(ax, L, dt, n_ticks=5):
    """Set imshow axis ticks to physical time (index × dt)."""
    tick_idx = np.linspace(0, L - 1, n_ticks, dtype=int)
    if dt is not None:
        tick_labels = [f"{int(i) * dt:.2g}" for i in tick_idx]
    else:
        tick_labels = [str(int(i)) for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels(tick_labels)


def _time_corr_matrix(X):

    X = np.asarray(X)
    assert X.ndim == 2, "attendu (M,L)"

    # centre par temps
    Z = X - X.mean(axis=0, keepdims=True)

    # std par temps (évite div0)
    std = Z.std(axis=0, ddof=1)
    std = np.where(std < 1e-12, 1.0, std)

    # covariance temporelle: (L,L) = (Z^T Z)/(M-1)
    M = Z.shape[0]
    Cov = (Z.T @ Z) / (M - 1)

    # corr = Cov / (std_t std_s)
    C = Cov / (std[:, None] * std[None, :])
    C = np.clip(C, -1.0, 1.0)
    return C


def plot_time_corr_matrices(X_sim, X_gen, title=None, title_suffix=None, dt=None, save_dir=None, plot_path="autocorr"):
    """
    3-panel publication-quality plot of the cross-time correlation matrices.

    Panels:
      0 — Train  :  C_train[t, s] = corr(X_t, X_s) over M train trajectories
      1 — Gen    :  same for generated trajectories
      2 — Diff   :  C_train − C_gen, with Frobenius-norm annotation

    Parameters
    ----------
    X_sim, X_gen : ndarray (M, L)
    title        : ignored (kept for backward compat) — no global suptitle
    title_suffix : optional channel label appended to panel titles
    dt           : time step (float); if given, axes show physical time t·dt
    save_dir, plot_path : where to save the figure
    """
    C_sim  = _time_corr_matrix(X_sim)
    C_gen  = _time_corr_matrix(X_gen)
    C_diff = C_sim - C_gen
    diff_fro = float(np.linalg.norm(C_diff, ord="fro"))
    L = C_sim.shape[0]

    with plt.rc_context(_CORR_RCPARAMS):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
        cmap = "RdBu_r"

        # --- Panel 0 : Train ---
        im0 = axes[0].imshow(C_sim, vmin=-1, vmax=1, cmap=cmap, origin="upper", aspect="equal")
        t0 = "Train" + (f" — {title_suffix}" if title_suffix else "")
        axes[0].set_title(t0)
        _add_time_ticks(axes[0], L, dt)
        axes[0].set_xlabel(r"$s$")
        axes[0].set_ylabel(r"$t$")
        cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cb0.set_label(r"$\hat{\rho}(t,s)$")

        # --- Panel 1 : Gen ---
        im1 = axes[1].imshow(C_gen, vmin=-1, vmax=1, cmap=cmap, origin="upper", aspect="equal")
        t1 = "Gen" + (f" — {title_suffix}" if title_suffix else "")
        axes[1].set_title(t1)
        _add_time_ticks(axes[1], L, dt)
        axes[1].set_xlabel(r"$s$")
        axes[1].set_ylabel(r"$t$")
        cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cb1.set_label(r"$\hat{\rho}(t,s)$")

        # --- Panel 2 : Difference (symmetric colorrange) ---
        vabs = max(float(np.abs(C_diff).max()), 1e-6)
        im2 = axes[2].imshow(C_diff, vmin=-vabs, vmax=vabs, cmap=cmap, origin="upper", aspect="equal")
        axes[2].set_title(r"$\hat{\rho}^{\rm train} - \hat{\rho}^{\rm gen}$")
        _add_time_ticks(axes[2], L, dt)
        axes[2].set_xlabel(r"$s$")
        axes[2].set_ylabel(r"$t$")
        cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        cb2.set_label(r"$\Delta\hat{\rho}(t,s)$")

        # Frobenius norm annotation (bottom-right, inside panel)
        axes[2].text(
            0.97, 0.03,
            fr"$\|\Delta\hat{{\rho}}\|_F = {diff_fro:.3f}$",
            transform=axes[2].transAxes,
            ha="right", va="bottom",
            fontsize=14,
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75),
        )

        saved_path = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            stem   = os.path.splitext(os.path.basename(plot_path))[0]
            suffix = f"_{title_suffix}" if title_suffix else ""
            saved_path = os.path.join(save_dir, f"{stem}_autocorr{suffix}.png")
            fig.savefig(saved_path, dpi=300, bbox_inches="tight")

        plt.close(fig)

    return {"corr_time_diff_fro": diff_fro, "C_sim": C_sim, "C_gen": C_gen, "saved_path": saved_path}

