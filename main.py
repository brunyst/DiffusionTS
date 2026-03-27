import sys
import os
import glob as _glob
import subprocess
import warnings
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: saves figures without opening windows
warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")

import hydra
import wandb
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from utils.imports_statiques import *
from utils.imports_dynamiques import *


def _wb_image(path):
    """Return a wandb.Image at full resolution.

    Upload happens in a background thread so bandwidth is not a concern.
    Plots are saved at 300 dpi — we preserve every pixel for W&B panel quality.
    """
    return wandb.Image(str(path))


def _auto_run_name(cfg) -> str:
    """Nom du run W&B : hyperparams principaux pour identification rapide.
    Ex: OUrange_Mtrain=5000_L=256_d=1_archi=ScoreNet1DBaseline"""
    archi = cfg.training.model_label.replace(" ", "")
    return (
        f"{cfg.data.data_label}"
        f"_Mtrain={cfg.data.M_train}"
        f"_L={cfg.data.L}"
        f"_d={cfg.data.d}"
        f"_archi={archi}"
    )


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    device = get_device()
    print(device)

    set_seed(cfg.project.seed)

    print(OmegaConf.to_yaml(cfg))

    # Determine run_id: prefer the one pre-assigned by jzsub.py (via WANDB_RUN_ID env var
    # or cfg.training.run_id), so the local runs/ folder name matches W&B and Jean Zay.
    pre_run_id = (
        os.environ.get("WANDB_RUN_ID")
        or (cfg.training.get("run_id") if cfg.training.get("run_id") else None)
    )

    run_name = os.environ.get("WANDB_NAME") or cfg.wandb.get("name") or _auto_run_name(cfg)
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity") or None,
        name=run_name,
        id=pre_run_id,            # keeps W&B run ID consistent with local runs/ folder
        tags=list(cfg.wandb.get("tags", [])) or None,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.get("mode", "online"),
        dir=HydraConfig.get().runtime.output_dir,
        save_code=False,          # skip code upload — code is already on GitHub
    )

    # After wandb.init the actual run_id is confirmed (auto-generated if pre_run_id was None)
    wb_run_id = wandb.run.id if wandb.run is not None else None
    run_id = wb_run_id or pre_run_id or "local"

    # Create flat run directory. Save the resolved Hydra config only if no
    # pre-submission snapshot was already written by jzsub.py — preserving the
    # original formatting (flow-style lists, comments, key order) in that case.
    from pathlib import Path as _Path
    import yaml as _yaml
    run_dir = _Path(dir_level) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _config_path = run_dir / "config.yaml"
    if not _config_path.exists():
        with open(_config_path, "w") as _cfg_f:
            _yaml.dump(
                OmegaConf.to_container(cfg, resolve=True),
                _cfg_f,
                default_flow_style=False,
                allow_unicode=True,
            )

    # conversion Hydra -> dict
    hyperparams_data = OmegaConf.to_container(cfg.data, resolve=True)
    hyperparams_schedule = OmegaConf.to_container(cfg.schedule, resolve=True)
    hyperparams_training = OmegaConf.to_container(cfg.training, resolve=True)
    hyperparams_sampling = OmegaConf.to_container(cfg.sampling, resolve=True)
    hyperparams_metrics = OmegaConf.to_container(cfg.metrics, resolve=True)

    # Inject the confirmed run_id so all pipeline functions use the same flat run_dir
    hyperparams_training["run_id"] = run_id

    # --- Callbacks: open figures as soon as generated (non-blocking, macOS) ---

    def _open_sim_figures(data_out):
        if sys.platform == "darwin":
            for f in data_out.get("sim_fig_paths", []):
                subprocess.Popen(["open", f])

    def _open_training_figures(train_out):
        if sys.platform == "darwin":
            p = train_out.get("loss_fig_path")
            if p and os.path.isfile(p):
                subprocess.Popen(["open", p])

    def _open_sampling_figures(run_id, sampling_out):
        if sys.platform == "darwin":
            plotgen_dir = sampling_out.get("plotgen_dir", "")
            if plotgen_dir:
                for f in sorted(_glob.glob(f"{plotgen_dir}/**/*.png", recursive=True)):
                    subprocess.Popen(["open", f])

    # Open memorization + autocorr figures before predictive/discriminative (non-blocking, macOS)
    def _open_extra_figures(fig_paths):
        if sys.platform == "darwin":
            for f in fig_paths:
                if os.path.isfile(f):
                    subprocess.Popen(["open", f])

    # Open MLE figures right after MLE, before slow predictive/discriminative (non-blocking, macOS)
    def _open_mle_figures(plotMLE_dir):
        if sys.platform == "darwin":
            if plotMLE_dir:
                for f in sorted(_glob.glob(f"{plotMLE_dir}/**/*.png", recursive=True)):
                    subprocess.Popen(["open", f])

    # training
    training_out = pipeline_data_schedule_training(
        hyperparams_data=hyperparams_data,
        hyperparams_schedule=hyperparams_schedule,
        hyperparams_training=hyperparams_training,
        dir_level=dir_level,
        device=device,
        wb_run_id=wb_run_id,
        on_data_done=_open_sim_figures,
        on_training_done=_open_training_figures,
    )

    # sampling + metrics
    sampling_metrics_out = pipeline_sampling_metrics(
        training_out=training_out,
        hyperparams_sampling=hyperparams_sampling,
        hyperparams_metrics=hyperparams_metrics,
        dir_level=dir_level,
        device=device,
        mode=cfg.project.mode,
        on_sampling_done=_open_sampling_figures,
        on_mle_done=_open_mle_figures,
        on_extra_figs_done=_open_extra_figures,
    )


    # ── Log all figures to W&B ────────────────────────────────────────────────
    # Fixed key convention (consistent across all runs, regardless of hyperparams):
    #   figures/data/series          — simulated training data
    #   figures/generated/series     — generated series (univarié)
    #   figures/generated/series_0…  — generated series par canal (multivarié)
    #   figures/MLE/params           — densités MLE de tous les paramètres
    #   figures/memorization/0…      — nearest-neighbor memorization check
    #   figures/autocorr/0…          — matrices de corrélation temps-temps
    #   figures/fmem/curve           — f_mem vs epochs (si track_fmem=true)
    if wandb.run is not None:
        media = {}

        # ── pipeline output handles ───────────────────────────────────────────
        sm = sampling_metrics_out if isinstance(sampling_metrics_out, dict) else {}
        best = sm.get("best", sm) if sm.get("grid") else sm
        sampling_out_inner = best.get("sampling_out", {}) or {}
        metrics_out_inner  = best.get("metrics_out",  {}) or {}

        # ── Simulation figures (1 file from plot_random_series) ───────────────
        seen_sim = set()
        for r in (training_out if isinstance(training_out, list) else []):
            for p in r.get("data_out", {}).get("sim_fig_paths", []):
                if p not in seen_sim and os.path.isfile(p):
                    seen_sim.add(p)
                    media["figures/data/series"] = _wb_image(p)
                    break  # 1 seul plot suffit

        # ── Generated series (1 fichier par canal) ────────────────────────────
        # glob.escape() nécessaire : les chemins contiennent parfois des "[...]"
        plotgen_dir = sampling_out_inner.get("plotgen_dir", "")
        if plotgen_dir:
            gen_pngs = sorted(_glob.glob(
                f"{_glob.escape(plotgen_dir)}/**/*.png", recursive=True
            ))
            if len(gen_pngs) == 1:
                media["figures/generated/series"] = _wb_image(gen_pngs[0])
            else:
                for i, p in enumerate(gen_pngs):
                    media[f"figures/generated/series_{i}"] = _wb_image(p)

        # ── MLE — 1 seul PNG contenant toutes les densités ───────────────────
        plotMLE_dir = metrics_out_inner.get("plotMLE_dir", "")
        if plotMLE_dir:
            mle_pngs = sorted(set(
                _glob.glob(f"{_glob.escape(plotMLE_dir)}/*.png") +
                _glob.glob(f"{_glob.escape(plotMLE_dir)}/**/*.png", recursive=True)
            ))
            if mle_pngs:
                media["figures/MLE/params"] = _wb_image(mle_pngs[0])

        # ── Memorization / autocorr / fmem — extra_fig_paths ─────────────────
        # Chaque fichier est classé par son dossier parent (méthode robuste)
        counts: dict = {}
        for p in metrics_out_inner.get("extra_fig_paths", []):
            if not p or not os.path.isfile(p):
                continue
            fname = os.path.basename(p).lower()
            if "fmem" in fname:
                media["figures/fmem/curve"] = _wb_image(p)
            elif "memorization" in fname or "nearest" in fname:
                i = counts.get("memorization", 0)
                media[f"figures/memorization/{i}"] = _wb_image(p)
                counts["memorization"] = i + 1
            elif "autocorr" in fname or "corr" in fname:
                i = counts.get("autocorr", 0)
                media[f"figures/autocorr/{i}"] = _wb_image(p)
                counts["autocorr"] = i + 1

        if media:
            wandb.log(media)

    wandb.finish()


if __name__ == "__main__":
    main()