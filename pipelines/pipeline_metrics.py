import wandb
from utils.imports_statiques import *
from pipelines.utils_pipelines import *
from metrics.MLE_params import *
from metrics.predictive_score import *
from metrics.discriminative_score import *
from metrics.nnd import compute_nnd, compute_fmem
from pipelines.pipeline_sampling import generate_samples_from_checkpoint
from utils.plots_timeseries import plot_fmem_over_epochs


def pipeline_metrics(
    hyperparams_data,
    hyperparams_schedule,
    hyperparams_training,
    hyperparams_sampling,
    hyperparams_metrics,
    data_out,
    schedule_out,
    training_out,
    sampling_out,
    dir_level,
    device,
    on_mle_done=None,
    on_extra_figs_done=None,
    run_dir=None,
):
    # Hyperparams metrics
    data_label = data_out["data_label"]
    raw_label = data_out["raw_label"]
    scorenet_dir = training_out["scorenet_dir"]
    method = hyperparams_sampling["method"]
    M_gen  = hyperparams_sampling.get("M_gen", None)
    if M_gen is None:
        M_gen = len(data_out.get("X_train", []))   # default: same as training set size
    params_dicts = data_out.get("params_dicts", None)
    fix_params = hyperparams_sampling.get("fix_params", None)
    dt = hyperparams_data.get("dt", None)

    X_gen = sampling_out["X_gen"]
    X_data = data_out.get("X_train")   # use full train set — subsampling to match X_gen is done inside _mle_kdeplot

    datagen_label = data_out.get("datagen_label", "Generated")
    gen_path = sampling_out.get("gen_path", "generated.npy")
    force_MLE = hyperparams_metrics.get("force_MLE", False)

    # Params dicts (for MLE plots)
    params, params1, params2, params3, params4, params5 = split_params_dicts(params_dicts, n=6)

    # Directories — flat runs/<run_id>/ when run_dir is provided,
    # otherwise fall back to legacy 5-level hierarchical paths.
    if run_dir is not None:
        _run_dir      = Path(run_dir)
        plotMLE_dir   = str(_run_dir / "figures" / "MLE")
        paramsMLE_dir = str(_run_dir / "MLE_params")
        mem_dir       = str(_run_dir / "figures" / "memorization")
        autocorr_dir  = str(_run_dir / "figures" / "autocorr")
    else:
        d = data_out["d"]
        experiments_base = build_experiments_base(
            data_out=data_out,
            schedule_out=schedule_out,
            hyperparams_schedule=hyperparams_schedule,
            hyperparams_training=hyperparams_training,
            scorenet_dir=scorenet_dir,
            d=d,
        )
        sampling_dir  = build_sampling_dir(hyperparams_sampling)
        plotMLE_dir   = f"{dir_level}/figures/{experiments_base}/{sampling_dir}/MLE"
        paramsMLE_dir = f"{dir_level}/experiments/{experiments_base}/{sampling_dir}/MLE"
        base_fig_dir  = os.path.dirname(plotMLE_dir)
        mem_dir       = os.path.join(base_fig_dir, "memorization")
        autocorr_dir  = os.path.join(base_fig_dir, "autocorr")
    gen_plot_path = gen_path

    # MLE
    cfg = get_metricsconfig(raw_label)
    suptitle = hyperparams_metrics.get("suptitle", None) or cfg["suptitle"]
    mle_fn = cfg["mle_fn"]

    mle_kwargs = dict(
        X_data=X_data,
        X_gen=X_gen,
        data_label=data_label,
        gen_label=datagen_label,
        filter_outliers=True,
        save_img=True,
        plot_dir=plotMLE_dir,
        plot_path=gen_plot_path,
        param_dir=paramsMLE_dir,
        show=True,
        force=force_MLE,
        suptitle=suptitle,
        fix=fix_params,
        dt=dt,
    )

    for i, p in enumerate((params1, params2, params3, params4, params5), start=1):
        if p is not None:
            mle_kwargs[f"params{i}"] = p

    params_cfg = DATA_CONFIG.get(raw_label, {}).get("params", [])
    has_dt = any(
        p == "dt" or (isinstance(p, tuple) and p[0] == "dt")
        for p in params_cfg
    )

    if (dt is None) or (not has_dt):
        mle_kwargs.pop("dt", None)

    out = mle_fn(**mle_kwargs) or {}

    # --- Wasserstein W1 on MLE parameter distributions ---
    true_params = cfg.get("true_params", ())
    if true_params:
        w1_mle = compute_wasserstein_mle(
            paramsMLE_dir=paramsMLE_dir,
            param_names=true_params,
            data_label=data_label,
            gen_label=datagen_label,
            gen_plot_path=gen_plot_path,
        )
        out.update(w1_mle)

    # Open MLE figures immediately, before slow predictive/discriminative computation
    if on_mle_done is not None:
        on_mle_done(plotMLE_dir)

    # Summary Table
    summary = {
        "raw_label": raw_label,
        "method": method,
        "M_gen": M_gen,
        "plotMLE_dir": plotMLE_dir,
        "paramsMLE_dir": paramsMLE_dir,
    }

    out.setdefault("plotMLE_dir", plotMLE_dir)
    out.setdefault("paramsMLE_dir", paramsMLE_dir)
    out.setdefault("data_label", data_label)
    out.setdefault("schedule_dir", schedule_out["schedule_dir"])
    out.setdefault("scorenet_dir", scorenet_dir)
    out.setdefault("method", method)

    # mem_dir and autocorr_dir are set above (flat or legacy, depending on run_dir)

    # ==========================================================
    # Convert to numpy for metrics that expect arrays
    # ==========================================================
    X_data_np = X_data.detach().cpu().numpy() if torch.is_tensor(X_data) else np.asarray(X_data)
    X_gen_np = X_gen.detach().cpu().numpy() if torch.is_tensor(X_gen) else np.asarray(X_gen)

    X_sim = data_out.get("X_train", None)
    X_sim_np = None
    if X_sim is not None:
        X_sim_np = X_sim.detach().cpu().numpy() if torch.is_tensor(X_sim) else np.asarray(X_sim)

    # Channel names
    if X_gen_np.ndim == 3:
        d = X_gen_np.shape[-1]
        channel_names = ["S", "v"] if d == 2 else [f"dim_{i}" for i in range(d)]
    else:
        d = 1
        channel_names = ["X"]

    extra_fig_paths = []

    # --- Memorization check ---
    if X_sim_np is not None:
        if X_gen_np.ndim == 2 and X_sim_np.ndim == 2:
            mem_path = plot_generated_vs_nearest_simulated(
                X_gen=X_gen_np,
                X_sim=X_sim_np,
                n_gen=5,
                k=3,
                save_dir=mem_dir,
                plot_path=gen_plot_path,
            )
            if mem_path:
                extra_fig_paths.append(mem_path)
            out["memorization_info"] = mem_path

        elif X_gen_np.ndim == 3 and X_sim_np.ndim == 3:
            mem_out = {}
            for j in range(d):
                mem_path = plot_generated_vs_nearest_simulated(
                    X_gen=X_sim_np[:, :, j],
                    X_sim=X_sim_np[:, :, j],
                    n_gen=5,
                    k=3,
                    title_suffix=channel_names[j],
                    save_dir=mem_dir,
                    plot_path=gen_plot_path,
                )
                if mem_path:
                    extra_fig_paths.append(mem_path)
                mem_out[channel_names[j]] = mem_path
            out["memorization_info"] = mem_out

    # --- Time autocorrelation (time-time correlation matrix) ---
    if X_sim_np is not None:
        if X_gen_np.ndim == 2 and X_sim_np.ndim == 2:
            corr_out = plot_time_corr_matrices(
                X_sim=X_sim_np, X_gen=X_gen_np,
                dt=dt,
                save_dir=autocorr_dir, plot_path=gen_plot_path,
            )
            out["corr_time_diff_fro"] = corr_out["corr_time_diff_fro"]
            summary["corr_time_diff_fro"] = corr_out["corr_time_diff_fro"]
            if corr_out.get("saved_path"):
                extra_fig_paths.append(corr_out["saved_path"])

        elif X_gen_np.ndim == 3 and X_sim_np.ndim == 3:
            corr_out_all = {}
            corr_fro_all = {}

            for j in range(d):
                corr_out_j = plot_time_corr_matrices(
                    X_sim=X_sim_np[:, :, j],
                    X_gen=X_gen_np[:, :, j],
                    title_suffix=channel_names[j],
                    dt=dt,
                    save_dir=autocorr_dir,
                    plot_path=gen_plot_path,
                )
                if corr_out_j.get("saved_path"):
                    extra_fig_paths.append(corr_out_j["saved_path"])
                corr_out_all[channel_names[j]] = corr_out_j
                corr_fro_all[channel_names[j]] = corr_out_j["corr_time_diff_fro"]

            out["corr_time_info"] = corr_out_all
            out["corr_time_diff_fro"] = corr_fro_all
            summary["corr_time_diff_fro"] = corr_fro_all

    # ==========================================================
    # NND — Nearest Neighbor Distance (memorization scalar metric)
    # nnd_ratio < 1 → memorization | ≈ 1 → generalization | > 1 → mode dropping
    # ==========================================================
    if X_sim_np is not None:
        try:
            nnd_out = compute_nnd(X_gen_np, X_sim_np)
            out["nnd_gen_train"]   = nnd_out["nnd_gen_train"]
            out["nnd_train_train"] = nnd_out["nnd_train_train"]
            out["nnd_ratio"]       = nnd_out["nnd_ratio"]
            summary["nnd_gen_train"]   = round(nnd_out["nnd_gen_train"],   4)
            summary["nnd_train_train"] = round(nnd_out["nnd_train_train"], 4)
            summary["nnd_ratio"]       = round(nnd_out["nnd_ratio"],       4)

            ratio = nnd_out["nnd_ratio"]
            interp = (
                "🔴 MÉMORISATION"   if ratio < 0.5  else
                "🟡 frontière"       if ratio < 0.85 else
                "🟢 généralisation"  if ratio < 1.5  else
                "⚪ mode dropping"
            )
            print(
                f"[METRICS] NND  gen→train={nnd_out['nnd_gen_train']:.4f}"
                f"  train→train={nnd_out['nnd_train_train']:.4f}"
                f"  ratio={ratio:.4f}  {interp}"
            )
        except Exception as e:
            print(f"[NND] Warning: could not compute NND — {e}")

    # ==========================================================
    # F_MEM — memorisation fraction over training epochs (optional)
    # Enabled via metrics.fmem_m > 0  AND  training.track_fmem=true
    # ==========================================================
    fmem_m        = hyperparams_metrics.get("fmem_m", 0)
    fmem_ckpt_dir = training_out.get("fmem_ckpt_dir", None)

    if fmem_m and fmem_ckpt_dir and os.path.isdir(fmem_ckpt_dir) and X_sim_np is not None:
        ckpt_files = sorted(
            [f for f in os.listdir(fmem_ckpt_dir)
             if f.startswith("epoch_") and f.endswith(".pth")],
            key=lambda f: int(f.split("_")[1].split(".")[0]),
        )

        if ckpt_files:
            import time as _time
            n_ckpts = len(ckpt_files)
            fmem_nb_steps = int(hyperparams_metrics.get("fmem_nb_steps", 800))
            # Use fewer diffusion steps for f_mem: only proximity to train set matters,
            # not sample quality — 10x speedup vs the default nb_steps=1000.
            hyperparams_sampling_fmem = {**hyperparams_sampling, "nb_steps": fmem_nb_steps}
            print(f"[FMEM] Computing f_mem on {n_ckpts} checkpoints "
                  f"(M={fmem_m} samples, {fmem_nb_steps} diffusion steps each)…")
            fmem_epochs = []
            fmem_values = []
            _t0 = _time.time()

            for i, ckpt_file in enumerate(ckpt_files, 1):
                epoch_num = int(ckpt_file.split("_")[1].split(".")[0])
                ckpt_path = os.path.join(fmem_ckpt_dir, ckpt_file)
                _step_t0 = _time.time()
                try:
                    X_gen_ckpt = generate_samples_from_checkpoint(
                        weights_path=ckpt_path,
                        hyperparams_data=hyperparams_data,
                        hyperparams_schedule=hyperparams_schedule,
                        hyperparams_training=hyperparams_training,
                        hyperparams_sampling=hyperparams_sampling_fmem,
                        data_out=data_out,
                        schedule_out=schedule_out,
                        training_out=training_out,
                        M=int(fmem_m),
                        device=device,
                    )
                    X_gen_ckpt_np = (
                        np.asarray(X_gen_ckpt)
                        if not isinstance(X_gen_ckpt, np.ndarray)
                        else X_gen_ckpt
                    )
                    fmem = compute_fmem(X_gen_ckpt_np, X_sim_np)
                    fmem_epochs.append(epoch_num)
                    fmem_values.append(fmem)
                    _step_s  = _time.time() - _step_t0
                    _elapsed = _time.time() - _t0
                    _eta_s   = (_elapsed / i) * (n_ckpts - i)
                    _eta_str = f"{int(_eta_s // 60)}m{int(_eta_s % 60):02d}s"
                    print(f"  [{i}/{n_ckpts}] epoch {epoch_num:>6d}: "
                          f"f_mem = {fmem:.4f}  "
                          f"({_step_s:.1f}s/ckpt, ETA {_eta_str})")
                except Exception as e:
                    print(f"  [{i}/{n_ckpts}] [FMEM] Warning — epoch {epoch_num}: {e}")
                finally:
                    os.remove(ckpt_path)

            if fmem_epochs:
                fmem_plot_dir  = (
                    str(Path(run_dir) / "figures" / "fmem")
                    if run_dir is not None
                    else os.path.join(os.path.dirname(plotMLE_dir), "fmem")
                )
                fmem_plot_path = plot_fmem_over_epochs(
                    fmem_values=fmem_values,
                    epochs=fmem_epochs,
                    save_dir=fmem_plot_dir,
                    plot_path=gen_plot_path,
                )
                if fmem_plot_path:
                    extra_fig_paths.append(fmem_plot_path)
                out["fmem_epochs"] = fmem_epochs
                out["fmem_values"] = fmem_values
                print(f"[FMEM] Done. Plot saved to {fmem_plot_path}")

                if wandb.run is not None:
                    for ep, fv in zip(fmem_epochs, fmem_values):
                        wandb.log({"losses/fmem": fv}, step=ep)

    # Open memorization + autocorr figures before slow predictive/discriminative
    out["extra_fig_paths"] = extra_fig_paths
    if on_extra_figs_done is not None and extra_fig_paths:
        on_extra_figs_done(extra_fig_paths)

    # ===============================
    # Predictive & Discriminative Scores
    # ===============================

    compute_predictive = hyperparams_metrics.get("compute_predictive", False)
    compute_discriminative = hyperparams_metrics.get("compute_discriminative", False)

    pred_iterations = hyperparams_metrics.get("pred_iterations", 1000)
    disc_iterations = hyperparams_metrics.get("disc_iterations", 1000)

    dev = torch.device(device) if isinstance(device, str) else device

    # --- Predictive score ---
    if compute_predictive:
        if X_data_np.ndim == 2 and X_gen_np.ndim == 2:
            pred_score = predictive_score(
                ori_data=X_data_np,
                generated_data=X_gen_np,
                col_pred=0,
                iterations=pred_iterations,
                device=dev,
            )
            out["predictive_score"] = pred_score
            summary["predictive_score"] = pred_score

        elif X_data_np.ndim == 3 and X_gen_np.ndim == 3:
            pred_scores = {}
            for j in range(d):
                pred_scores[channel_names[j]] = predictive_score(
                    ori_data=X_data_np[:, :, j:j+1],   # garde un format 3D (M,L,1) si besoin
                    generated_data=X_gen_np[:, :, j:j+1],
                    col_pred=0,
                    iterations=pred_iterations,
                    device=dev,
                )

            out["predictive_score"] = pred_scores
            summary["predictive_score"] = pred_scores

    # --- Discriminative score ---
    if compute_discriminative:
        if dev.type == "cuda":
            device_ids = hyperparams_metrics.get(
                "device_ids",
                [dev.index if dev.index is not None else 0]
            )
        else:
            device_ids = None

        if X_data_np.ndim == 2:
            X_data_t = torch.tensor(X_data_np, dtype=torch.float32).to(device)
            X_gen_t = torch.tensor(X_gen_np, dtype=torch.float32).to(device)

            disc_score = discriminative_score(
                ori_data=X_data_t,
                generated_data=X_gen_t,
                iterations=disc_iterations,
                device=dev,
                device_ids=device_ids,
            )

            out["discriminative_score"] = disc_score
            summary["discriminative_score"] = disc_score

        elif X_data_np.ndim == 3:
            # 1) score global multivarié
            X_data_t = torch.tensor(X_data_np, dtype=torch.float32).to(device)
            X_gen_t = torch.tensor(X_gen_np, dtype=torch.float32).to(device)

            disc_score_global = discriminative_score(
                ori_data=X_data_t,
                generated_data=X_gen_t,
                iterations=disc_iterations,
                device=dev,
                device_ids=device_ids,
            )

            # 2) score par canal
            disc_scores_channels = {}
            for j in range(d):
                X_data_t_j = torch.tensor(X_data_np[:, :, j:j+1], dtype=torch.float32).to(device)
                X_gen_t_j = torch.tensor(X_gen_np[:, :, j:j+1], dtype=torch.float32).to(device)

                disc_scores_channels[channel_names[j]] = discriminative_score(
                    ori_data=X_data_t_j,
                    generated_data=X_gen_t_j,
                    iterations=disc_iterations,
                    device=dev,
                    device_ids=device_ids,
                )

            out["discriminative_score"] = {
                "global": disc_score_global,
                "channels": disc_scores_channels,
            }
            summary["discriminative_score"] = out["discriminative_score"]

    plot_summary_table("METRICS", summary)

    if wandb.run is not None:
        wb_metrics = {}

        if "predictive_score" in out:
            ps = out["predictive_score"]
            if isinstance(ps, dict):
                for ch, v in ps.items():
                    wb_metrics[f"metrics/predictive_score/{ch}"] = v
            else:
                wb_metrics["metrics/predictive_score"] = ps

        if "discriminative_score" in out:
            ds = out["discriminative_score"]
            if isinstance(ds, dict) and "global" in ds:
                wb_metrics["metrics/discriminative_score/global"] = ds["global"]
                for ch, v in ds.get("channels", {}).items():
                    wb_metrics[f"metrics/discriminative_score/{ch}"] = v
            else:
                wb_metrics["metrics/discriminative_score"] = ds

        if "corr_time_diff_fro" in out:
            fro = out["corr_time_diff_fro"]
            if isinstance(fro, dict):
                for ch, v in fro.items():
                    wb_metrics[f"metrics/time_corr_diff/{ch}"] = v
            else:
                wb_metrics["metrics/time_corr_diff"] = fro

        # NND — memorization metrics
        # nnd_ratio < 1 = memorization, ≈ 1 = generalization, > 1 = mode dropping
        if "nnd_gen_train" in out:
            wb_metrics["metrics/nnd_gen_train"]   = out["nnd_gen_train"]
            wb_metrics["metrics/nnd_train_train"]  = out["nnd_train_train"]
            wb_metrics["metrics/nnd_ratio"]        = out["nnd_ratio"]

        # MLE Wasserstein W1: per-parameter (raw + normalised) + aggregate mean
        # All relevant keys start with "mle_w1" — catches:
        #   mle_w1/<param>, mle_w1_norm/<param>, mle_w1_norm_mean
        for key, val in out.items():
            if key.startswith("mle_w1"):
                wb_metrics[f"metrics/{key}"] = val

        wandb.log(wb_metrics)

        # ── Écriture locale pour Optuna --no-wandb-sync ───────────────────────
        # wandb-summary.json n'est pas disponible en mode offline avant sync.
        # On écrit les métriques dans un fichier JSON lu directement par jzsub.py.
        if wandb.run is not None:
            import json as _json, os as _os
            _metric_path = _os.path.join(wandb.run.dir, "optuna_metric.json")
            with open(_metric_path, "w") as _f:
                _json.dump(wb_metrics, _f)

    return out