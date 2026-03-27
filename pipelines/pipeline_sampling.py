import shutil
import tempfile

from utils.imports_statiques import *
from pipelines.utils_pipelines import *
from models.sampler import *
from models.loss import *
from utils.data_preprocessing import *


def generate_samples_from_checkpoint(
    weights_path,
    hyperparams_data,
    hyperparams_schedule,
    hyperparams_training,
    hyperparams_sampling,
    data_out,
    schedule_out,
    training_out,
    M,
    device,
):
    """
    Minimal generation from a checkpoint file.

    Loads the model weights, runs the backward diffusion, and returns
    M denormalised generated samples as a numpy array.

    Used internally for f_mem tracking across training checkpoints.

    Returns
    -------
    X_gen : np.ndarray, shape (M, L) for univariate or (M, L, C) for multivariate
    """
    model_label    = hyperparams_training["model_label"]
    schedule       = schedule_out["schedule"]
    schedule_label = hyperparams_schedule["schedule_label"]
    T              = hyperparams_schedule["T"]
    L              = hyperparams_data["L"]

    method     = hyperparams_sampling["method"]
    nb_steps   = hyperparams_sampling["nb_steps"]
    eps_sample = hyperparams_sampling["eps_sample"]

    mu_R_train  = training_out["mu_R_train"]
    std_R_train = training_out["std_R_train"]
    diffusion   = training_out["diffusion"]

    x0             = hyperparams_data.get("x0", None)
    transform_meta = data_out.get("transform_meta", {})

    # ── Instantiate model and load checkpoint ──────────────────────────────
    cfg         = get_modelconfig(model_label)
    score_model = instantiate_score_model(
        scorenet_label=model_label,
        Net=cfg["model"],
        schedule_label=schedule_label,
        schedule=schedule,
        hyperparams_schedule=hyperparams_schedule,
        hyperparams_training=hyperparams_training,
        L=L,
        device=device,
    )
    score_model = torch.nn.DataParallel(score_model).to(device)
    score_model.load_state_dict(
        torch.load(weights_path, map_location=device)["model"]
    )
    score_model.eval()

    # ── Sampler (temp dir to avoid polluting data_gen/) ───────────────────
    tmp_dir = tempfile.mkdtemp(prefix="fmem_ckpt_")
    try:
        sampler = Sampler(
            model=score_model,
            diffusion=diffusion,
            schedule=schedule,
            device=device,
            input_size=(hyperparams_training["in_channels"], L),
            save_dir=tmp_dir,
            T=T,
            sigma_data=1.0,
        )

        with torch.no_grad():
            Xnorm_gen_t, _ = sampler.sample(
                series_label="fmem_tmp",
                method=method,
                batch_size=M,
                num_steps=nb_steps,
                eps=eps_sample,
                show_progress=False,
                datagen_label="fmem",
                force=True,
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Denormalise ────────────────────────────────────────────────────────
    Xg_norm_mcl = Xnorm_gen_t.detach().cpu().numpy()                      # (B, C, L)
    Xg_mcl      = denormalize_zscore(Xg_norm_mcl, mu_R_train, std_R_train) # (B, C, L)

    meta = dict(transform_meta)
    C    = hyperparams_training["in_channels"]
    if "transforms" in meta and len(meta["transforms"]) != C:
        meta["transforms"] = [None] * C

    Xg_plot_mcl = inverse_transform(Xg_mcl, meta, x0=x0)   # (B, C, L)
    Xg_plot     = np.transpose(Xg_plot_mcl, (0, 2, 1))      # (B, L, C)

    if C == 1:
        return Xg_plot[..., 0]   # (B, L)
    return Xg_plot               # (B, L, C)


def pipeline_sampling(
    hyperparams_data,
    hyperparams_schedule,
    hyperparams_training,
    hyperparams_sampling,
    data_out,
    schedule_out,
    training_out,
    dir_level,
    device,
    run_dir=None,
):
    # Hyperparams sampling
    model_label = hyperparams_training["model_label"]
    schedule = schedule_out["schedule"]
    schedule_label = hyperparams_schedule["schedule_label"]
    T = hyperparams_schedule["T"]

    L = hyperparams_data["L"]
    data_label = data_out["data_label"]
    raw_label = data_out["raw_label"]
    datagen_label = data_out["datagen_label"]
    schedule_dir = schedule_out["schedule_dir"]
    series_path = data_out["series_path"]

    method = hyperparams_sampling["method"]
    M_gen  = hyperparams_sampling.get("M_gen", None)
    nb_steps = hyperparams_sampling["nb_steps"]
    eps_sample = hyperparams_sampling["eps_sample"]
    force_gen = hyperparams_sampling.get("force_gen", False)
    n_plots = hyperparams_sampling["n_plots"]

    weights_path = training_out["weights_path"]
    mu_R_train = training_out["mu_R_train"]
    std_R_train = training_out["std_R_train"]
    diffusion = training_out["diffusion"]

    transform = hyperparams_data.get("transforms", None)
    X_train = data_out["X_train"]
    if M_gen is None:
        M_gen = len(X_train)   # default: generate as many as training series
    params = data_out["params_dicts"][0] if data_out.get("params_dicts") else None

    # ── Derive channels + embed_dim (au cas où pipeline_training n'a pas tourné) ─
    hyperparams_training = derive_channels(hyperparams_training)

    # Model sampler
    cfg = get_modelconfig(model_label)
    model = cfg["model"]
    model_params = read_params(hyperparams_training, cfg["params"])

    # Directories — flat runs/<run_id>/ when run_dir is provided,
    # otherwise fall back to legacy 5-level hierarchical paths.
    scorenet_dir = add_labels_from_dict(model_label, model_params)
    if run_dir is not None:
        _run_dir    = Path(run_dir)
        datagen_dir = str(_run_dir / "data_gen")
        plotgen_dir = str(_run_dir / "figures" / "gen")
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
        sampling_dir = build_sampling_dir(hyperparams_sampling)
        datagen_dir = f"{dir_level}/data_gen/{experiments_base}/{sampling_dir}"
        plotgen_dir = f"{dir_level}/figures/{experiments_base}/{sampling_dir}/genseries"

    score_model = instantiate_score_model(
        scorenet_label=model_label,
        Net=model,
        schedule_label=schedule_label,
        schedule=schedule,
        hyperparams_schedule=hyperparams_schedule,
        hyperparams_training=hyperparams_training,
        L=L,
        device=device
    )

    score_model = torch.nn.DataParallel(score_model).to(device)
    score_model.load_state_dict(torch.load(weights_path, map_location=device)['model'])

    # Sampler
    sigma_data = float(torch.as_tensor(std_R_train, dtype=torch.float32).mean())
    
    sampler = Sampler(
        model=score_model,
        diffusion=diffusion,
        schedule=schedule,
        device=device,
        input_size=(hyperparams_training["in_channels"], L),
        save_dir=datagen_dir,
        T=T,
        sigma_data=1.0,
    )
    
    Xnorm_gen_t, gen_path = sampler.sample(
        series_label=series_path,
        method=method,
        batch_size=M_gen,
        num_steps=nb_steps,
        eps=eps_sample,
        show_progress=True,
        datagen_label=datagen_label,
        force=force_gen,
    )

    Xnorm_gen = Xnorm_gen_t.squeeze().detach().cpu().numpy()

    # Postprocess + plots
    dt = hyperparams_data.get("dt", None)
    x0 = hyperparams_data.get("x0", None)
    S0 = hyperparams_data.get("S0", None)
    v0 = hyperparams_data.get("v0", None)
    
    cfg = get_samplingconfig(raw_label)

    meta = dict(data_out["transform_meta"])  # copie locale

    # Corrige le nombre de transforms pour qu'il corresponde au nombre de canaux
    C_meta = hyperparams_training["in_channels"]
    if "transforms" in meta and len(meta["transforms"]) != C_meta:
        meta["transforms"] = [None] * C_meta

    # Xt
    Xt_blc = as_BLC(np.asarray(X_train[:M_gen]))   # (B,L,C)
    Xt_mcl = np.transpose(Xt_blc, (0, 2, 1))            # (B,C,L)
    Xt_plot_mcl = inverse_transform(Xt_mcl, meta, x0=x0)
    Xt_plot = np.transpose(Xt_plot_mcl, (0, 2, 1))      # (B,L,C)
    
    # --- Xg (généré, normalisé -> dénormalisé) ---
    Xg_norm_mcl = Xnorm_gen_t.detach().cpu().numpy()                      # (B,C,L)
    Xg_mcl = denormalize_zscore(Xg_norm_mcl, mu_R_train, std_R_train)      # (B,C,L)
    Xg_plot_mcl = inverse_transform(Xg_mcl, meta, x0=x0)                   # (B,C,L)
    Xg_plot = np.transpose(Xg_plot_mcl, (0, 2, 1))                         # (B,L,C)
    
    # raw pour retour (en BLC)
    Xg = np.transpose(Xg_norm_mcl, (0, 2, 1))   # (B,L,C) espace modèle
    Xt = Xt_blc                                 # (B,L,C)
    
    B, L_, C = Xt_plot.shape
    
    
    series_labels = list(cfg["series_labels"])
    plot_titles = list(cfg["plot_titles"])
    
    if len(series_labels) < C:
        series_labels += [f"ch{c}" for c in range(len(series_labels), C)]
    if len(plot_titles) < C:
        plot_titles += [f"Simulated vs. Generated (ch{c})" for c in range(len(plot_titles), C)]

    # Summary Table
    summary = {
    }
    
    # Plot
    for c in range(C):
        full_path = plot_random_time_series(
            X_data=Xt_plot[..., c],
            X_gen=Xg_plot[..., c],
            params1=params,
            data_label=datagen_label,
            N=L,
            dt=dt,   # always passed as-is
            plot_dir=plotgen_dir,
            plot_path=(f"{series_labels[c]}_{series_path}" if C > 1 else series_path),
            save_img=True,
            suptitle=(plot_titles[c] if C > 1 else plot_titles[0]),
            n_plot=n_plots,
            same_scale=False,
        )
        summary[f"Plot path {c}"] = full_path

        print("X_gen shape:", Xg_plot[..., c].shape)
        print("finite:", np.isfinite(Xg_plot[..., c]).all())
        print("nan count:", np.isnan(Xg_plot[..., c]).sum(), "inf count:", np.isinf(Xg_plot[..., c]).sum())
        print("min/max:", np.nanmin(Xg_plot[..., c]), np.nanmax(Xg_plot[..., c]))



    plot_summary_table("SAMPLER", summary)
    
    # Returned fields
    post = {
        "X_gen_raw": Xg,        # model space
        "X_train_raw": Xt,
        "X_gen": Xg_plot,       # physical space
        "X_train_MLE": Xt_plot,
    }
    
    if C == 1:
        post["X_gen"] = Xg_plot[..., 0]
        post["X_train_MLE"] = Xt_plot[..., 0]

    # -------------------------
    # 7) Return
    # -------------------------
    out = {
        "ScoreNet_sampler": score_model,
        "sampler": sampler,
        "Xnorm_gen": Xnorm_gen,
        "gen_path": gen_path,
        "datagen_dir": datagen_dir,
        "plotgen_dir": plotgen_dir,
        "scorenet_dir": scorenet_dir,
    }
    out.update(post)
    return out