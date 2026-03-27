from utils.imports_statiques import *
from utils.utils import *
from pipelines.utils_pipelines import *
from models.trainer import *
from models.loss import *
from utils.plots_training import *



def pipeline_training(
    hyperparams_schedule,
    hyperparams_training,
    data_out,
    schedule_out,
    dir_level,
    device,
    folder_run_id=None,
    run_dir=None,
):
    # Data
    X_train = data_out["X_train"]
    X_test = data_out["X_test"]
    L = int(X_train.shape[1])
    
    series_path = data_out["series_path"]
    data_label = data_out["data_label"]

    # Schedule
    schedule = schedule_out["schedule"]
    schedule_dir = schedule_out["schedule_dir"]

    with torch.no_grad():
        t_test = torch.tensor([0.1, 1.0, 3.0], device=device)
        print("sigma:", schedule.sigma(t_test))

    # Model    
    model_label = hyperparams_training["model_label"]
    force_train = hyperparams_training.get("force_train", True)
    drop_last = hyperparams_training.get("drop_last", True)

    # Training hyperparams
    batch_size = hyperparams_training["batch_size"]

    # ── Effective batch size: min(M_train, batch_size) ────────────────────────
    # Ensures batch_size never exceeds dataset size (paper: B = min(n, 512))
    M_train = len(X_train)
    batch_size_suggested = batch_size
    if batch_size > M_train:
        batch_size = M_train
        hyperparams_training = {**hyperparams_training, "batch_size": batch_size}

    lr = hyperparams_training["lr"]
    scheduler_cfg = hyperparams_training["scheduler"]
    n_epochs = hyperparams_training["n_epochs"]

    # ── Effective n_epochs: min(n_epochs, 1000 × M_train) ─────────────────────
    # Prevents wasting compute on tiny datasets: e.g. M_train=4 → max 4 000 epochs
    n_epochs_suggested = n_epochs
    n_epochs_eff = min(n_epochs, 1000 * M_train)
    if n_epochs_eff < n_epochs:
        n_epochs = n_epochs_eff
        hyperparams_training = {**hyperparams_training, "n_epochs": n_epochs}
    patience = hyperparams_training["patience"]
    loss_type = hyperparams_training["loss_type"]
    T = hyperparams_schedule["T"]

    # ── Derive channels + embed_dim ───────────────────────────────────────────
    hyperparams_training = derive_channels(hyperparams_training)

    # ── Summary: suggested (Optuna) vs effective (after caps/derivations) ─────
    def _fmt(suggested, effective):
        if suggested != effective:
            return f"{effective}  (Optuna: {suggested})"
        return str(effective)

    print(f"[TRAINING] ── Paramètres suggérés → effectifs ───────────────")
    print(f"[TRAINING]   M_train    : {M_train}")
    print(f"[TRAINING]   batch_size : {_fmt(batch_size_suggested, batch_size)}")
    print(f"[TRAINING]   n_epochs   : {_fmt(n_epochs_suggested, n_epochs)}")
    print(f"[TRAINING]   lr         : {lr}")
    print(f"[TRAINING]   channels   : {hyperparams_training['channels']}")
    print(f"[TRAINING]   embed_dim  : {hyperparams_training['embed_dim']}")
    print(f"[TRAINING] ──────────────────────────────────────────────────")

    # Model params
    cfg = get_modelconfig(model_label)
    model = cfg["model"]
    model_params = read_params(hyperparams_training, cfg["params"])
    scorenet_dir = add_labels_from_dict(model_label, model_params)
    scorenet_dir = add_label(scorenet_dir, "loss", loss_type)

    # Path building: flat runs/<run_id>/ structure when run_dir is provided,
    # otherwise fall back to legacy 5-level hierarchical paths.
    if run_dir is not None:
        _run_dir    = Path(run_dir)
        weights_dir = str(_run_dir / "weights")
        losses_dir  = str(_run_dir / "losses")
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
        weights_dir = f"{dir_level}/experiments/{experiments_base}/weights"
        losses_dir  = f"{dir_level}/experiments/{experiments_base}/losses"

    track_fmem    = hyperparams_training.get("track_fmem", False)
    fmem_n        = int(hyperparams_training.get("fmem_n", 10))
    fmem_ckpt_dir = os.path.join(weights_dir, "fmem_ckpts") if track_fmem else None

    # Si track_fmem est activé, on force le ré-entraînement :
    # sans entraînement, aucun checkpoint fmem n'est créé → le plot est impossible.
    if track_fmem:
        force_train = True

    # Dataloaders
    Xnorm_train_loader, mu_R_train, std_R_train = make_dataloader(
        X_train,
        batch_size=batch_size,
        shuffle=True,
        normalize=True,
        drop_last=drop_last,
    )

    Xnorm_test_loader, _, _ = make_dataloader(
        X_test,
        batch_size=batch_size,
        shuffle=False,
        normalize=True,
        drop_last=drop_last,
    )

    # Model + optimizer
    schedule_label = hyperparams_schedule["schedule_label"]
        
    score_model = instantiate_score_model(
        scorenet_label=model_label,
        Net=model,
        schedule_label=schedule_label,
        schedule=schedule,
        hyperparams_schedule=hyperparams_schedule,
        hyperparams_training=hyperparams_training,
        L=L,
        device=device,
    )

    score_model = torch.nn.DataParallel(score_model).to(device)
    optimizer = Adam(score_model.parameters(), lr=lr)

    scheduler = None

    if scheduler_cfg is not None and scheduler_cfg.get("enabled", False):
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get("mode", "min"),
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 10),
            threshold=scheduler_cfg.get("threshold", 1e-4),
            threshold_mode=scheduler_cfg.get("threshold_mode", "rel"),
            cooldown=scheduler_cfg.get("cooldown", 0),
            min_lr=scheduler_cfg.get("min_lr", 0.0),
        )

    trainable_params = sum(p.numel() for p in score_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in score_model.parameters())

    sigma_data = float(torch.as_tensor(std_R_train).mean())

    gauss_diff = GaussianDiffusion(
        schedule=schedule,
        loss_type=loss_type,
        T=T,
        sigma_data=sigma_data,
    )

    # Summary Table
    summary = {
        "model_label": model_label,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "loss_type": loss_type,
    }

    for k, v in model_params.items():
        summary[f"{k}"] = v

    plot_summary_table("TRAINER", summary)

    # Training
    train_losses, val_losses, weights_path, losses_path = train(
        model=score_model,
        train_loader=Xnorm_train_loader,
        test_loader=Xnorm_test_loader,
        loss_fn=gauss_diff.loss,
        optimizer=optimizer,
        device=device,
        series_label=series_path.split("_", 1)[1],
        weights_dir=weights_dir,
        losses_dir=losses_dir,
        n_epochs=n_epochs,
        patience=patience,
        force=force_train,
        data_label=data_label,
        scheduler=scheduler,
        fmem_ckpt_dir=fmem_ckpt_dir,
        fmem_n=fmem_n,
    )

    best_val = float(min(val_losses)) if val_losses is not None else float("inf")

    # Summary Table
    summary = {
        "scorenet_dir": scorenet_dir,
        "weights_path": weights_path,
        "losses_path": losses_path,
        "best_val_loss": best_val
    }
    plot_summary_table("TRAINER", summary)

    if run_dir is not None:
        loss_fig_dir = str(Path(run_dir) / "figures" / "losses")
    else:
        loss_fig_dir = f"{dir_level}/figures/{experiments_base}"
    loss_fig_path = plot_avg_loss_curves(
        train_losses, val_losses,
        save_dir=loss_fig_dir,
        plot_path=series_path,
    )

    return {
        "ScoreNet": score_model,
        "optimizer": optimizer,
        "Xnorm_train_loader": Xnorm_train_loader,
        "Xnorm_test_loader": Xnorm_test_loader,
        "mu_R_train": mu_R_train,
        "std_R_train": std_R_train,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "weights_path": weights_path,
        "losses_path": losses_path,
        "scorenet_dir": scorenet_dir,
        "weights_dir": weights_dir,
        "losses_dir": losses_dir,
        "diffusion": gauss_diff,
        "folder_run_id": folder_run_id,
        "loss_fig_path": loss_fig_path,
        "fmem_ckpt_dir": fmem_ckpt_dir,
    }