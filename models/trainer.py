import wandb
from utils.imports_statiques import *
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    device,
    series_label,
    data_label,
    weights_dir,
    losses_dir,
    n_epochs=500,
    patience=30,
    force=False,
    clip_grad=1.0,
    use_amp=False,
    scheduler=None,
    fmem_ckpt_dir=None,
    fmem_n=10,
):

    # Normalize device
    device = torch.device(device) if not isinstance(device, torch.device) else device

    # -----------------------
    # Paths
    # -----------------------
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(losses_dir, exist_ok=True)

    weights_path = os.path.join(weights_dir, f"{data_label}_weights_{series_label}.pth")
    losses_path = os.path.join(losses_dir, f"{data_label}_losses_{series_label}.json")

    # -----------------------
    # AMP selection
    # -----------------------
    if use_amp is None:
        use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # -----------------------
    # If checkpoint exists
    # -----------------------
    if os.path.exists(weights_path) and not force:
        print()
        print(f"[TRAINING] Checkpoint already exists: {weights_path}")
        print("[TRAINING] Loading checkpoint and skipping training.")

        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)

        train_losses, val_losses = [], []
        if os.path.exists(losses_path):
            try:
                with open(losses_path, "r") as f:
                    losses = json.load(f)
                train_losses = losses.get("train_losses", [])
                val_losses = losses.get("val_losses", [])
            except json.JSONDecodeError:
                print(f"[WARN] Loss file {losses_path} is empty or corrupted. Ignoring it.")
                train_losses, val_losses = [], []

        return train_losses, val_losses, weights_path, losses_path

    # -----------------------
    # Training loop
    # -----------------------
    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = -1
    counter_es = 0
    saved_once = False

    start_time = time.time()

    tqdm_epoch = trange(n_epochs, leave=True)
    for epoch in tqdm_epoch:
        # ----- TRAIN -----
        model.train()
        sum_train = 0.0
        n_train = 0

        for batch in tqdm(train_loader, leave=False):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)
            bs = int(x.shape[0])

            optimizer.zero_grad()
            loss = loss_fn(model, x)
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad))
            optimizer.step()

            # weighted average by batch size
            sum_train += float(loss.detach().item())
            n_train += 1

        avg_train_loss = sum_train / max(n_train, 1)
        train_losses.append(avg_train_loss)

        # ----- VALIDATION -----
        model.eval()
        sum_val = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in test_loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(device)
                bs = int(x.shape[0])

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    val_loss = loss_fn(model, x)

                sum_val += float(val_loss.detach().item())
                n_val += 1

        avg_val_loss = sum_val / max(n_val, 1)
        val_losses.append(avg_val_loss)

        if scheduler is not None:
            window = 5
            smoothed = sum(val_losses[-window:]) / window if len(val_losses) >= window else avg_val_loss
            scheduler.step(smoothed)

        current_lr = optimizer.param_groups[0]["lr"]


        print(
            f"Epoch {epoch+1}: "
            f"Train Loss = {avg_train_loss:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}, "
            f"LR = {current_lr:.3e}"
        )

        if wandb.run is not None:
            wandb.log({
                "losses/train": avg_train_loss,
                "losses/val":   avg_val_loss,
                "lr":           current_lr,
            }, step=epoch + 1)


        # ----- PERIODIC CHECKPOINT FOR F_MEM TRACKING -----
        if fmem_ckpt_dir is not None and (epoch + 1) % fmem_n == 0:
            os.makedirs(fmem_ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(fmem_ckpt_dir, f"epoch_{epoch + 1}.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch + 1}, ckpt_path)

        # ----- EARLY STOPPING & SAVE BEST -----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            counter_es = 0

            # checkpoint
            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "series_label": series_label,
                "data_label": data_label,
                "use_amp": bool(use_amp),
                "clip_grad": clip_grad,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,

            }

            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            torch.save(ckpt, weights_path)
            saved_once = True

            if wandb.run is not None:
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}",
                    type="model",
                    metadata={"epoch": epoch + 1, "best_val_loss": best_val_loss},
                )
                artifact.add_file(weights_path)
                wandb.run.log_artifact(artifact)
        else:
            counter_es += 1
            if patience is not None and counter_es >= patience:
                print(f"[TRAINING] Early stopping at epoch {epoch+1} (best (epoch, loss): ({best_epoch}, {best_val_loss}))")
                break

    # -----------------------
    # Save losses / metadata
    # -----------------------
    elapsed = time.time() - start_time

    meta = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "n_epochs_ran": len(train_losses),
        "patience": patience,
        "elapsed_sec": elapsed,
        "device": str(device),
        "use_amp": bool(use_amp),
        "clip_grad": clip_grad,
        "series_label": series_label,
        "data_label": data_label,
    }

    with open(losses_path, "w") as f:
        json.dump(meta, f, indent=2)

    print()
    if saved_once:
        print(f"[TRAINING] Saved best checkpoint at {weights_path}")
    print(f"[TRAINING] Saved losses/meta at {losses_path}")

    elapsed_min = elapsed / 60.0
    elapsed_h = elapsed / 3600.0
    
    print(
        f"[TRAINING] Training time: "
        f"{elapsed:.1f}s | {elapsed_min:.2f}min | {elapsed_h:.2f}h"
    )

    return train_losses, val_losses, weights_path, losses_path