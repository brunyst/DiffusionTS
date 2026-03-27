from utils.imports_statiques import *
from pipelines.utils_pipelines import *
from pipelines.pipeline_data import *
from pipelines.pipeline_schedule import *
from pipelines.pipeline_training import *
from pipelines.pipeline_sampling import *
from pipelines.pipeline_metrics import *
from pipelines.dicts import *


def pipeline_data_schedule_training(
    hyperparams_data: dict,
    hyperparams_schedule: dict,
    hyperparams_training: dict,
    dir_level: str,
    device,
    wb_run_id=None,
    on_data_done=None,
    on_training_done=None,
):
    data_grid = expand_grid(hyperparams_data)
    sched_grid = expand_grid(hyperparams_schedule)
    train_grid = expand_grid(hyperparams_training)

    n_combos = len(data_grid) * len(sched_grid) * len(train_grid)

    runs = []
    data_cache = {}
    schedule_cache = {}
    combo_idx = 0

    for hp_data in data_grid:
        data_key = short_hash({"hyperparams_data": hp_data})
        if data_key not in data_cache:
            data_out = pipeline_simulate_data(hp_data, dir_level, device)
            data_cache[data_key] = data_out
            if on_data_done is not None:
                on_data_done(data_out)
        else:
            data_out = data_cache[data_key]

        for hp_sched in sched_grid:
            sched_key = short_hash({"hyperparams_schedule": hp_sched})
            if sched_key not in schedule_cache:
                schedule_out = pipeline_schedule(hp_sched)
                schedule_cache[sched_key] = schedule_out
            else:
                schedule_out = schedule_cache[sched_key]

            for hp_train in train_grid:
                # For single runs (submitted via jzsub.py), use the pre-assigned run_id
                # so folder name matches W&B and local runs/ dir. For grid search,
                # fall back to a deterministic hash (run_id from hp_train is not unique
                # across combos since it comes from the base config).
                existing_run_id = hp_train.get("run_id")
                run_id = (
                    existing_run_id
                    if existing_run_id and n_combos == 1
                    else get_run_id(hp_data, hp_sched, hp_train, n=10)
                )
                hp_train_run = inject_run_id(hp_train, run_id)

                # Flat run directory: runs/<run_id>/
                run_dir = Path(dir_level) / "runs" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)

                # W&B-based folder ID (kept for backward compat, not used for paths)
                if wb_run_id is not None:
                    folder_run_id = f"{wb_run_id}_{combo_idx:02d}" if n_combos > 1 else wb_run_id
                else:
                    folder_run_id = run_id
                combo_idx += 1

                train_out = pipeline_training(
                    hyperparams_schedule=hp_sched,
                    hyperparams_training=hp_train_run,
                    data_out=data_out,
                    schedule_out=schedule_out,
                    dir_level=dir_level,
                    device=device,
                    folder_run_id=folder_run_id,
                    run_dir=str(run_dir),
                )

                if on_training_done is not None:
                    on_training_done(train_out)

                best_val = float(min(train_out["val_losses"])) if train_out.get("val_losses") else float("inf")

                runs.append(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "config": {"data": hp_data, "schedule": hp_sched, "training": hp_train},
                        "best_val": best_val,
                        "data_out": data_out,
                        "schedule_out": schedule_out,
                        "training_out": train_out,
                    }
                )

    runs.sort(key=lambda r: r["best_val"])
    return runs



def pipeline_sampling_metrics(
    training_out,
    hyperparams_sampling: dict,
    hyperparams_metrics: dict,
    dir_level: str,
    device,
    mode: str = "best",
    compare_architectures: bool = False,
    arch_list=None,
    transform_list=None,
    on_sampling_done=None,
    on_mle_done=None,
    on_extra_figs_done=None,
    on_metrics_done=None,
):
    runs = ensure_list(training_out, name="training_out")
    selected = runs if mode == "all" else [runs[0]]

    all_outputs = []
    for r in selected:
        run_id = r["run_id"]
        run_dir = r.get("run_dir")
        cfg = r["config"]
        hp_data = cfg["data"]
        hp_sched = cfg["schedule"]
        hp_train = cfg["training"]

        data_out = r["data_out"]
        schedule_out = r["schedule_out"]
        train_out = r["training_out"]

        hp_train_run = inject_run_id(hp_train, run_id)

        sampling_out = pipeline_sampling(
            hyperparams_data=hp_data,
            hyperparams_schedule=hp_sched,
            hyperparams_sampling=hyperparams_sampling,
            hyperparams_training=hp_train_run,
            data_out=data_out,
            schedule_out=schedule_out,
            training_out=train_out,
            dir_level=dir_level,
            device=device,
            run_dir=run_dir,
        )

        if on_sampling_done is not None:
            on_sampling_done(run_id, sampling_out)

        print("[METRICS] Computing evaluation metrics (this may take a few minutes)...")
        metrics_out = pipeline_metrics(
            hyperparams_data=hp_data,
            hyperparams_schedule=hp_sched,
            hyperparams_training=hp_train_run,
            hyperparams_sampling=hyperparams_sampling,
            hyperparams_metrics=hyperparams_metrics,
            data_out=data_out,
            schedule_out=schedule_out,
            training_out=train_out,
            sampling_out=sampling_out,
            dir_level=dir_level,
            device=device,
            on_mle_done=on_mle_done,
            on_extra_figs_done=on_extra_figs_done,
            run_dir=run_dir,
        )

        if on_metrics_done is not None:
            on_metrics_done(run_id, metrics_out)

        out = {
            "run_id": run_id,
            "best_val": r["best_val"],
            "sampling_out": sampling_out,
            "metrics_out": metrics_out,
        }
        if isinstance(metrics_out, dict) and ("stats" in metrics_out):
            out["stats"] = metrics_out["stats"]

        all_outputs.append(out)

    return all_outputs[0] if mode == "best" else {
        "grid": True,
        "n_runs": len(all_outputs),
        "results": all_outputs,
        "best": all_outputs[0],
    }