from utils.imports_statiques import *
from models.schedules import *
from pipelines.utils_pipelines import *


def pipeline_schedule(hyperparams_schedule):

    # Schedule hyperparameters
    T = hyperparams_schedule["T"]
    eps = hyperparams_schedule["eps"]
    schedule_label = hyperparams_schedule["schedule_label"]

    # Schedule parameters (hyperparams + default values)
    cfg = get_scheduleconfig(schedule_label)
    schedule = cfg["schedule"]
    schedule_params = read_params(hyperparams_schedule, cfg["params"])
    #schedule_params["T"] = T
    #schedule_params["eps"] = eps
    schedule = schedule(**schedule_params)

    # Schedule directory
    schedule_dir = add_labels_from_dict(schedule_label, schedule_params)

    # Summary Table
    summary = {
        "schedule_label": schedule_label,
        "schedule_dir": schedule_dir,
    }

    for k, v in schedule_params.items():
        summary[f"{k}"] = v

    plot_summary_table("SCHEDULER", summary)


    schedule_dir_full = build_schedule_dir_full(
        {"schedule_dir": schedule_dir},
        hyperparams_schedule,
    )

    return {
        "schedule": schedule,
        "schedule_dir": schedule_dir,
        "schedule_dir_full": schedule_dir_full,
    }