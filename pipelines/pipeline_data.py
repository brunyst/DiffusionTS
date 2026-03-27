from utils.imports_statiques import *
from pipelines.utils_pipelines import *
from utils.data_simulate import *
from utils.data_preprocessing import *


def pipeline_simulate_data(
    hyperparams_data, 
    dir_level, 
    device
):
    
    # Data hyperparameters
    M_train = hyperparams_data["M_train"]
    M_test = M_test = max(1, int(hyperparams_data["M_train"] * 0.1))
    L = hyperparams_data["L"]
    d = hyperparams_data["d"]
    
    transforms = hyperparams_data.get("transforms", None)
    plot_params = hyperparams_data["plot_params"]
    plot_series = hyperparams_data["plot_series"]
    n_plots = hyperparams_data["n_plots"]
    force = hyperparams_data["force"]

    raw_label = hyperparams_data["data_label"]
    data_dir = f"{dir_level}/data_sim"
    
    data_label = add_label(f"{raw_label}SIM", "transforms", transforms)
    datagen_label = add_label(f"{raw_label}GEN", "transforms", transforms)
    
    # Simulators train / test
    sim_train = SimulateData(M=M_train, device=device, data_dir=data_dir, force=force, verbose=False)
    sim_test = SimulateData(M=M_test, device=device, data_dir=data_dir, force=force, verbose=False)
    
    # Pour les modèles Heston : theta_range est stocké sous heston_theta_range dans le config
    # pour éviter le conflit avec theta_range de OUrange (liste de listes)
    if "Heston" in raw_label and "heston_theta_range" in hyperparams_data:
        hyperparams_data["theta_range"] = hyperparams_data["heston_theta_range"]

    # Data parameters (hyperparams + default values)
    cfg = get_dataconfig(raw_label)
    params_data = read_params(hyperparams_data, cfg["params"])
    dt = params_data.get("dt", None)

    # Param dicts for plotting params
    params_dicts = None
    getp = getattr(SimulateData, cfg.get("get_params"))
    params_dicts = ensure_tuple_params(getp(**params_data))

    # Simulation (train/test)
    X_train, X_test, filename_train, filepath_train, filename_test, filepath_test = simulate_pair(sim_train, sim_test, cfg["simulate"], **params_data)
    X_train = take_first_d_dims(X_train, d)
    X_test = take_first_d_dims(X_test, d)
    
    # Series path & plots
    series_path = filename_train.replace(f"M={M_train}_", f"Mtrain={M_train}_Mtest={M_test}_").replace(".npy", "")

    # Summary Table
    summary = {
        "raw_label": raw_label,
        "data_label": data_label,
        "datagen_label": datagen_label,
        "d": d,
        "M_train": M_train,
        "M_test": M_test,
        "transforms": transforms,
        "filepath_train": filepath_train,
        "filepath_test": filepath_test,
        "series_path": series_path,
        "Mean": X_train.mean(),
        "Std": X_train.std(),
        "Min": X_train.min(),
        "Max": X_train.max(),
    }

    # Ajout des paramètres spécifiques du modèle (theta, mu, sigma, etc.)
    for k, v in params_data.items():
        summary[f"{k}"] = v

    plot_summary_table("SIMULATOR", summary)

    sim_fig_dir = f"{dir_level}/figures/data_sim/{data_label}"
    sim_fig_paths = plot_random_series(
        X_test=X_train,
        params_dicts=params_dicts,
        plot_params=plot_params,
        plot_series=plot_series,
        n_plots=n_plots,
        N=L,
        dt=dt,
        save_dir=sim_fig_dir,
        data_label=data_label,
        plot_path=series_path,
    )

    # Apply transforms
    x0 = hyperparams_data.get("x0", None)
    X_train, transform_meta = apply_transform(X_train, transforms=transforms, x0=x0)
    X_test, _ = apply_transform(X_test, transforms=transforms, x0=x0)

    data_params_dir = build_data_params_dir(
        {"raw_label": raw_label, "series_path": series_path, "transform": transforms},
        d,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "raw_label": raw_label,
        "data_label": data_label,
        "datagen_label": datagen_label,
        "series_path": series_path,
        "d": d,
        "data_params_dir": data_params_dir,
        "params_dicts": params_dicts,
        "transform": transforms,
        "transform_meta": transform_meta,
        "sim_fig_paths": sim_fig_paths or [],
    }