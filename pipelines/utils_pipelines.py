from utils.imports_statiques import *
from utils.plots_timeseries import *
from pipelines.dicts import *

def _format_hparams_block(hyperparams_data, hyperparams_schedule, scorenet_label):
    # data: on enlève transform/force/plot_*
    ignore = {"transforms", "force", "plot_series", "plot_params", "n_plots"}
    data_items = [(k, v) for k, v in hyperparams_data.items() if k not in ignore]
    data_str = "DATA: " + ", ".join([f"{k}={v}" for k, v in data_items])

    # schedule (sans T qui est l'axe x)
    schedule_items = [(k, v) for k, v in hyperparams_schedule.items() if k != "T"]
    sched_str = "SCHEDULE: " + ", ".join([f"{k}={v}" for k, v in schedule_items])

    arch_str = f"ARCH: {scorenet_label}"
    return data_str + "\n" + sched_str + "\n" + arch_str

# ----------------------------------------
# --------- Common Utils -----------------
# ----------------------------------------


def add_label(baselabel, label, value):
    return f"{baselabel}_{label}={value}"

def add_labels_from_dict(baselabel, params_dict):
    out = baselabel
    for k, v in params_dict.items():
        out = add_label(out, k, v)
    return out

def derive_channels(hyperparams_training):
    """Dérive 'channels' depuis base_channel_width + nb_channels si nécessaire.
    Retourne un nouveau dict avec la clé 'channels' ajoutée (liste plate).
    Idempotent : si 'channels' existe déjà, retourne le dict inchangé.
    """
    if hyperparams_training.get("channels") is not None:
        return hyperparams_training
    base_channel_width = hyperparams_training.get("base_channel_width")
    nb_channels_arch   = hyperparams_training.get("nb_channels")
    if base_channel_width is None or nb_channels_arch is None:
        return hyperparams_training
    base_channel_width = int(base_channel_width)
    nb_channels_arch   = int(nb_channels_arch)
    nb_groups = hyperparams_training.get("nb_groups", 8)
    assert base_channel_width % nb_groups == 0, (
        f"base_channel_width ({base_channel_width}) must be divisible by nb_groups ({nb_groups})"
    )
    computed_channels = [base_channel_width * (2 ** i) for i in range(nb_channels_arch)]
    hp = {**hyperparams_training, "channels": computed_channels}
    if not hp.get("embed_dim"):
        hp["embed_dim"] = 4 * computed_channels[-1]
    return hp


def read_params(hp, params):
    out = {}
    for item in params:
        if isinstance(item, tuple):
            k, default = item
            out[k] = hp.get(k, default)
        else:
            out[item] = hp[item]
    return out

def plot_summary_table(title, data: dict):
    df = pd.DataFrame(
        [(k, v) for k, v in data.items()],
        columns=pd.MultiIndex.from_product([[title], ["field", "value"]])
    )
    display(df)


# ----------------------------------------
# --------- Utils pipeline_data ----------
# ----------------------------------------


def get_dataconfig(data_label):
    cfg = DATA_CONFIG.get(data_label)
    if cfg is None:
        raise ValueError(f"[SIMULATOR] data_label inconnu : {data_label}")
    return cfg


def ensure_tuple_params(params_dicts):
    if params_dicts is None:
        return None
    return params_dicts if isinstance(params_dicts, tuple) else (params_dicts,)


def simulate_pair(sim_train, sim_test, method_name, **kwargs):
    fn_train = getattr(sim_train, method_name)
    fn_test = getattr(sim_test, method_name)
    X_train, filename_train, filepath_train = fn_train(**kwargs)
    X_test, filename_test, filepath_test = fn_test(**kwargs)
    return X_train, X_test, filename_train, filepath_train, filename_test, filepath_test


def take_first_d_dims(X, d: int):
    X = np.asarray(X)
    if d < 1:
        raise ValueError(f"[SIMULATOR] d must be >= 1, got {d}")

    if X.ndim == 2:
        if d != 1:
            raise ValueError(f"[SIMULATOR] X is (B,L) so only d=1 is valid, got d={d}")
        return X

    if X.ndim != 3:
        raise ValueError(f"[SIMULATOR] Expected X of shape (B,L) or (B,L,C), got {X.shape}")

    C = X.shape[2]
    if d > C:
        raise ValueError(f"[SIMULATOR] d={d} exceeds C={C} for shape {X.shape}")

    return X[:, :, :d]


def plot_random_series(
    X_test,
    params_dicts,
    plot_params,
    plot_series,
    n_plots,
    N,
    dt,
    save_dir=None,
    data_label="",
    plot_path="series",
):
    if not plot_series:
        return []

    params1 = params_dicts[0] if (plot_params and params_dicts) else None
    save_img = save_dir is not None
    saved = []

    if X_test.ndim == 3:
        C = X_test.shape[-1]
        for c in range(C):
            fp = plot_random_time_series(
                X_data=X_test[:, :, c],
                params1=params1,
                save_img=save_img,
                plot_dir=save_dir or "plots",
                data_label=data_label,
                plot_path=f"ch{c}_{plot_path}",
                N=N,
                dt=dt,
                n_plot=n_plots,
                same_scale=False,
            )
            if fp:
                saved.append(fp)
        return saved

    fp = plot_random_time_series(
        X_data=X_test,
        params1=params1,
        save_img=save_img,
        plot_dir=save_dir or "plots",
        data_label=data_label,
        plot_path=plot_path,
        N=N,
        dt=dt,
        n_plot=n_plots,
    )
    if fp:
        saved.append(fp)
    return saved

# ----------------------------------------
# --------- Utils pipeline_schedule ----------
# ----------------------------------------

def get_scheduleconfig(schedule_label):
    cfg = SCHEDULE_CONFIG.get(schedule_label)
    if cfg is None:
        raise ValueError(f"[SCHEDULER] schedule_label inconnu : {schedule_label}")
    return cfg

# --------------------------------------------
# --------- Utils pipeline_training ----------
# --------------------------------------------


def get_modelconfig(model_label):
    cfg = MODEL_CONFIG.get(model_label)
    if cfg is None:
        raise ValueError(f"[TRAINER] model_label inconnu : {model_label}")
    return cfg


def instantiate_score_model(
    *,
    scorenet_label: str,
    Net,
    schedule_label: str,
    schedule,
    hyperparams_schedule: dict,
    hyperparams_training: dict,
    L: int,
    device,
):
    """
    Returns a torch.nn.Module (not DataParallel) ready to be wrapped.
    """

    in_channels = hyperparams_training["in_channels"]
    out_channels = hyperparams_training["out_channels"]
    channels = hyperparams_training["channels"]
    embed_dim = hyperparams_training["embed_dim"]
    kernel_size = hyperparams_training["kernel_size"]
    nb_groups = hyperparams_training["nb_groups"]

    return Net(
        perturbation_kernel_std=schedule.sigma,
        channels=channels,
        embed_dim=embed_dim,
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
        nb_groups=nb_groups,
    )


# --------------------------------------------
# --------- Utils pipeline_sampling ----------
# --------------------------------------------

def get_samplingconfig(data_label):
    cfg = SAMPLING_CONFIG.get(data_label)
    if cfg is None:
        raise ValueError(f"[SAMPLER] data_label inconnu : {data_label}")
    return cfg

def get_series_length(X) -> int:
    if X.ndim == 2:
        return int(X.shape[1])
    if X.ndim == 3:
        return int(X.shape[1])
    raise ValueError(f"Unexpected X shape: {X.shape}")


def as_BLC(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        return x[..., None]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected (B,L) or (B,L,C), got {x.shape}")


# --------------------------------------------
# --------- Utils pipeline_metrics ----------
# --------------------------------------------


def get_metricsconfig(data_label):
    cfg = METRICS_CONFIG.get(data_label)
    if cfg is None:
        raise ValueError(f"[METRICS] data_label inconnu : {data_label}")
    return cfg

def plot_params_distrib(*, raw_label: str, **kwargs):
    cfg = METRICS_CONFIG.get(raw_label)
    if cfg is None or "mle_fn" not in cfg:
        raise ValueError(f"[METRICS] plot_params_distrib: raw_label '{raw_label}' non supporté")

    return cfg["mle_fn"](**kwargs)


def split_params_dicts(params_dicts, n=6):
    out = [None] * n
    if params_dicts is None:
        return tuple(out)

    if not isinstance(params_dicts, (list, tuple)):
        params_dicts = (params_dicts,)

    for i in range(min(len(params_dicts), n)):
        out[i] = params_dicts[i]
    return tuple(out)


# --------------------------------------------
# --------- Utils pipeline_gridsearch --------
# --------------------------------------------


def expand_grid(hparams: dict) -> list[dict]:
    keys = list(hparams.keys())
    values = [(v if isinstance(v, list) else [v]) for v in (hparams[k] for k in keys)]
    return [{k: combo[i] for i, k in enumerate(keys)} for combo in itertools.product(*values)]


def short_hash(obj, n=8) -> str:
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:n]


def ensure_list(x, name="value"):
    if not isinstance(x, list) or len(x) == 0:
        raise ValueError(f"{name} doit être une liste non vide.")
    return x


def get_run_id(hp_data: dict, hp_sched: dict, hp_train: dict, n=10) -> str:
    return short_hash({"data": hp_data, "schedule": hp_sched, "training": hp_train}, n=n)


def inject_run_id(hp_train: dict, run_id: str) -> dict:
    hp = copy.deepcopy(hp_train)
    hp["run_id"] = run_id
    return hp


# ── Hierarchical path builders ────────────────────────────────────────────────

def build_data_params_dir(data_out: dict, d: int) -> str:
    """Level 2: data params from series_path (raw_label stripped) + d + transforms."""
    raw_label   = data_out["raw_label"]
    series_path = data_out["series_path"]
    transforms  = data_out.get("transform")

    # strip leading "{raw_label}_" to avoid redundancy with Level-1 folder
    params = series_path[len(raw_label) + 1:] if series_path.startswith(raw_label + "_") else series_path
    params = f"{params}_d={d}"
    if transforms is not None:
        params = f"{params}_transforms={transforms}"
    return params


def build_schedule_dir_full(schedule_out: dict, hyperparams_schedule: dict) -> str:
    """Level 3: schedule dir + T + eps."""
    T   = hyperparams_schedule["T"]
    eps = hyperparams_schedule["eps"]
    return f"{schedule_out['schedule_dir']}_T={T}_eps={eps}"


def build_training_dir(hyperparams_training: dict) -> str:
    """Level 5: training hyperparams directory name."""
    hp    = hyperparams_training
    sched = hp.get("scheduler")
    parts = [
        f"lr={hp['lr']}",
        f"ep={hp['n_epochs']}",
        f"bs={hp['batch_size']}",
        f"pat={hp['patience'] if hp.get('patience') is not None else 'none'}",
        f"dl={hp.get('drop_last', True)}",
    ]
    if sched is not None:
        parts += [
            f"sfac={sched.get('factor', 0.5)}",
            f"smod={sched.get('mode', 'min')}",
            f"spat={sched.get('patience', 10)}",
            f"sthr={sched.get('threshold', 1e-4)}",
            f"sthm={sched.get('threshold_mode', 'rel')}",
            f"scdn={sched.get('cooldown', 0)}",
            f"smin={sched.get('min_lr', 0.0)}",
        ]
    return "_".join(parts)


def build_sampling_dir(hyperparams_sampling: dict) -> str:
    """Sampling sub-directory (last level under data_gen/ and figures/)."""
    hp = hyperparams_sampling
    return f"{hp['method']}_Mgen={hp['M_gen']}_steps={hp['nb_steps']}_eps={hp['eps_sample']}"


def build_experiments_base(
    data_out: dict,
    schedule_out: dict,
    hyperparams_schedule: dict,
    hyperparams_training: dict,
    scorenet_dir: str,
    d: int,
) -> str:
    """Full 5-level base path shared by experiments/, data_gen/, figures/."""
    lvl1 = data_out["raw_label"]
    lvl2 = build_data_params_dir(data_out, d)
    lvl3 = build_schedule_dir_full(schedule_out, hyperparams_schedule)
    lvl4 = scorenet_dir
    lvl5 = build_training_dir(hyperparams_training)
    return f"{lvl1}/{lvl2}/{lvl3}/{lvl4}/{lvl5}"


def flatten_cfg(cfg, prefix=""):
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_cfg(v, key))
        else:
            if isinstance(v, (list, tuple, np.ndarray)):
                out[key] = tuple(v)
            else:
                out[key] = v
    return out


def varying_keys(runs, allowed_prefixes=("schedule.", "training.", "data.")):
    vals = defaultdict(set)
    for r in runs:
        flat = flatten_cfg(r["config"])
        for k, v in flat.items():
            if any(k.startswith(p) for p in allowed_prefixes):
                vals[k].add(v)
    return {k for k, s in vals.items() if len(s) > 1}, vals


def short_key(k):
    mapping = {
        "training.scorenet_label": "arch",
        "schedule.T": "T",
        "data.transform": "tr",
        "training.kernel_size": "K",
        "training.channels": "ch",
        "training.embed_dim": "emb",
        "training.lr": "lr",
        "training.batch_size": "bs",
        "training.n_epochs": "ep",
        "schedule.schedule_label": "sched",
        "schedule.betatype": "beta",
        "schedule.betamin": "bmin",
        "schedule.betamax": "bmax",
        "schedule.sigmamin": "smin",
        "schedule.sigmamax": "smax",
        "schedule.rho": "rho",
    }
    return mapping.get(k, k.split(".")[-1])


def fmt(v):
    if isinstance(v, float):
        if float(v).is_integer():
            return str(int(v))
        return f"{v:.3g}"
    return str(v)


def make_auto_label(cfg, varying_keys, order=None, max_parts=6):
    flat = flatten_cfg(cfg)
    keys = [k for k in flat.keys() if k in varying_keys]
    if order is not None:
        keys = sorted(keys, key=lambda k: (order.index(k) if k in order else 10_000, k))
    else:
        keys = sorted(keys)

    parts = [f"{short_key(k)}={_fmt(flat[k])}" for k in keys[:max_parts]]
    if len(keys) > max_parts:
        parts.append("…")
    return " ".join(parts) if parts else "run"