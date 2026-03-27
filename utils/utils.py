from utils.imports_statiques import *
from pipelines.pipeline import *


def get_device(device = None):
    def _is_available(device):
        dev = device.lower()
        if dev == "cuda":
            return torch.cuda.is_available()
        if dev == "mps":
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if dev == "cpu":
            return True
        return False

    if device is not None:
        pref = str(device).lower()
        if _is_available(pref):
            return pref
        print(f"[WARNING] Device '{device}' indisponible, fallback automatique.")

    if _is_available("cuda"):
        return "cuda"
    if _is_available("mps"):
        return "mps"
    return "cpu"


def set_seed(seed):
    """
    Sets the random seed across Python, NumPy, and PyTorch to ensure reproducibility.
    :param seed: [int]; the seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_dataloader(
    X, # attendu au final ~ [M, C, L]
    batch_size=1024,
    shuffle=True,
    drop_last=True,
    pin_memory=None,
    normalize=True,
    as_torch=True,
):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

    X = np.asarray(X)
    mu, sigma = None, None

    if X.ndim == 2:
        # (M, L) -> (M, 1, L)
        if normalize:
            mu = X.mean(axis=(0, 1), keepdims=True)
            sigma = X.std(axis=(0, 1), keepdims=True) + 1e-8
            X = (X - mu.squeeze()) / sigma.squeeze()
            mu = mu.reshape(1, 1, 1)
            sigma = sigma.reshape(1, 1, 1)

        X = X[:, None, :]  # (M, 1, L)

    elif X.ndim == 3:
        # X reçu comme (M, L, C) -> conversion en (M, C, L)
        X = np.transpose(X, (0, 2, 1))

        if normalize:
            mu = X.mean(axis=(0, 2), keepdims=True)          # (1, C, 1)
            sigma = X.std(axis=(0, 2), keepdims=True) + 1e-8 # (1, C, 1)
            X = (X - mu) / sigma

    else:
        raise ValueError(f"[ERROR] X doit être 2D ou 3D, reçu shape {X.shape}")

    X_torch = torch.as_tensor(X, dtype=torch.float32).contiguous() if as_torch else X
    ds = TensorDataset(X_torch)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        persistent_workers=False,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dl, mu, sigma


def safe_filename(name, ext=".npy", max_len=255):
    name = name.replace("/", "_")
    allowed_len = max_len - len(ext)
    if len(name) > allowed_len:
        name = name[:allowed_len]
    return name + ext