from utils.imports_statiques import *


# ============================================================
# Returns / log-returns: X ~ [M, L]
# ============================================================

def prices_to_logreturns(X):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"[ERROR] prices_to_logreturns expects 2D (M,L), got {X.shape}")
    R = np.zeros_like(X, dtype=np.float32)
    R[:, 1:] = np.diff(np.log(np.clip(X, 1e-12, None)), axis=1)
    return R


def prices_to_returns(X, normalize=True):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"[ERROR] prices_to_returns expects 2D (M,L), got {X.shape}")
    R = np.zeros_like(X, dtype=np.float32)
    if normalize:
        R[:, 1:] = X[:, 1:] / np.clip(X[:, :-1], 1e-12, None) - 1.0
    else:
        R[:, 1:] = X[:, 1:] - X[:, :-1]
    return R


def logreturns_to_prices(x0, R):
    R = np.asarray(R)
    if R.ndim != 2:
        raise ValueError(f"[ERROR] logreturns_to_prices expects 2D (M,L), got {R.shape}")
    x0 = np.asarray(x0, dtype=np.float32)
    if x0.ndim == 0:
        x0 = np.full((R.shape[0],), float(x0), dtype=np.float32)
    return (x0[:, None] * np.exp(np.cumsum(R, axis=1))).astype(np.float32)


def returns_to_prices(x0, R, normalize=True):
    R = np.asarray(R)
    if R.ndim != 2:
        raise ValueError(f"[ERROR] returns_to_prices expects 2D (M,L), got {R.shape}")
    x0 = np.asarray(x0, dtype=np.float32)
    if x0.ndim == 0:
        x0 = np.full((R.shape[0],), float(x0), dtype=np.float32)

    if normalize:
        return (x0[:, None] * np.cumprod(1.0 + R, axis=1)).astype(np.float32)

    X = np.zeros_like(R, dtype=np.float32)
    X[:, 0] = x0
    X[:, 1:] = x0[:, None] + np.cumsum(R[:, 1:], axis=1)
    return X


# ============================================================
# Normalizations
# ============================================================


def normalize_zscore(X, axis=None, eps=1e-8):
    X = np.asarray(X)
    if axis is None:
        mu = X.mean()
        sigma = X.std() + eps
        return ((X - mu) / sigma).astype(np.float32), float(mu), float(sigma)

    mu = X.mean(axis=axis, keepdims=True)
    sigma = X.std(axis=axis, keepdims=True) + eps
    return ((X - mu) / sigma).astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)


def denormalize_zscore(X, mu, sigma):
    X = np.asarray(X, dtype=np.float32)
    mu = np.asarray(mu, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)
    return (X * sigma + mu).astype(np.float32)


def normalize_minmax(X, axis=None, eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    if axis is None:
        x_min = float(np.min(X))
        x_max = float(np.max(X))
        denom = (x_max - x_min) + eps
        return ((X - x_min) / denom).astype(np.float32), x_min, x_max

    x_min = np.min(X, axis=axis, keepdims=True).astype(np.float32)
    x_max = np.max(X, axis=axis, keepdims=True).astype(np.float32)
    denom = (x_max - x_min) + eps
    return ((X - x_min) / denom).astype(np.float32), x_min, x_max


def denormalize_minmax(X, x_min, x_max, eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    x_min = np.asarray(x_min, dtype=np.float32)
    x_max = np.asarray(x_max, dtype=np.float32)
    return (X * ((x_max - x_min) + eps) + x_min).astype(np.float32)


# ============================================================
# Generic transform / inverse_transform
# ============================================================


def _ensure_MCL(X):
    X = np.asarray(X)
    if X.ndim == 2:
        # (M, L) -> (M, 1, L)
        X_mcl = X[:, None, :]
        def restore(Y_mcl):
            Y_mcl = np.asarray(Y_mcl)
            return Y_mcl[:, 0, :]  # back to (M, L)
        return X_mcl, restore, True

    if X.ndim == 3:
        # assume already (M, C, L)
        X_mcl = X
        def restore(Y_mcl):
            return np.asarray(Y_mcl)
        return X_mcl, restore, False

    raise ValueError(f"[ERROR] Expected 2D (M,L) or 3D (M,C,L), got {X.shape}")


def _parse_transforms(transforms, C):
    if transforms is None:
        return [None] * C

    if isinstance(transforms, str):
        return [transforms] * C

    if isinstance(transforms, (list, tuple)):
        if len(transforms) != C:
            raise ValueError(f"[ERROR] transforms list must have length C={C}, got {len(transforms)}")
        return list(transforms)

    if isinstance(transforms, dict):
        out = [None] * C
        for k, v in transforms.items():
            if not (0 <= int(k) < C):
                raise ValueError(f"[ERROR] transform dict key {k} out of range [0, {C-1}]")
            out[int(k)] = v
        return out

    raise ValueError(f"[ERROR] Unsupported transforms type: {type(transforms)}")


def _parse_x0(x0, C, M):
    if x0 is None:
        return [None] * C

    if np.isscalar(x0):
        v = float(x0)
        return [np.full((M,), v, dtype=np.float32) for _ in range(C)]

    if isinstance(x0, dict):
        out = [None] * C
        for k, v in x0.items():
            c = int(k)
            if not (0 <= c < C):
                raise ValueError(f"[ERROR] x0 dict key {k} out of range [0, {C-1}]")
            if v is None:
                out[c] = None
            elif np.isscalar(v):
                out[c] = np.full((M,), float(v), dtype=np.float32)
            else:
                va = np.asarray(v, dtype=np.float32)
                if va.ndim != 1 or va.shape[0] != M:
                    raise ValueError(f"[ERROR] x0[{c}] must be shape (M,), got {va.shape}")
                out[c] = va
        return out

    if isinstance(x0, (list, tuple)) and len(x0) == C:
        out = []
        for c in range(C):
            v = x0[c]
            if v is None:
                out.append(None)
            elif np.isscalar(v):
                out.append(np.full((M,), float(v), dtype=np.float32))
            else:
                va = np.asarray(v, dtype=np.float32)
                if va.ndim != 1 or va.shape[0] != M:
                    raise ValueError(f"[ERROR] x0[{c}] must be shape (M,), got {va.shape}")
                out.append(va)
        return out

    x0a = np.asarray(x0, dtype=np.float32)
    if x0a.ndim == 1 and x0a.shape[0] == M:
        return [x0a for _ in range(C)]
    if x0a.ndim == 1 and x0a.shape[0] == C:
        return [np.full((M,), float(x0a[c]), dtype=np.float32) for c in range(C)]
    if x0a.ndim == 2 and x0a.shape == (M, C):
        return [x0a[:, c] for c in range(C)]

    raise ValueError(f"[ERROR] Unsupported x0 type/shape: {type(x0)} / {getattr(x0, 'shape', None)}")


def apply_transform(X, transforms=None, x0=None):
    X_mcl, restore, input_was_2d = _ensure_MCL(X)
    X_mcl = np.asarray(X_mcl, dtype=np.float32)

    M, C, L = X_mcl.shape
    tr_list = _parse_transforms(transforms, C)

    Y = np.empty_like(X_mcl, dtype=np.float32)

    for c in range(C):
        Xt = X_mcl[:, c, :]
        tr = tr_list[c]

        if tr is None:
            Y[:, c, :] = Xt
        elif tr == "R":
            Y[:, c, :] = prices_to_returns(Xt, normalize=False)
        elif tr == "Rnorm":
            Y[:, c, :] = prices_to_returns(Xt, normalize=True)
        elif tr == "logR":
            Y[:, c, :] = prices_to_logreturns(Xt)
        else:
            raise ValueError(f"[ERROR] Unknown transform option for channel {c}: {tr}")

    meta = {
        "transforms": tr_list,
        "input_was_2d": input_was_2d,
        "C": C,
        "x0": x0,
    }

    return restore(Y), meta


def inverse_transform(X_trans, meta, x0=None):
    if not isinstance(meta, dict) or "transforms" not in meta:
        raise ValueError("[ERROR] meta must be the dict returned by transform().")

    transforms = meta["transforms"]

    X_mcl, restore, _ = _ensure_MCL(X_trans)
    X_mcl = np.asarray(X_mcl, dtype=np.float32)

    M, C, L = X_mcl.shape
    if len(transforms) != C:
        raise ValueError(f"[ERROR] meta.transforms length {len(transforms)} != C={C}")

    if x0 is None:
        x0 = meta.get("x0", None)

    x0_list = _parse_x0(x0, C=C, M=M)

    Y = np.empty_like(X_mcl, dtype=np.float32)

    for c in range(C):
        Xt = X_mcl[:, c, :]
        tr = transforms[c]

        if tr is None:
            Y[:, c, :] = Xt
            continue

        if tr in ("logR", "R", "Rnorm"):
            x0c = x0_list[c]
            if x0c is None:
                raise ValueError(f"[ERROR] inverse_transform for {tr} needs x0 for channel {c}.")
            if tr == "logR":
                Y[:, c, :] = logreturns_to_prices(x0c, Xt)
            elif tr == "R":
                Y[:, c, :] = returns_to_prices(x0c, Xt, normalize=False)
            else:  # "Rnorm"
                Y[:, c, :] = returns_to_prices(x0c, Xt, normalize=True)
            continue

        raise ValueError(f"[ERROR] Unknown transform option for channel {c}: {tr}")

    return restore(Y)

