"""
Nearest Neighbor Distance (NND) — memorization metric for diffusion models.

Principle
---------
For each generated sample, find its nearest neighbor in the *training* set (L2).
Compare this to the typical within-training-set distance (self-NND).

    nnd_ratio = nnd_gen_train / nnd_train_train

  ratio << 1  →  generated samples collapse onto training points  → MEMORIZATION
  ratio ≈ 1   →  generated samples are as "spread out" as the training set → GENERALIZATION
  ratio >> 1  →  generated samples are far from all training points → MODE DROPPING

References
----------
- Carlini et al. (2023) "Extracting Training Data from Diffusion Models"
- "On Memorization in Diffusion Models" (NeurIPS 2023)
"""

import numpy as np


def _batch_nn2_dists(X_query: np.ndarray, X_ref: np.ndarray) -> tuple:
    """
    For each row in X_query (M_q, D), return distances to its 1st and 2nd
    nearest neighbours in X_ref (M_r, D).

    Returns
    -------
    dists_nn1 : (M_q,) — distance to nearest neighbour
    dists_nn2 : (M_q,) — distance to second nearest neighbour
    """
    assert X_ref.shape[0] >= 2, "X_ref must have at least 2 samples to compute NN2"

    M_q = X_query.shape[0]
    dists_nn1 = np.empty(M_q, dtype=np.float32)
    dists_nn2 = np.empty(M_q, dtype=np.float32)

    sq_r = np.sum(X_ref ** 2, axis=1)  # (M_r,)

    batch = 256
    for start in range(0, M_q, batch):
        end = min(start + batch, M_q)
        xq = X_query[start:end]                        # (B, D)
        sq_q = np.sum(xq ** 2, axis=1, keepdims=True)  # (B, 1)
        dists_sq = np.maximum(
            sq_q + sq_r[None, :] - 2.0 * (xq @ X_ref.T), 0.0
        )  # (B, M_r)

        # argpartition guarantees the 2 smallest entries are in [:2] (unsorted)
        idx2 = np.argpartition(dists_sq, kth=1, axis=1)[:, :2]
        for bi in range(end - start):
            d1, d2 = np.sort(dists_sq[bi, idx2[bi]])
            dists_nn1[start + bi] = np.sqrt(d1)
            dists_nn2[start + bi] = np.sqrt(d2)

    return dists_nn1, dists_nn2


def compute_fmem(X_gen: np.ndarray, X_train: np.ndarray, k: float = 1 / 3) -> float:
    """
    Compute the memorization fraction f_mem.

    A generated sample x is considered memorised if:
        ||x - NN1||_2 / ||x - NN2||_2  <  k
    where NN1 and NN2 are its 1st and 2nd nearest neighbours in X_train (L2).

    Reference: Yoon et al. (2023), Bonnaire et al. (2025) — k = 1/3.

    Parameters
    ----------
    X_gen   : (M_gen, L) or (M_gen, L, d)
    X_train : (M_train, L) or (M_train, L, d) — must have >= 2 rows
    k       : memorisation threshold (default 1/3)

    Returns
    -------
    f_mem : float in [0, 1] — fraction of memorised generated samples
    """
    X_gen   = np.asarray(X_gen,   dtype=np.float32)
    X_train = np.asarray(X_train, dtype=np.float32)

    if X_gen.ndim == 3:
        X_gen   = X_gen.reshape(X_gen.shape[0],     -1)
    if X_train.ndim == 3:
        X_train = X_train.reshape(X_train.shape[0], -1)

    assert X_gen.ndim == 2 and X_train.ndim == 2, \
        f"Expected 2-D arrays after flattening, got {X_gen.shape}, {X_train.shape}"

    if X_train.shape[0] < 2:
        return 0.0  # cannot compute NN2

    dists_nn1, dists_nn2 = _batch_nn2_dists(X_gen, X_train)

    # avoid division by zero (if NN2 == 0, set ratio to inf → not memorised)
    safe_nn2 = np.where(dists_nn2 > 0.0, dists_nn2, np.inf)
    ratios   = dists_nn1 / safe_nn2

    return float(np.mean(ratios < k))


def _batch_nn_dist(X_query: np.ndarray, X_ref: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """
    For each row in X_query (M_q, D), find the L2 distance to its nearest
    neighbour in X_ref (M_r, D).

    Args:
        X_query:      shape (M_q, D) — flattened samples
        X_ref:        shape (M_r, D) — flattened reference samples
        exclude_self: if True, masks out the exact self-match (use when
                      X_query and X_ref are the same array)

    Returns:
        dists: shape (M_q,) — minimum L2 distance per query sample
    """
    M_q = X_query.shape[0]
    min_dists = np.empty(M_q, dtype=np.float32)

    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b^T  (numerically stable batching)
    sq_r = np.sum(X_ref ** 2, axis=1)  # (M_r,)

    batch = 256  # rows processed at once to keep memory usage low
    for start in range(0, M_q, batch):
        end = min(start + batch, M_q)
        xq = X_query[start:end]                       # (B, D)
        sq_q = np.sum(xq ** 2, axis=1, keepdims=True) # (B, 1)
        dists_sq = sq_q + sq_r[None, :] - 2.0 * (xq @ X_ref.T)  # (B, M_r)
        dists_sq = np.maximum(dists_sq, 0.0)           # numerical safety

        if exclude_self:
            for bi in range(end - start):
                dists_sq[bi, start + bi] = np.inf

        min_dists[start:end] = np.sqrt(np.min(dists_sq, axis=1))

    return min_dists


def compute_nnd(X_gen: np.ndarray, X_train: np.ndarray) -> dict:
    """
    Compute Nearest Neighbor Distance metrics between generated and training samples.

    Inputs can be:
      - 2-D: (M, L)       — univariate time series
      - 3-D: (M, L, d)    — multivariate time series  (flattened to (M, L*d))

    Returns
    -------
    dict with keys:
        nnd_gen_train   : mean min-L2 distance  generated → training  (scalar)
        nnd_train_train : mean min-L2 distance  training  → training  (baseline, scalar)
        nnd_ratio       : nnd_gen_train / nnd_train_train
                          < 1 = memorization, ≈ 1 = generalisation, > 1 = mode-dropping
    """
    X_gen = np.asarray(X_gen, dtype=np.float32)
    X_train = np.asarray(X_train, dtype=np.float32)

    # Flatten to 2-D
    if X_gen.ndim == 3:
        X_gen = X_gen.reshape(X_gen.shape[0], -1)
    if X_train.ndim == 3:
        X_train = X_train.reshape(X_train.shape[0], -1)

    assert X_gen.ndim == 2 and X_train.ndim == 2, \
        f"Expected 2-D arrays after flattening, got {X_gen.shape}, {X_train.shape}"

    # --- gen → train (memorization signal) ---
    dists_gt = _batch_nn_dist(X_gen, X_train, exclude_self=False)
    nnd_gen_train = float(np.mean(dists_gt))

    # --- train → train  (within-set baseline, exclude self-match) ---
    dists_tt = _batch_nn_dist(X_train, X_train, exclude_self=True)
    nnd_train_train = float(np.mean(dists_tt))

    # --- ratio ---
    nnd_ratio = (nnd_gen_train / nnd_train_train) if nnd_train_train > 0.0 else float("inf")

    return {
        "nnd_gen_train":   nnd_gen_train,    # raw distance — smaller = more memorisation
        "nnd_train_train": nnd_train_train,  # baseline distance between training samples
        "nnd_ratio":       nnd_ratio,        # < 1 = memorisation, ≈ 1 = generalisation
    }
