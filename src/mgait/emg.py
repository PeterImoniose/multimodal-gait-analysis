"""EMG: gait-cycle activation profiles and muscle synergies.

The release provides the Delsys mean-absolute-value (MAV) output at about
16 Hz rather than raw EMG, so frequency-domain EMG analysis is not possible.
The MAV envelope is still well suited to timing and synergy analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from . import config as C
from . import io
from .gait import Events, resample_curve

MAV_WINDOW_S = 1.0 / C.FS_EMG


def emg_cycles(subject: str, speed: str, ev: Events, strides: pd.DataFrame,
               bad_channels: set[str] = frozenset()) -> tuple[pd.DataFrame, dict] | None:
    """Ipsilateral EMG for each stride, time-normalised to 101 points (uV).

    Gait events come from a system on the same trigger network (IMU) or one
    verified to be aligned with it (force plates). Each MAV sample summarises
    the preceding window, so its timestamp is shifted back by half a window.
    """
    emg = io.load_emg(subject, speed)
    if emg is None or strides.empty:
        return None
    t = emg.index.to_numpy() - MAV_WINDOW_S / 2
    rows, curves = [], {m: [] for m in C.MUSCLES}
    for r in strides[strides["qc_ok"]].itertuples():
        if r.t_next_hs > t[-1] or r.t_hs < t[0]:
            continue
        grid = np.linspace(r.t_hs, r.t_next_hs, 101)
        row = {"side": r.side, "t_hs": r.t_hs, "stride_time": r.stride_time, "stance_pct": r.stance_pct}
        for m in C.MUSCLES:
            col = f"{r.side}_{m}"
            curves[m].append(np.interp(grid, t, emg[col].to_numpy()) if col not in bad_channels
                             else np.full(101, np.nan))
        rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.insert(0, "speed", speed)
    df.insert(0, "subject", subject)
    return df, {m: np.array(v, np.float32) for m, v in curves.items()}


def vaf(X: np.ndarray, Xhat: np.ndarray) -> float:
    """Variance accounted for (uncentred), the usual synergy goodness-of-fit."""
    return float(1 - np.sum((X - Xhat) ** 2) / np.sum(X ** 2))


def synergies(X: np.ndarray, k_max: int = 5, seed: int = C.SEED) -> dict:
    """NMF of a muscles x samples matrix for k = 1..k_max.

    X should be non-negative and each muscle scaled to unit peak (so that
    large-amplitude muscles do not dominate). Returns VAF per k and the
    weights W (muscles x k) and activations H (k x samples) per k.
    """
    out = {"vaf": {}, "vaf_muscle": {}, "W": {}, "H": {}}
    for k in range(1, k_max + 1):
        model = NMF(n_components=k, init="nndsvda", max_iter=3000, tol=1e-6, random_state=seed)
        W = model.fit_transform(X)
        H = model.components_
        # put the scale in H so that each synergy vector has unit norm
        norm = np.linalg.norm(W, axis=0) + 1e-12
        W, H = W / norm, H * norm[:, None]
        Xhat = W @ H
        out["vaf"][k] = vaf(X, Xhat)
        out["vaf_muscle"][k] = 1 - np.sum((X - Xhat) ** 2, axis=1) / np.sum(X ** 2, axis=1)
        out["W"][k], out["H"][k] = W, H
    return out


def n_synergies(fit: dict, threshold: float = 0.90, muscle_threshold: float = 0.75) -> int:
    """Smallest k whose overall VAF reaches `threshold` and whose per-muscle
    VAF reaches `muscle_threshold` for every muscle."""
    for k in sorted(fit["vaf"]):
        if fit["vaf"][k] >= threshold and fit["vaf_muscle"][k].min() >= muscle_threshold:
            return k
    return max(fit["vaf"])


def prepare_matrix(X: np.ndarray, floor_pct: float = 2.0) -> np.ndarray:
    """Remove each muscle's noise floor (its low percentile) and scale to unit peak.

    The MAV envelope never reaches zero (sensor and skin noise), and an
    uncorrected floor is a constant shared by all muscles that a single
    synergy reconstructs, which inflates VAF."""
    X = X - np.percentile(X, floor_pct, axis=1, keepdims=True)
    X = np.clip(X, 0, None)
    return X / X.max(axis=1, keepdims=True)


def match_synergies(W_ref: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedy best-cosine pairing of the columns of W to those of W_ref."""
    from scipy.optimize import linear_sum_assignment
    S = W_ref.T @ W
    r, c = linear_sum_assignment(-S)
    return c, S[r, c]
