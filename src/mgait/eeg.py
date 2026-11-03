"""EEG during walking: spectra, band power and gait-locked contamination checks.

Dry-electrode EEG recorded while walking is exposed to cable sway, electrode
micro-movement and head acceleration at the step frequency. The checks here
quantify how much of the recorded signal is locked to the gait cycle, which
determines whether any downstream EEG feature can be read as brain activity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from . import config as C
from . import io

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
REGIONS = {
    "frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "central": ["C3", "Cz", "C4"],
    "temporal": ["T3", "T4", "T5", "T6"],
    "parietal": ["P3", "Pz", "P4"],
    "occipital": ["O1", "O2"],
}


def preprocess(eeg: pd.DataFrame, fs: float = C.FS_EEG) -> np.ndarray:
    """1 Hz high-pass and 50 Hz notch (zero-phase); returns samples x channels."""
    x = eeg.to_numpy(float)
    x = x - x.mean(axis=0)
    b, a = signal.butter(4, 1.0, btype="high", fs=fs)
    x = signal.filtfilt(b, a, x, axis=0)
    bn, an = signal.iirnotch(50.0, 30.0, fs=fs)
    return signal.filtfilt(bn, an, x, axis=0)


def psd(x: np.ndarray, fs: float = C.FS_EEG, seg_s: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    f, p = signal.welch(x, fs=fs, nperseg=int(seg_s * fs), axis=0)
    keep = (f >= 0.5) & (f <= 45)
    return f[keep], p[keep]


def band_powers(f: np.ndarray, p: np.ndarray) -> dict:
    """Relative band power (fraction of 1-45 Hz power) for each channel."""
    tot = p[(f >= 1) & (f <= 45)].sum(axis=0)
    out = {}
    for name, (lo, hi) in BANDS.items():
        out[name] = p[(f >= lo) & (f < hi)].sum(axis=0) / tot
    return out


def harmonic_ratio(f: np.ndarray, p: np.ndarray, f_step: float, n_harm: int = 6,
                   half: float = 0.1, guard: float = 0.2, side: float = 0.5) -> np.ndarray:
    """Mean power at the first n step-frequency harmonics divided by power in
    flanking bands (per channel). 1.0 means no gait-locked peak; values above
    1 mean spectral peaks locked to stepping."""
    num, den = 0, 0
    for k in range(1, n_harm + 1):
        fk = k * f_step
        if fk + guard + side > f[-1]:
            break
        on = np.abs(f - fk) <= half
        off = (np.abs(f - fk) > guard) & (np.abs(f - fk) <= guard + side)
        num = num + p[on].mean(axis=0)
        den = den + p[off].mean(axis=0)
    return num / den


def head_coherence(x: np.ndarray, head_acc: np.ndarray, fs_eeg: float = C.FS_EEG,
                   fs_imu: float = C.FS_IMU, band: tuple = (0.5, 10.0)) -> np.ndarray:
    """Mean magnitude-squared coherence (in band) between each EEG channel and
    vertical head acceleration. Coherence magnitude does not depend on a small
    constant timing offset between the systems."""
    q = int(round(fs_eeg / fs_imu))
    xd = signal.decimate(x, q, axis=0, zero_phase=True)
    n = min(len(xd), len(head_acc))
    xd, h = xd[:n], head_acc[:n]
    out = []
    for j in range(xd.shape[1]):
        f, c = signal.coherence(xd[:, j], h, fs=fs_imu, nperseg=int(4 * fs_imu))
        out.append(c[(f >= band[0]) & (f <= band[1])].mean())
    return np.array(out)


def hs_locked_average(x: np.ndarray, hs_times: np.ndarray, fs: float = C.FS_EEG,
                      pre: float = 0.2, post: float = 0.8) -> np.ndarray:
    """Average EEG (samples x channels) around heel strikes (low-passed at 30 Hz)."""
    b, a = signal.butter(4, 30, fs=fs)
    xl = signal.filtfilt(b, a, x, axis=0)
    i0, i1 = int(pre * fs), int(post * fs)
    epochs = []
    for t in hs_times:
        c = int(round(t * fs))
        if c - i0 >= 0 and c + i1 < len(xl):
            seg = xl[c - i0:c + i1]
            epochs.append(seg - seg[:i0].mean(axis=0))
    return np.mean(epochs, axis=0) if epochs else np.full((i0 + i1, x.shape[1]), np.nan)
