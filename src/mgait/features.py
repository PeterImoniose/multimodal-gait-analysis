"""Trial-level and stride-level feature tables built from the caches.

One row per (subject, speed) trial, with features grouped by modality so that
each modality can be modelled on its own or fused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import eeg as E
from . import io

FP_FEATURES = ["stride_time", "stance_pct", "double_support_pct", "step_time", "fz_peak1", "fz_valley",
               "fz_peak2", "loading_rate", "fz_impulse", "fy_brake_peak", "fy_prop_peak",
               "fy_brake_impulse", "fy_prop_impulse", "fx_range", "fx_medial_peak"]
IMU_FEATURES = ["stride_time", "stance_pct", "hip_flex_max", "hip_ext_max", "hip_rom", "knee_flex_stance",
                "knee_flex_swing", "knee_rom", "knee_at_hs", "ankle_dorsi_max", "ankle_plantar_max",
                "ankle_rom", "hip_abd_rom"]


def load_cache():
    fp = pd.read_parquet(C.CACHE / "fp_strides.parquet")
    imu = pd.read_parquet(C.CACHE / "imu_strides.parquet")
    emg = pd.read_parquet(C.CACHE / "emg_strides.parquet")
    return fp, imu, emg


def symmetry_index(left: float, right: float) -> float:
    """Robinson symmetry index (%): 0 is perfect symmetry."""
    return float(200 * abs(left - right) / (left + right)) if (left + right) else np.nan


def _trial_summary(df: pd.DataFrame, feats: list[str], prefix: str) -> pd.DataFrame:
    ok = df[df["qc_ok"]]
    g = ok.groupby(["subject", "speed"])
    out = g[feats].median().add_prefix(prefix)
    out[f"{prefix}stride_cv"] = g["stride_time"].std() / g["stride_time"].mean() * 100
    out[f"{prefix}n_strides"] = g.size()
    side = ok.groupby(["subject", "speed", "side"])[["stance_time", "stride_time"] if "stance_time" in ok else ["stride_time"]].median()
    if "stance_time" in ok:
        si = side["stance_time"].unstack("side")
        out[f"{prefix}stance_si"] = 200 * (si["L"] - si["R"]).abs() / (si["L"] + si["R"])
    return out


def emg_trial_features(emg: pd.DataFrame, curves: dict) -> pd.DataFrame:
    """Per muscle (left and right averaged): mean activation (log10 uV), peak
    timing (% cycle) and stance/swing activity ratio."""
    key = emg["subject"] + "|" + emg["speed"]
    rows = {}
    for m in C.MUSCLES:
        X = curves[m]
        df = pd.DataFrame(X)
        df["key"], df["side"], df["stance"] = key.values, emg["side"].values, emg["stance_pct"].values
        prof = df.groupby(["key", "side"]).mean(numeric_only=True)
        stance = prof.pop("stance")
        vals = prof.to_numpy()
        mean_act = np.log10(np.nanmean(vals, axis=1))
        peak = np.where(np.isnan(vals).all(1), np.nan, np.nanargmax(np.nan_to_num(vals, nan=-1), axis=1))
        so = np.round(stance.to_numpy()).astype(int)
        ratio = np.array([np.nanmean(v[:s]) / np.nanmean(v[s:]) if s < 100 else np.nan for v, s in zip(vals, so)])
        t = pd.DataFrame({f"emg_{m}_act": mean_act, f"emg_{m}_peak": peak, f"emg_{m}_st_sw": np.log10(ratio)},
                         index=prof.index).groupby(level="key").mean()
        rows[m] = t
    out = pd.concat(rows.values(), axis=1)
    out.index = pd.MultiIndex.from_tuples([tuple(k.split("|")) for k in out.index], names=["subject", "speed"])
    return out


def eeg_trial_features(eeg: pd.DataFrame) -> pd.DataFrame:
    """Region-averaged relative band power, log power, gait-harmonic ratio and
    head-acceleration coherence."""
    reg = {ch: r for r, chs in E.REGIONS.items() for ch in chs}
    d = eeg.assign(region=eeg["channel"].map(reg))
    cols = [f"rel_{b}" for b in E.BANDS] + ["log_power", "harmonic_ratio", "head_coherence"]
    d["harmonic_ratio"] = np.log10(d["harmonic_ratio"])
    w = d.groupby(["subject", "speed", "region"])[cols].mean().unstack("region")
    w.columns = [f"eeg_{r}_{c}" for c, r in w.columns]
    return w


def trial_table() -> pd.DataFrame:
    fp, imu, emg = load_cache()
    meta = io.load_subjects()
    t_fp = _trial_summary(fp, FP_FEATURES, "fp_")
    t_imu = _trial_summary(imu.rename(columns={}), IMU_FEATURES, "imu_")
    t_imu["imu_speed_est"] = imu.groupby(["subject", "speed"])["pelvis_speed"].first()
    t_emg = emg_trial_features(emg, dict(np.load(C.CACHE / "emg_curves.npz")))
    t_eeg = eeg_trial_features(pd.read_parquet(C.CACHE / "eeg_channels.parquet"))
    inv = io.inventory().set_index(["subject", "speed"])
    T = inv.join([t_fp, t_imu, t_emg, t_eeg], how="left")
    sync = pd.read_parquet(C.TABLES / "sync.parquet").set_index(["subject", "speed"])
    T["fp_imu_lag_s"] = sync["final_lag_s"]
    T["fp_imu_synced"] = (sync["final_lag_s"].abs() <= 0.1) & (sync["hs_detect"] > 0.9)
    T["fp_swapped"] = sync["swapped"]
    T = T.reset_index()
    T["speed_ms"] = T["speed"].map(C.SPEED_MS)
    T = T.join(meta[["sex", "age", "height_m", "mass_kg", "bmi", "bmi_class", "leg_len_m", "preferred_speed"]],
               on="subject")
    T["rel_speed"] = T["speed_ms"] / T["preferred_speed"]
    T["froude"] = T["speed_ms"] ** 2 / (C.G * T["leg_len_m"])
    T["fp_cadence"] = 120 / T["fp_stride_time"]
    T["fp_stride_length"] = T["speed_ms"] * T["fp_stride_time"]
    T["fp_stride_length_norm"] = T["fp_stride_length"] / T["leg_len_m"]
    T["pilot"] = T["subject"].isin(C.PILOT_SUBJECTS)
    return T


def modality_columns(T: pd.DataFrame) -> dict:
    return {
        "Force plates": [c for c in T if c.startswith("fp_") and T[c].dtype.kind == "f"
                         and c not in ("fp_imu_lag_s", "fp_n_strides", "fp_stride_length", "fp_stride_length_norm")],
        "IMU": [c for c in T if c.startswith("imu_") and c not in ("imu_n_strides", "imu_speed_est")],
        "EMG": [c for c in T if c.startswith("emg_")],
        "EEG": [c for c in T if c.startswith("eeg_")],
    }
