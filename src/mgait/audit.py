"""Per-trial quality checks for every modality.

Each function reads one trial and returns a flat dict of checks, so the full
release can be scanned in parallel and stored as one table per modality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from . import config as C
from . import io


def _band_ratio(x: np.ndarray, fs: float, f0: float, half: float = 1.0) -> float:
    """Power within +/-half Hz of f0 relative to 1-45 Hz power (Welch)."""
    f, p = signal.welch(x, fs=fs, nperseg=int(fs * 4))
    band = (f >= f0 - half) & (f <= f0 + half)
    ref = (f >= 1) & (f <= 45)
    return float(p[band].sum() / max(p[ref].sum(), 1e-12))


def audit_eeg(subject: str, speed: str) -> tuple[dict, list[dict]] | None:
    if not io.trial_paths(subject, speed)["eeg"].exists():
        return None
    hdr = io.eeg_header(subject, speed)
    if not io.eeg_header_ok(subject, speed):
        return {"subject": subject, "speed": speed, "header_ok": False,
                "date": hdr.get("Date"), "time": hdr.get("Time")}, []
    raw = io.load_eeg(subject, speed, reref=False)
    t = raw.index.to_numpy()
    dt = np.diff(t)
    seq = raw.attrs["adc_sequence"].astype(int)
    seq_gap = int(((np.diff(seq) % 256) != 1).sum())
    trig = raw.attrs["trigger"]
    n_trig_changes = int((np.diff(trig) != 0).sum())
    trial = {
        "subject": subject, "speed": speed, "header_ok": True, "n": len(raw), "duration_s": float(t[-1] - t[0]),
        "fs_est": float((len(t) - 1) / (t[-1] - t[0])), "dt_std_ms": float(dt.std() * 1e3),
        "dt_max_ms": float(dt.max() * 1e3), "adc_seq_gaps": seq_gap,
        "trigger_values": ",".join(str(v) for v in np.unique(trig)), "trigger_changes": n_trig_changes,
        "date": hdr.get("Date"), "time": hdr.get("Time"), "mains_hz": hdr.get("Mains_Frequency_(Hz)"),
        "reference": hdr.get("Reference location"),
    }
    reref = io.load_eeg(subject, speed, reref=True)
    fs = C.FS_EEG
    b, a = signal.butter(4, [1, 40], btype="band", fs=fs)
    chans = []
    for ch in C.EEG_CHANNELS:
        x = reref[ch].to_numpy()
        xf = signal.filtfilt(b, a, x)
        chans.append({
            "subject": subject, "speed": speed, "channel": ch,
            "raw_offset_uv": float(np.median(x)),
            "flat_frac": float((np.abs(np.diff(x)) < 1e-6).mean()),
            "bp_sd_uv": float(xf.std()),
            "bp_absmax_uv": float(np.abs(xf).max()),
            "frac_over_100uv": float((np.abs(xf) > 100).mean()),
            "line50_ratio": _band_ratio(x - x.mean(), fs, 50.0),
        })
    return trial, chans


def audit_emg(subject: str, speed: str) -> dict | None:
    df = io.load_emg(subject, speed)
    if df is None:
        return None
    t = df.index.to_numpy()
    out = {"subject": subject, "speed": speed, "n": len(df), "duration_s": float(t[-1] - t[0]),
           "fs_est": float((len(t) - 1) / (t[-1] - t[0])), "n_nan": int(df.isna().sum().sum()),
           "n_channels": df.shape[1]}
    for c in df.columns:
        x = df[c].to_numpy()
        out[f"{c}_median_uv"] = float(np.median(x))
        out[f"{c}_p99_uv"] = float(np.percentile(x, 99))
        # Fraction of the trial at the channel's floor (a disconnected or saturated sensor sits flat)
        out[f"{c}_flat_frac"] = float((np.abs(np.diff(x)) < 1e-3).mean())
    return out


def audit_fp(subject: str, speed: str, mass_kg: float | None = None) -> dict | None:
    df = io.load_fp(subject, speed)
    if df is None:
        return None
    b, a = signal.butter(4, 20, fs=C.FS_FP)
    lz = signal.filtfilt(b, a, df["L_Fz"].to_numpy(float))
    rz = signal.filtfilt(b, a, df["R_Fz"].to_numpy(float))
    tot = lz + rz
    bw_n = float(np.median(tot))
    out = {
        "subject": subject, "speed": speed, "n": len(df), "duration_s": len(df) / C.FS_FP,
        "timestamps_all_zero": df.attrs["timestamps_all_zero"], "has_header": df.attrs["has_header"],
        "L_COPx_med": float(df["L_COPx"].median()), "L_COPy_med": float(df["L_COPy"].median()),
        "R_COPx_med": float(df["R_COPx"].median()), "R_COPy_med": float(df["R_COPy"].median()),
        "total_fz_median_n": bw_n, "mass_from_fp_kg": bw_n / C.G,
        "L_share": float(np.mean(lz) / np.mean(tot)),
        # Unloaded-belt baseline: the 2nd percentile of each plate (both feet are never off together in walking)
        "L_baseline_n": float(np.percentile(lz, 2)), "R_baseline_n": float(np.percentile(rz, 2)),
        "frac_total_below_half_bw": float((tot < 0.5 * bw_n).mean()),
        "n_nan": int(df.isna().sum().sum()),
    }
    if mass_kg is not None and np.isfinite(mass_kg):
        out["mass_reported_kg"] = mass_kg
        out["mass_diff_pct"] = 100 * (out["mass_from_fp_kg"] - mass_kg) / mass_kg
    return out


def audit_imu(subject: str, speed: str) -> dict | None:
    folder = io.trial_paths(subject, speed)["imu"]
    if not folder.exists():
        return None
    info = io.imu_info(subject, speed)
    ja = io.load_imu(subject, speed, "Joint_Angles_ZXY")
    pelvis = io.load_imu(subject, speed, "Segment_Position", ["Pelvis x", "Pelvis y", "Pelvis z"])
    empty = sorted(p.stem for p in folder.glob("*.csv") if p.stat().st_size < 100 and p.stem != "Markers")
    files = sorted(p.stem for p in folder.glob("*.csv"))
    t = ja.index.to_numpy()
    knee = ja[["Right Knee Flexion/Extension", "Left Knee Flexion/Extension"]].to_numpy()
    return {
        "subject": subject, "speed": speed, "n": len(ja), "duration_s": float(t[-1] - t[0]),
        "frame_rate": float(info.get("Frame Rate", "nan")), "quality": info.get("Processing Quality"),
        "recorded_utc": info.get("Recorded Date UTC"), "source_file": info.get("Original File Name", "").split("\\")[-1],
        "n_files": len(files), "empty_files": ",".join(empty), "n_empty": len(empty),
        "n_nan": int(ja.isna().sum().sum()),
        "knee_rom_R": float(np.ptp(knee[:, 0])), "knee_rom_L": float(np.ptp(knee[:, 1])),
        # MVN integrates treadmill walking as overground progression, so the pelvis
        # path length over the trial is the suit's own estimate of walking speed
        "pelvis_path_m": float(np.hypot(np.diff(pelvis["Pelvis x"]), np.diff(pelvis["Pelvis y"])).sum()),
        "pelvis_net_m": float(np.hypot(pelvis["Pelvis x"].iloc[-1] - pelvis["Pelvis x"].iloc[0],
                                       pelvis["Pelvis y"].iloc[-1] - pelvis["Pelvis y"].iloc[0])),
        "pelvis_z_mean_m": float(pelvis["Pelvis z"].mean()),
    }


def check_metadata(record_dates: pd.Series | None = None) -> pd.DataFrame:
    """Row-level consistency checks on the released subject sheet.

    record_dates: optional per-subject recording date (from the EEG headers),
    used to recompute age from date of birth.
    """
    m = io.load_subjects(keep_dob=True)
    out = m[["sex", "age", "height_mm", "mass_kg", "bmi_reported", "bmi", "preferred_speed"]].copy()
    out["bmi_abs_diff"] = (out["bmi"] - out["bmi_reported"]).abs()
    dob = pd.to_datetime(m["dob"].str.replace("\\", "/", regex=False), format="%d/%m/%Y", errors="coerce")
    out["dob_parsed"] = dob.notna()
    if record_dates is not None:
        rd = pd.to_datetime(record_dates.reindex(out.index), format="%d/%m/%Y", errors="coerce")
        out["record_date"] = rd
        out["age_from_dob"] = np.floor((rd - dob).dt.days / 365.25)
        out["age_diff"] = out["age"] - out["age_from_dob"]
    return out
