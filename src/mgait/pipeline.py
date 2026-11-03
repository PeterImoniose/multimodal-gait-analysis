"""End-to-end processing of one trial and assembly of the analysis caches.

    python scripts/build_cache.py

Everything the notebooks read is produced here, so the processing settings are
fixed in one place (config.py) before any analysis is run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal

from . import config as C
from . import eeg as E
from . import emg as M
from . import gait as G
from . import io

SWAP_MARGIN = 0.01     # swapped alignment must beat the original by this correlation margin
SWAP_MAX_LAG = 0.10    # ... and land within this many seconds of zero


def emg_bad_channels(ratio_threshold: float = 5.0) -> dict:
    """Channel-trials whose median MAV is more than ratio_threshold times the
    population median for that muscle (noise-dominated or poorly attached)."""
    a = pd.read_parquet(C.TABLES / "audit_emg.parquet")
    cols = [c for c in a.columns if c.endswith("_median_uv")]
    ratio = a[cols] / a[cols].median()
    out = {}
    for i, r in a.iterrows():
        bad = {c.replace("_median_uv", "") for c in cols if ratio.loc[i, c] > ratio_threshold}
        out[(r["subject"], r["speed"])] = bad
    return out


def process_trial(subject: str, speed: str, bad_emg: set) -> dict:
    out = {"subject": subject, "speed": speed}
    fp = G.fp_process(subject, speed)
    im = G.imu_process(subject, speed)
    sync = {"subject": subject, "speed": speed, "fp": fp is not None, "imu": im is not None,
            "fp_usable": bool(fp is not None and fp.usable), "swapped": False}
    if fp is not None and fp.usable and im is not None:
        dur = min(fp.t[-1], im.t[-1])
        a = G.estimate_lag(fp.events, im.events, dur)
        b = G.estimate_lag(fp.events, im.events, dur, swap_other=True)
        sync.update(dict(a))
        sync.update({f"swap_{k}": v for k, v in b.items()})
        if b["peak_r"] > a["peak_r"] + SWAP_MARGIN and abs(b["lag_s"]) <= SWAP_MAX_LAG:
            fp = G.fp_process(subject, speed, swap=True)
            sync["swapped"] = True
            a = G.estimate_lag(fp.events, im.events, dur)
        sync.update({f"final_{k}": v for k, v in a.items()})
        m = G.match_events(fp.events, im.events, lag=0.0)
        for kind in ("hs", "to"):
            e = m.loc[m["kind"] == kind, "err"]
            sync[f"{kind}_detect"] = float(e.notna().mean())
            sync[f"{kind}_bias_ms"] = float(e.mean() * 1e3)
            sync[f"{kind}_sd_ms"] = float(e.std() * 1e3)
        out["event_errors"] = m.assign(subject=subject, speed=speed)
    if fp is not None:
        sync["fp_bw_n"] = fp.bw_n
        sync["fp_baseline_L"], sync["fp_baseline_R"] = fp.baseline_n["L"], fp.baseline_n["R"]
        sync["fp_note"] = fp.note
    out["sync"] = sync

    if fp is not None and fp.usable:
        out["fp_strides"], out["fp_curves"] = G.fp_stride_features(fp)
        out["fp_strides"]["swapped"] = sync["swapped"]
    if im is not None:
        out["imu_strides"], out["imu_curves"] = G.imu_stride_features(im)
        out["imu_strides"]["pelvis_speed"] = im.pelvis_speed

    # Events for EMG/EEG: the IMU shares their trigger network; force plates are the fallback
    ev_src = "imu" if im is not None else ("fp" if fp is not None and fp.usable else None)
    ev = im if ev_src == "imu" else fp if ev_src == "fp" else None
    if ev is not None:
        res = M.emg_cycles(subject, speed, ev.events, ev.strides, bad_emg)
        if res is not None:
            out["emg_cycles"], out["emg_curves"] = res
            out["emg_cycles"]["event_source"] = ev_src

    raw = io.load_eeg(subject, speed)
    if raw is not None:
        x = E.preprocess(raw)
        f, p = E.psd(x)
        bp = E.band_powers(f, p)
        st = None
        for src in (fp if fp is not None and fp.usable else None, im):
            if src is not None and not src.strides.empty:
                st = src.strides.loc[src.strides["qc_ok"], "stride_time"].median()
                break
        f_step = 2.0 / st if st else np.nan
        hr = E.harmonic_ratio(f, p, f_step) if st else np.full(len(C.EEG_CHANNELS), np.nan)
        coh = (E.head_coherence(x, im.head_acc[:, 2]) if im is not None and im.head_acc is not None
               else np.full(len(C.EEG_CHANNELS), np.nan))
        hs = np.concatenate([ev.events.hs["L"], ev.events.hs["R"]]) if ev is not None else np.array([])
        erp = E.hs_locked_average(x, np.sort(hs)) if len(hs) else None
        rows = []
        for j, ch in enumerate(C.EEG_CHANNELS):
            r = {"subject": subject, "speed": speed, "channel": ch, "harmonic_ratio": hr[j],
                 "head_coherence": coh[j], "f_step": f_step,
                 "log_power": float(np.log10(p[(f >= 1) & (f <= 45), j].sum()))}
            r.update({f"rel_{b}": float(v[j]) for b, v in bp.items()})
            rows.append(r)
        out["eeg_channels"] = pd.DataFrame(rows)
        out["eeg_psd"] = (f.astype(np.float32), p.astype(np.float32))
        out["eeg_erp"] = erp
        out["eeg_dec60"] = signal.decimate(x, 5, axis=0, zero_phase=True).astype(np.float32)
    if im is not None and im.head_acc is not None:
        out["head_acc"] = im.head_acc[:, 2].astype(np.float32)
    return out


def _stack(results: list[dict], table: str, curves: str | None):
    frames, arrs = [], {}
    for r in results:
        df = r.get(table)
        if df is None or len(df) == 0:
            continue
        frames.append(df)
        if curves:
            for k, v in r[curves].items():
                arrs.setdefault(k, []).append(v)
    df = pd.concat(frames, ignore_index=True)
    arrs = {k: np.concatenate(v) for k, v in arrs.items()}
    return df, arrs


def build(n_jobs: int = -1) -> None:
    inv = io.inventory()
    bad = emg_bad_channels()
    trials = [(r.subject, r.speed) for r in inv.itertuples()]
    results = Parallel(n_jobs=n_jobs)(delayed(process_trial)(s, v, bad.get((s, v), set())) for s, v in trials)

    pd.DataFrame([r["sync"] for r in results]).to_parquet(C.TABLES / "sync.parquet")
    pd.concat([r["event_errors"] for r in results if "event_errors" in r]).to_parquet(C.CACHE / "event_errors.parquet")
    for table, curves, name in (("fp_strides", "fp_curves", "fp"), ("imu_strides", "imu_curves", "imu"),
                                ("emg_cycles", "emg_curves", "emg")):
        df, arrs = _stack(results, table, curves)
        df.to_parquet(C.CACHE / f"{name}_strides.parquet")
        np.savez_compressed(C.CACHE / f"{name}_curves.npz", **arrs)
    eeg_rows = [r for r in results if "eeg_channels" in r]
    pd.concat([r["eeg_channels"] for r in eeg_rows]).to_parquet(C.CACHE / "eeg_channels.parquet")
    keys = np.array([f"{r['subject']}_{r['speed']}" for r in eeg_rows])
    np.savez_compressed(C.CACHE / "eeg_psd.npz", keys=keys, f=eeg_rows[0]["eeg_psd"][0],
                        P=np.stack([r["eeg_psd"][1] for r in eeg_rows]))
    erp_rows = [r for r in eeg_rows if r.get("eeg_erp") is not None]
    np.savez_compressed(C.CACHE / "eeg_erp.npz", keys=np.array([f"{r['subject']}_{r['speed']}" for r in erp_rows]),
                        erp=np.stack([r["eeg_erp"] for r in erp_rows]).astype(np.float32))
    # 60 Hz EEG and head acceleration, for the cross-subject coherence control
    np.savez_compressed(C.CACHE / "eeg_head60.npz",
                        **{f"eeg_{r['subject']}_{r['speed']}": r["eeg_dec60"] for r in eeg_rows},
                        **{f"head_{r['subject']}_{r['speed']}": r["head_acc"] for r in results if "head_acc" in r})
