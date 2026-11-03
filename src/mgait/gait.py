"""Gait events, strides and stride-level features from force plates and IMU.

Force plates (split-belt treadmill, one plate per foot) are the reference for
gait events: a foot is in contact while its plate carries more than a small
fraction of body weight. The IMU suit gives an independent estimate from foot
segment speed, which is near zero in stance once MVN has converted treadmill
walking into overground progression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import signal

from . import config as C
from . import io

N_POINTS = 101  # time-normalised curve length (0-100 % of stance or stride)


# ----------------------------------------------------------------------------- helpers

def lowpass(x: np.ndarray, fc: float, fs: float, order: int = 4, axis: int = 0) -> np.ndarray:
    b, a = signal.butter(order, fc, fs=fs)
    return signal.filtfilt(b, a, x, axis=axis)


def contacts_from_mask(mask: np.ndarray, fs: float, min_on: float, min_off: float) -> tuple[np.ndarray, np.ndarray]:
    """Onset/offset sample indices of True runs, after closing short gaps and
    dropping short runs. Runs touching the trial edges are discarded."""
    m = mask.astype(bool).copy()
    d = np.diff(m.astype(int))
    on, off = np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1
    if m[0]:
        off = off[1:] if len(off) and (len(on) == 0 or off[0] < on[0]) else off
    if len(on) and len(off) and off[0] < on[0]:
        off = off[1:]
    n = min(len(on), len(off))
    on, off = on[:n], off[:n]
    # close gaps shorter than min_off
    keep_on, keep_off = [], []
    for a, b in zip(on, off):
        if keep_off and (a - keep_off[-1]) < min_off * fs:
            keep_off[-1] = b
        else:
            keep_on.append(a)
            keep_off.append(b)
    on, off = np.array(keep_on, int), np.array(keep_off, int)
    ok = (off - on) >= min_on * fs
    return on[ok], off[ok]


def resample_curve(x: np.ndarray, n: int = N_POINTS) -> np.ndarray:
    """Linear time-normalisation of a 1-D or 2-D (time x channels) segment."""
    x = np.asarray(x, float)
    src = np.linspace(0, 1, len(x))
    dst = np.linspace(0, 1, n)
    if x.ndim == 1:
        return np.interp(dst, src, x)
    return np.column_stack([np.interp(dst, src, x[:, j]) for j in range(x.shape[1])])


@dataclass
class Events:
    """Contact onsets (heel strike, HS) and offsets (toe off, TO) in seconds."""
    hs: dict = field(default_factory=dict)   # side -> array
    to: dict = field(default_factory=dict)

    def contact_series(self, fs: float, duration: float) -> dict:
        n = int(round(duration * fs))
        out = {}
        for side in ("L", "R"):
            s = np.zeros(n)
            for a, b in zip(self.hs[side], self.to[side]):
                s[int(a * fs):min(int(b * fs), n)] = 1
            out[side] = s
        return out


def stride_table(ev: Events) -> pd.DataFrame:
    """One row per complete stride (HS to next ipsilateral HS) with temporal parameters."""
    rows = []
    for side, other in (("L", "R"), ("R", "L")):
        hs, to = ev.hs[side], ev.to[side]
        ohs, oto = ev.hs[other], ev.to[other]
        for i in range(len(hs) - 1):
            t0, t1 = hs[i], hs[i + 1]
            tos = to[(to > t0) & (to < t1)]
            if len(tos) != 1:
                continue
            tto = tos[0]
            c_hs = ohs[(ohs > t0) & (ohs < t1)]
            c_to = oto[(oto > t0) & (oto < t1)]
            if len(c_hs) != 1 or len(c_to) != 1:
                continue
            stride = t1 - t0
            # double support: from ipsilateral HS to contralateral TO, and from contralateral HS to ipsilateral TO
            ds1 = max(c_to[0] - t0, 0) if c_to[0] < tto else 0.0
            ds2 = max(tto - c_hs[0], 0) if c_hs[0] < tto else 0.0
            rows.append({
                "side": side, "idx": i, "t_hs": t0, "t_to": tto, "t_next_hs": t1,
                "stride_time": stride, "stance_time": tto - t0, "swing_time": t1 - tto,
                "stance_pct": 100 * (tto - t0) / stride,
                "step_time": t1 - c_hs[0],  # contralateral HS to this foot's next HS
                "double_support_pct": 100 * (ds1 + ds2) / stride,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("t_hs").reset_index(drop=True)
    med = df.groupby("side")["stride_time"].transform("median")
    df["qc_ok"] = ((df["stride_time"] - med).abs() < 0.25 * med) & df["stance_pct"].between(45, 85)
    return df


# ----------------------------------------------------------------------------- force plates

FP_FS_OUT = 200.0  # processed force data are kept at 200 Hz


@dataclass
class FPTrial:
    subject: str
    speed: str
    t: np.ndarray
    z: dict
    y: dict
    x: dict
    copy: dict
    bw_n: float
    baseline_n: dict
    swapped: bool
    usable: bool
    note: str
    events: Events
    strides: pd.DataFrame


def _plate_baseline(z: np.ndarray) -> float:
    """Unloaded level of a plate: median of the lowest quarter of samples (swing phase)."""
    return float(np.median(z[z <= np.percentile(z, 25)]))


def fp_process(subject: str, speed: str, swap: bool = False) -> FPTrial | None:
    raw = io.load_fp(subject, speed)
    if raw is None:
        return None
    fs = C.FS_FP
    step = int(fs / FP_FS_OUT)
    sides = ("R", "L") if swap else ("L", "R")
    z, y, x, cy, base = {}, {}, {}, {}, {}
    for out_side, src in zip(("L", "R"), sides):
        fz = lowpass(raw[f"{src}_Fz"].to_numpy(float), 20, fs)
        b = _plate_baseline(fz)
        fz = fz - b
        swing = fz < 30
        fy = lowpass(raw[f"{src}_Fy"].to_numpy(float), 20, fs)
        fx = lowpass(raw[f"{src}_Fx"].to_numpy(float), 20, fs)
        fy -= np.median(fy[swing]) if swing.any() else 0
        fx -= np.median(fx[swing]) if swing.any() else 0
        z[out_side], y[out_side], x[out_side] = fz[::step], fy[::step], fx[::step]
        cy[out_side] = raw[f"{src}_COPy"].to_numpy(float)[::step]
        base[out_side] = b
    t = np.arange(len(z["L"])) / FP_FS_OUT
    bw = float(np.mean(z["L"] + z["R"]))
    usable, note = True, ""
    if bw < 200:
        usable, note = False, "plates unloaded (total vertical force near zero)"
    ev = Events()
    if usable:
        thr = max(30.0, 0.05 * bw)
        for side in ("L", "R"):
            on, off = contacts_from_mask(z[side] > thr, FP_FS_OUT, min_on=0.2, min_off=0.08)
            ev.hs[side], ev.to[side] = t[on], t[np.minimum(off, len(t) - 1)]
        strides = stride_table(ev)
    else:
        ev.hs = {"L": np.array([]), "R": np.array([])}
        ev.to = {"L": np.array([]), "R": np.array([])}
        strides = pd.DataFrame()
    return FPTrial(subject, speed, t, z, y, x, cy, bw, base, swap, usable, note, ev, strides)


def fp_stride_features(tr: FPTrial) -> tuple[pd.DataFrame, dict]:
    """Kinetic features per stride (forces in body weights) and stance-normalised curves."""
    if not tr.usable or tr.strides.empty:
        return pd.DataFrame(), {}
    fs, bw = FP_FS_OUT, tr.bw_n
    # Sign of the anterior-posterior force: braking (first half of stance) is made negative
    first, last = [], []
    for r in tr.strides.itertuples():
        i0, i1 = int(r.t_hs * fs), int(r.t_to * fs)
        seg = tr.y[r.side][i0:i1]
        k = max(len(seg) // 3, 1)
        first.append(seg[:k].mean())
        last.append(seg[-k:].mean())
    ap_sign = -1.0 if np.median(first) > np.median(last) else 1.0
    # Medio-lateral force points towards the body midline for most of stance; orient each side so
    # that medial is positive (the two plates' x axes point the same way, so raw signs are opposite)
    ml_sign = {}
    for side in ("L", "R"):
        means = [tr.x[side][int(r.t_hs * fs):int(r.t_to * fs)].mean()
                 for r in tr.strides.itertuples() if r.side == side]
        ml_sign[side] = 1.0 if (np.median(means) if means else 1.0) >= 0 else -1.0
    feats, curves = [], {"Fz": [], "Fy": [], "Fx": []}
    for r in tr.strides.itertuples():
        i0, i1 = int(r.t_hs * fs), int(r.t_to * fs)
        fz = tr.z[r.side][i0:i1] / bw
        fy = ap_sign * tr.y[r.side][i0:i1] / bw
        fx = ml_sign[r.side] * tr.x[r.side][i0:i1] / bw
        cz, cy_, cx = resample_curve(fz), resample_curve(fy), resample_curve(fx)
        p1i = int(np.argmax(cz[:50]))
        p2i = 50 + int(np.argmax(cz[50:]))
        valley = float(cz[p1i:p2i + 1].min()) if p2i > p1i else np.nan
        # loading rate between 20 % and 80 % of the first peak (BW/s)
        seg = fz[:max(int(np.argmax(fz[:len(fz) // 2])) + 1, 2)]
        pk = seg.max()
        try:
            a = np.flatnonzero(seg >= 0.2 * pk)[0]
            b = np.flatnonzero(seg >= 0.8 * pk)[0]
            lr = (0.6 * pk) / max((b - a) / fs, 1 / fs)
        except IndexError:
            lr = np.nan
        cop = tr.copy[r.side][i0:i1]
        mid = slice(int(0.1 * len(cop)), int(0.9 * len(cop)) or 1)
        feats.append({
            "fz_peak1": float(cz[p1i]), "fz_peak2": float(cz[p2i]), "fz_valley": valley,
            "fz_peak1_pct": float(p1i), "fz_peak2_pct": float(p2i),
            "loading_rate": float(lr),
            "fz_impulse": float(np.trapezoid(fz, dx=1 / fs)),
            "fy_brake_peak": float(cy_[:60].min()), "fy_prop_peak": float(cy_[40:].max()),
            "fy_brake_impulse": float(np.trapezoid(np.minimum(fy, 0), dx=1 / fs)),
            "fy_prop_impulse": float(np.trapezoid(np.maximum(fy, 0), dx=1 / fs)),
            "fx_range": float(np.ptp(cx)), "fx_medial_peak": float(cx.max()),
            "cop_ap_excursion": float(np.ptp(cop[mid])) if len(cop) > 10 else np.nan,
        })
        curves["Fz"].append(cz)
        curves["Fy"].append(cy_)
        curves["Fx"].append(cx)
    df = pd.concat([tr.strides.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    df.insert(0, "speed", tr.speed)
    df.insert(0, "subject", tr.subject)
    df["bw_n"] = bw
    return df, {k: np.array(v, np.float32) for k, v in curves.items()}


# ----------------------------------------------------------------------------- IMU

IMU_ANGLES = {
    "hip_flex": "Hip Flexion/Extension", "hip_abd": "Hip Abduction/Adduction",
    "knee_flex": "Knee Flexion/Extension", "ankle_dorsi": "Ankle Dorsiflexion/Plantarflexion",
}
SIDE_NAME = {"L": "Left", "R": "Right"}


@dataclass
class IMUTrial:
    subject: str
    speed: str
    t: np.ndarray
    foot_speed: dict
    angles: dict          # side -> DataFrame of angle series
    pelvis_speed: float
    head_acc: np.ndarray | None
    events: Events
    strides: pd.DataFrame
    thr_frac: float


def imu_process(subject: str, speed: str, thr_frac: float = C.IMU_THR_FRAC) -> IMUTrial | None:
    """Gait events from foot segment horizontal speed: contact while speed is
    below thr_frac of the trial's 95th-percentile foot speed."""
    vel = io.load_imu(subject, speed, "Segment_Velocity")
    if vel is None:
        return None
    ja = io.load_imu(subject, speed, "Joint_Angles_ZXY")
    acc = io.load_imu(subject, speed, "Segment_Acceleration", ["Head x", "Head y", "Head z"])
    fs = C.FS_IMU
    t = vel.index.to_numpy()
    ev, fsp, ang = Events(), {}, {}
    for side in ("L", "R"):
        nm = SIDE_NAME[side]
        sp = np.hypot(vel[f"{nm} Foot x"].to_numpy(float), vel[f"{nm} Foot y"].to_numpy(float))
        sp = lowpass(sp, 6, fs)
        thr = thr_frac * np.percentile(sp, 95)
        on, off = contacts_from_mask(sp < thr, fs, min_on=0.25, min_off=0.15)
        ev.hs[side], ev.to[side] = t[on], t[np.minimum(off, len(t) - 1)]
        fsp[side] = sp
        ang[side] = pd.DataFrame({k: ja[f"{nm} {v}"].to_numpy(float) for k, v in IMU_ANGLES.items()})
    pel = np.hypot(vel["Pelvis x"].to_numpy(float), vel["Pelvis y"].to_numpy(float))
    strides = stride_table(ev)
    return IMUTrial(subject, speed, t, fsp, ang, float(np.median(pel)),
                    acc.to_numpy(float) if acc is not None else None, ev, strides, thr_frac)


def imu_stride_features(tr: IMUTrial) -> tuple[pd.DataFrame, dict]:
    """Kinematic features per stride and stride-normalised joint-angle curves."""
    if tr.strides.empty:
        return pd.DataFrame(), {}
    fs = C.FS_IMU
    feats, curves = [], {k: [] for k in IMU_ANGLES}
    for r in tr.strides.itertuples():
        i0, i1 = int(round(r.t_hs * fs)), int(round(r.t_next_hs * fs)) + 1
        a = tr.angles[r.side].iloc[i0:i1].to_numpy()
        cur = resample_curve(a)
        so = int(round(r.stance_pct))
        hip, knee, ank, abd = cur[:, 0], cur[:, 2], cur[:, 3], cur[:, 1]
        feats.append({
            "hip_flex_max": hip.max(), "hip_ext_max": -hip.min(), "hip_rom": np.ptp(hip),
            "knee_flex_stance": knee[:max(so // 2, 1)].max(), "knee_flex_swing": knee[so:].max() if so < 100 else np.nan,
            "knee_rom": np.ptp(knee), "knee_at_hs": knee[0],
            "ankle_dorsi_max": ank[:so].max() if so > 0 else np.nan, "ankle_plantar_max": -ank.min(),
            "ankle_rom": np.ptp(ank), "hip_abd_rom": np.ptp(abd),
        })
        for j, k in enumerate(IMU_ANGLES):
            curves[k].append(cur[:, j])
    df = pd.concat([tr.strides.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    df.insert(0, "speed", tr.speed)
    df.insert(0, "subject", tr.subject)
    return df, {k: np.array(v, np.float32) for k, v in curves.items()}


# ----------------------------------------------------------------------------- alignment

def estimate_lag(ev_ref: Events, ev_other: Events, duration: float, fs: float = 100.0,
                 max_lag: float = 4.0, swap_other: bool = False) -> dict:
    """Time shift that best aligns two sets of contact events.

    Returns lag (s) such that other(t + lag) matches ref(t), the peak
    correlation, and the ratio of the best peak to the best peak more than
    0.3 s away (a value near 1 means the alignment is ambiguous, because gait
    is periodic).
    """
    a = ev_ref.contact_series(fs, duration)
    b = ev_other.contact_series(fs, duration)
    if swap_other:
        b = {"L": b["R"], "R": b["L"]}
    lags = np.arange(-int(max_lag * fs), int(max_lag * fs) + 1)
    n = len(a["L"])
    score = np.zeros(len(lags))
    for side in ("L", "R"):
        x = a[side] - a[side].mean()
        y = b[side] - b[side].mean()
        full = signal.correlate(y, x, mode="full")  # index n-1 is zero lag
        mid = len(x) - 1
        score += full[mid + lags] / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)
    score /= 2
    k = int(np.argmax(score))
    far = np.abs(lags - lags[k]) > 0.3 * fs
    second = float(score[far].max()) if far.any() else np.nan
    return {"lag_s": float(lags[k] / fs), "peak_r": float(score[k]), "second_r": second,
            "peak_ratio": float(score[k] / second) if second and second > 0 else np.nan}


def match_events(ref: Events, other: Events, lag: float = 0.0, tol: float = 0.3) -> pd.DataFrame:
    """Pair each reference event with the nearest event of the same type and
    side in the other system (after shifting it by -lag). Unmatched reference
    events get NaN error."""
    rows = []
    for kind in ("hs", "to"):
        for side in ("L", "R"):
            r = getattr(ref, kind)[side]
            o = getattr(other, kind)[side] - lag
            for tr in r:
                if len(o) == 0:
                    rows.append({"kind": kind, "side": side, "t": tr, "err": np.nan})
                    continue
                j = int(np.argmin(np.abs(o - tr)))
                e = o[j] - tr
                rows.append({"kind": kind, "side": side, "t": tr, "err": e if abs(e) <= tol else np.nan})
    return pd.DataFrame(rows)


def apply_offsets(ev: Events, hs_offset: float, to_offset: float) -> Events:
    """Shift detected events by fixed calibration offsets (seconds)."""
    return Events(hs={k: v - hs_offset for k, v in ev.hs.items()},
                  to={k: v - to_offset for k, v in ev.to.items()})
