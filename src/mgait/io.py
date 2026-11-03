"""Loaders for every modality, returning tidy pandas objects with clean names.

Trials are identified by (subject, speed), e.g. ("S10", "0.75"). Loaders return
None for a trial that is not in the release, so callers can skip it explicitly.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C

# ----------------------------------------------------------------------------- metadata

_META_RENAME = {
    "Subject": "subject", "IMPAIRED/HEALTHY": "status", "GENDER": "sex", "DOB": "dob",
    "Age": "age", "HEIGHT(mm)": "height_mm", "BMI": "bmi_reported", "WEIGHT(kg)": "mass_kg",
    "FOOT LENGTH (mm)": "foot_len_mm", "L-UPPER ARM": "upper_arm_len", "L-LOWER ARM": "lower_arm_len",
    "L-UPPER LEG": "upper_leg_len", "L-LOWER LEG": "lower_leg_len", "L-LEG": "leg_len",
    "NECK": "neck_len", "SHOULDER": "shoulder_len", "C-CHEST": "chest_circ", "C-WAIST": "waist_circ",
    "C-HIP": "hip_circ", "C-THIGH": "thigh_circ", "C-ANKLE": "ankle_circ",
    "C-UPPER ARM": "upper_arm_circ", "C-WRIST": "wrist_circ", "Normal Speed (m/s)": "preferred_speed",
}


def load_subjects_raw() -> pd.DataFrame:
    """The metadata sheet exactly as released (including its summary rows)."""
    return pd.read_csv(C.SUBJECT_FILE)


def load_subjects(keep_dob: bool = False) -> pd.DataFrame:
    """Participant table: one row per subject, summary rows removed.

    Date of birth is a direct identifier and is dropped unless explicitly
    requested (it is only used in the audit to check the age column).
    BMI is recomputed from height and mass; the WHO category is derived from it.
    """
    df = load_subjects_raw().rename(columns=_META_RENAME)
    df = df[df["subject"].astype(str).str.fullmatch(r"S\d+")].copy()
    num = [c for c in df.columns if c not in ("subject", "status", "sex", "dob")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df["sex"] = df["sex"].str.strip().str.upper()
    df["height_m"] = df["height_mm"] / 1000.0
    df["bmi"] = df["mass_kg"] / df["height_m"] ** 2
    df["bmi_class"] = pd.cut(df["bmi"], [0, 25, 30, 100], right=False,
                             labels=["Normal", "Overweight", "Obese"])
    df["leg_len_m"] = df["leg_len"] / 1000.0
    if not keep_dob:
        df = df.drop(columns=["dob"])
    df["sid"] = df["subject"].str[1:].astype(int)
    return df.sort_values("sid").set_index("subject")


# ----------------------------------------------------------------------------- inventory

def trial_paths(subject: str, speed: str) -> dict:
    return {
        "eeg": C.EEG_DIR / f"{subject}_{speed}_raw.csv",
        "emg": C.EMG_DIR / f"EMG_{subject}_{speed}.csv",
        "fp": C.FP_DIR / f"FP_{subject}_{speed}.csv",
        "imu": C.IMU_DIR / f"IMU_{subject}_{speed}",
    }


def inventory() -> pd.DataFrame:
    """Which modality exists for every subject x speed."""
    rows = []
    for s in C.SUBJECTS:
        for v in C.SPEEDS:
            p = trial_paths(s, v)
            rows.append({"subject": s, "speed": v, **{k: p[k].exists() for k in p}})
    inv = pd.DataFrame(rows)
    inv["all4"] = inv[["eeg", "emg", "fp", "imu"]].all(axis=1)
    return inv


# ----------------------------------------------------------------------------- EEG

EEG_STANDARD_HEADER = ["Time", "P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz", "CM", "A1", "Fp1", "Fp2",
                       "T3", "T5", "O1", "O2", "X3", "X2", "F7", "F8", "X1", "A2", "T6", "T4"]


def eeg_header_ok(subject: str, speed: str) -> bool:
    path = trial_paths(subject, speed)["eeg"]
    with open(path, encoding="utf-8", errors="replace") as fh:
        for _ in range(15):
            fh.readline()
        cols = [c.strip() for c in fh.readline().split(",")]
    return cols[:25] == EEG_STANDARD_HEADER

def eeg_header(subject: str, speed: str) -> dict:
    path = trial_paths(subject, speed)["eeg"]
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for _ in range(15):
            line = fh.readline()
            if line.startswith("#"):
                key, _, rest = line[1:].partition(",")
                out[key.strip(" =:")] = rest.split(",")[0].strip()
    return out


def load_eeg(subject: str, speed: str, reref: bool = True) -> pd.DataFrame | None:
    """EEG in microvolts, 19 channels in 10-20 order, indexed by time (s).

    The headset records against Pz, so Pz is absent from the file. With
    reref=True every channel, including the reconstructed Pz (0 V against
    itself), is re-referenced to linked ears (A1+A2)/2, as in the authors'
    EEG_referencing.m script.
    """
    path = trial_paths(subject, speed)["eeg"]
    if not path.exists():
        return None
    df = pd.read_csv(path, skiprows=15)
    df.columns = [c.strip() for c in df.columns]
    if list(df.columns[:25]) != EEG_STANDARD_HEADER:
        # One release file (S32_1) carries a corrupted montage header in both CSV and EDF;
        # its channel identities cannot be verified, so it is treated as unusable.
        return None
    sig = df[[c for c in C.EEG_CHANNELS if c != "Pz"] + ["A1", "A2"]].astype(float).copy()
    sig["Pz"] = 0.0
    if reref:
        le = (sig["A1"] + sig["A2"]) / 2.0
        sig = sig.sub(le, axis=0)
    out = sig[C.EEG_CHANNELS].copy()
    out.index = pd.Index(df["Time"].to_numpy(float), name="t")
    out.attrs["trigger"] = df["Trigger"].to_numpy()
    out.attrs["adc_sequence"] = df["ADC_Sequence"].to_numpy()
    out.attrs["a1a2"] = sig[["A1", "A2"]].to_numpy() if not reref else None
    return out


# ----------------------------------------------------------------------------- EMG

_EMG_RE = re.compile(r"^(R|L) ([A-Z ]+?): EMG")
_EMG_ABBR = {"TIBIALIS ANTERIOR": "TA", "VASTUS LATERALIS": "VL", "RECTUS FEMORIS": "RF",
             "GASTROCNEMIUS MEDIAL HEAD": "GM", "SEMITENDINOSUS": "ST",
             "GASTROCNEMIUS LATERAL HEAD": "GL"}


def load_emg(subject: str, speed: str) -> pd.DataFrame | None:
    """EMG mean-absolute-value envelopes in microvolts, columns like 'R_TA'."""
    path = trial_paths(subject, speed)["emg"]
    if not path.exists():
        return None
    df = pd.read_csv(path)
    names = {}
    for c in df.columns[1:]:
        m = _EMG_RE.match(c.strip())
        if m:
            names[c] = f"{m.group(1)}_{_EMG_ABBR[m.group(2).strip()]}"
    out = df[list(names)].rename(columns=names).astype(float) * 1e6
    out.index = pd.Index(df.iloc[:, 0].to_numpy(float), name="t")
    return out


# ----------------------------------------------------------------------------- force plates

def load_fp(subject: str, speed: str) -> pd.DataFrame | None:
    """Bilateral force plate data at 1000 Hz, columns like 'L_Fz', 'R_COPy'.

    Plate 1 is the left belt and plate 2 the right belt (dataset README). The
    exported timestamp, sync and aux columns are all zero, so time is
    reconstructed from the sample index.
    """
    path = trial_paths(subject, speed)["fp"]
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        has_header = fh.readline().startswith("1:Timestamp")
    if has_header:
        df = pd.read_csv(path, dtype=np.float32, engine="pyarrow")
    else:
        # S2 files are exported without a header row; the 22 columns are in the standard order
        df = pd.read_csv(path, header=None, names=FP_COLUMNS, dtype=np.float32, engine="pyarrow")
    keep = {}
    for side, num in (("L", "1"), ("R", "2")):
        for q in ("Fx", "Fy", "Fz", "Mx", "My", "Mz", "COPx", "COPy"):
            keep[f"{num}:{q}"] = f"{side}_{q}"
    out = df[list(keep)].rename(columns=keep)
    out.index = pd.Index(np.arange(len(out)) / C.FS_FP, name="t")
    out.attrs["timestamps_all_zero"] = bool((df["1:Timestamp"] == 0).all())
    out.attrs["has_header"] = has_header
    return out


FP_COLUMNS = [f"{n}:{q}" for n in ("1", "2") for q in
              ("Timestamp", "Sync", "Aux", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "COPx", "COPy")]


# ----------------------------------------------------------------------------- IMU

IMU_FILES = ["Joint_Angles_ZXY", "Segment_Position", "Segment_Velocity", "Segment_Acceleration",
             "Segment_Angular_Velocity", "Center_of_Mass", "Sensor_Free_Acceleration"]


def imu_info(subject: str, speed: str) -> dict:
    path = trial_paths(subject, speed)["imu"] / "General_Information.csv"
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        k, _, v = line.partition(",")
        out[k.strip()] = v.strip()
    return out


def load_imu(subject: str, speed: str, kind: str = "Joint_Angles_ZXY",
             columns: list[str] | None = None) -> pd.DataFrame | None:
    """One Xsens MVN export file for a trial, indexed by time (s) at 60 Hz."""
    folder = trial_paths(subject, speed)["imu"]
    path = folder / f"{kind}.csv"
    if not path.exists() or path.stat().st_size < 100:
        # A few exports (S46_1, all of S47) contain empty placeholder files
        return None
    df = pd.read_csv(path, engine="pyarrow")
    frame = df.pop("Frame").to_numpy()
    if columns is not None:
        df = df[columns]
    df = df.astype(np.float32)
    df.index = pd.Index(frame / C.FS_IMU, name="t")
    return df


@lru_cache(maxsize=1)
def imu_lower_limb_angles() -> list[str]:
    cols = []
    for side in ("Right", "Left"):
        cols += [f"{side} Hip Flexion/Extension", f"{side} Hip Abduction/Adduction",
                 f"{side} Hip Internal/External Rotation",
                 f"{side} Knee Flexion/Extension", f"{side} Knee Abduction/Adduction",
                 f"{side} Ankle Dorsiflexion/Plantarflexion", f"{side} Ankle Internal/External Rotation"]
    return cols + ["L5S1 Flexion/Extension", "L5S1 Lateral Bending"]
