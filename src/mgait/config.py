"""Paths and fixed constants for the multimodal gait analysis.

The raw dataset is read in place and never modified. Its location defaults to
the folder next to this project and can be overridden with the MGAIT_RAW
environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_RAW = Path(
    "D:/files/multimodal-gait-dataset-1.0.0/"
    "a-multimodal-gait-dataset-of-brain-activity-muscle-activity-kinematics-and-ground-forces-in-young-adults-1.0.0/"
    "Dataset"
)
RAW = Path(os.environ.get("MGAIT_RAW", _DEFAULT_RAW))
EEG_DIR = RAW / "EEG_Data" / "EEG.csv"
EMG_DIR = RAW / "EMG_Data"
FP_DIR = RAW / "FP_Data"
SUBJECT_FILE = RAW / "subject-info.csv"

# IMU archive is extracted once into the project (git-ignored)
IMU_DIR = Path(os.environ.get("MGAIT_IMU", ROOT / "data" / "imu"))
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

SUBJECTS = [f"S{i}" for i in range(1, 60)]
SPEEDS = ["0.5", "0.75", "1"]            # file-name labels
SPEED_MS = {"0.5": 0.5, "0.75": 0.75, "1": 1.0}

# Nominal sampling rates (Hz)
FS_EEG = 300.0
FS_EMG = 1.0 / 0.0623076923              # Delsys MAV output, about 16.05 Hz
FS_FP = 1000.0
FS_IMU = 60.0

# Development subjects: the only trials used to choose processing settings
# (IMU contact threshold). Pilot sessions S1-S3 used a different protocol.
PILOT_SUBJECTS = ["S1", "S2", "S3"]
DEV_SUBJECTS = [f"S{i}" for i in range(4, 14)]
IMU_THR_FRAC = 0.30   # chosen on DEV_SUBJECTS: lowest mean |HS|+|TO| timing error vs force plates

G = 9.81
SEED = 42

EEG_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
                "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]

MUSCLES = ["TA", "VL", "RF", "GM", "ST", "GL"]
MUSCLE_NAMES = {
    "TA": "Tibialis anterior", "VL": "Vastus lateralis", "RF": "Rectus femoris",
    "GM": "Gastrocnemius medialis", "ST": "Semitendinosus", "GL": "Gastrocnemius lateralis",
}

for _p in (CACHE, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)
