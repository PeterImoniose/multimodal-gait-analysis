"""Scan every trial of every modality and store the audit tables.

    python scripts/run_audit.py

Writes results/tables/audit_{eeg_trials,eeg_channels,emg,fp,imu}.parquet
"""

from __future__ import annotations

import time

import pandas as pd
from joblib import Parallel, delayed

from mgait import audit, config as C, io


def main() -> None:
    t0 = time.time()
    inv = io.inventory()
    inv.to_parquet(C.TABLES / "inventory.parquet")
    meta = io.load_subjects()
    trials = [(r.subject, r.speed) for r in inv.itertuples()]
    par = Parallel(n_jobs=-1, verbose=0)

    eeg = [r for r in par(delayed(audit.audit_eeg)(s, v) for s, v in trials) if r is not None]
    pd.DataFrame([r[0] for r in eeg]).to_parquet(C.TABLES / "audit_eeg_trials.parquet")
    pd.DataFrame([c for r in eeg for c in r[1]]).to_parquet(C.TABLES / "audit_eeg_channels.parquet")
    print(f"EEG {len(eeg)} trials  {time.time() - t0:.0f}s")

    emg = [r for r in par(delayed(audit.audit_emg)(s, v) for s, v in trials) if r is not None]
    pd.DataFrame(emg).to_parquet(C.TABLES / "audit_emg.parquet")
    print(f"EMG {len(emg)} trials  {time.time() - t0:.0f}s")

    fp = [r for r in par(delayed(audit.audit_fp)(s, v, meta.loc[s, "mass_kg"]) for s, v in trials) if r is not None]
    pd.DataFrame(fp).to_parquet(C.TABLES / "audit_fp.parquet")
    print(f"FP  {len(fp)} trials  {time.time() - t0:.0f}s")

    imu = [r for r in par(delayed(audit.audit_imu)(s, v) for s, v in trials) if r is not None]
    pd.DataFrame(imu).to_parquet(C.TABLES / "audit_imu.parquet")
    print(f"IMU {len(imu)} trials  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
