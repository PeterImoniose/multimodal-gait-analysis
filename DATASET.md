# Data card

## Source

Katmah R., AlShehhi A., Kosaji D., Al-Rahmani N., Abdullah M., Hulleck A. A., Khalaf K.
*A Multimodal Gait Dataset of Brain Activity, Muscle Activity, Kinematics and Ground Forces in Young Adults*,
version 1.0.0 (PhysioNet package `multimodal-gait-dataset-1.0.0`). Khalifa University, Abu Dhabi.
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).

The raw data are not redistributed in this repository. Download the package, extract it, and point the code at
its `Dataset` folder (see "Setup" below).

## Contents

| Item | Detail |
|---|---|
| Participants | 59 healthy adults (37 female, 22 male), median age 22, BMI 18 to 38 kg/m2 |
| Protocol | Split-belt treadmill, 0.50, 0.75 and 1.00 m/s; 60 s recorded per speed after 60 s of adaptation |
| EEG | DSI-24 dry headset, 19 channels (10-20), 300 Hz, recorded against Pz; CSV and EDF |
| EMG | Delsys Trigno, 12 muscles (tibialis anterior, vastus lateralis, rectus femoris, gastrocnemius medialis and lateralis, semitendinosus; both legs); released as a 16 Hz mean-absolute-value envelope |
| IMU | Xsens MVN Awinda, 17 sensors, 60 Hz; 18 export files per trial (joint angles, segment kinematics, centre of mass) |
| Force plates | Bertec split-belt instrumented treadmill, 1000 Hz, forces, moments and centre of pressure per belt |
| Metadata | `subject-info.csv`: sex, date of birth, age, height, mass, BMI, segment lengths and circumferences, preferred overground speed |

Synchronisation: EEG, EMG and IMU share a wireless trigger network; the treadmill was started by hand.

## Known issues found in this project

The full register with treatment and status is produced by notebook 01 (`results/tables/data_issues.csv`).
The most consequential:

| Area | Issue | Treatment |
|---|---|---|
| Force plates | One plate unzeroed by -70 to -270 N in S4, S5, S7, S21, S57 (S21 appears empty) | Swing-phase re-zeroing per plate |
| Force plates | Left/right plates swapped for S5 | Relabelled; confirmed by COP side and IMU contact alignment |
| Force plates | Timestamp, sync and aux columns all zero; S2 files lack a header | Time from sample index; standard columns assigned |
| Metadata | Body mass for S24 is 92.4 kg but the plates measure 110 kg at every speed | Measured body weight used for normalisation |
| Metadata | Age column reflects about 1 January 2025, not the 2023-2024 test dates | Used descriptively only |
| Metadata | Full dates of birth are included despite the anonymisation statement | Used only to verify age; never written to outputs |
| EEG | S32_1 has a corrupted montage header (CSV and EDF) | Excluded |
| EEG | S8_0.5 contains a 15 s timing gap | Excluded from spectral analysis |
| EEG | EDF files are the CSV data shifted by 40 samples (133 ms) and truncated to 58 s | CSV used |
| EMG | 55 channel-trials are noise-dominated (median envelope above 5x the muscle norm) | Channel set to missing |
| IMU | Empty placeholder files for S46_1 and S47; right-foot stream fails in S53_1 | Intact files only; S53_1 has no IMU strides |
| Sync | 11 of 168 trials have force plate to IMU offsets above 0.1 s (manual treadmill start) | Excluded from event-level comparisons |

## Analysis cohorts

| Analysis | Trials | Participants |
|---|---|---|
| Force-plate gait analysis | 173 | 59 |
| IMU kinematics | 169 | 58 |
| EMG gait cycles | 168 | 58 |
| EEG spectral analysis | 175 | 59 |
| Force plate vs IMU event agreement (aligned trials) | 157 | 55 |

## Setup

1. Extract the package so that `.../Dataset/EEG_Data`, `EMG_Data`, `FP_Data`, `IMU_Data` and `subject-info.csv` exist.
2. Set `MGAIT_RAW` to that `Dataset` folder (default: `D:/files/multimodal-gait-dataset-1.0.0/.../Dataset`).
3. Extract `IMU_Data/IMU.tar.gz` into `data/imu/` in this repository (about 9 GB), or set `MGAIT_IMU` to its location.

## Ethics and privacy

The study was approved by the Khalifa University Institutional Review Board (H19-038). This project treats the
data as personal data: dates of birth are not propagated, and notebook 06 shows that every modality supports
re-identification of participants across walking conditions (87 to 93 % at chance 2 %).
