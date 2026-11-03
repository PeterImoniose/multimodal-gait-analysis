# Multimodal Gait Analysis: Sensor Validation, Biomechanics and Data Governance

An end-to-end analysis of a public multimodal walking dataset (59 adults; EEG, EMG, full-body IMU and an
instrumented split-belt treadmill; three speeds) to answer a deployment question: **which sensing
technologies carry reliable, generalisable gait information, and which do not?**

The work covers a full data audit and correction of the release, a reference biomechanical analysis, a
validation of the wearable IMU suit against force plates on held-out participants, muscle-synergy analysis, a
four-way test of whether the walking EEG is brain signal or motion artefact, and participant-wise machine
learning including a re-identification (privacy) test. A board-level presentation summarises the decisions
the evidence supports.

## Key findings

| # | Finding | Evidence |
|---|---|---|
| 1 | **The release is usable only after correction.** 20 issues were found, including swapped and unzeroed force plates, a mislabelled body mass, a corrupted EEG montage, EDF files offset by 133 ms from the CSVs, and released dates of birth. All 173 force-plate trials were recovered. | Notebook 01, `results/tables/data_issues.csv` |
| 2 | **The IMU suit reproduces force-plate timing and walking speed.** Stride time ICC 0.999 (limits +/-12 ms); 99 % of gait events detected; suit-estimated speed within 4.3 % of belt speed. Stance percentage agrees less tightly (ICC 0.77). | Notebook 03 |
| 3 | **A "virtual force plate" works for loading summaries, not waveforms.** IMU features predict braking, propulsion, stance and double support on unseen participants with R2 0.85 to 0.92, but stride-level force waveforms are only 6 to 10 % better than a speed-matched template, and the impact peak is not recoverable. | Notebook 03 |
| 4 | **The walking EEG is dominated by motion artefact.** Step-locked spectral peaks in 94 % of trials, doubling per +0.5 m/s; coherence with the participant's own head motion four times that with other participants'; 9 of 11 apparent EEG speed effects vanish after artefact adjustment; two artefact indices predict speed better (62 %) than 30 EEG band-power features (49 %). | Notebooks 05, 06 |
| 5 | **Muscle coordination is stable across people and speeds.** Four shared synergies (weight acceptance, push-off, swing deceleration, foot clearance) match individual synergies with median cosine similarity 0.98. | Notebook 04 |
| 6 | **Every modality is a biometric identifier.** Participants are re-identified from a trial at an unseen speed with 87 to 93 % accuracy (chance 2 %); sex is inferred from gait with 88 % accuracy (baseline 62 %). | Notebook 06 |
| 7 | **Validation protocol changes the answer.** A random stride-level split overstates random-forest accuracy by 13 points (98 % vs 85 %) relative to participant-wise validation. | Notebook 06 |

Reference biomechanics (notebook 02): between 0.5 and 1.0 m/s cadence rises 32 % and stride length 50 %,
braking and propulsive forces and the vertical loading rate roughly double, and stride-time variability falls
from 3.3 % to 1.9 %. Speed is gained mainly through longer strides (59 % of the increase). Adjusted for speed
and sex, higher BMI is associated with longer stance, more double support and a lower body-weight-normalised
loading rate.

## Decision summary

| Modality | Supports | Does not support | Recommendation |
|---|---|---|---|
| Force plates | Reference timing and kinetics | Portability | Retain as the calibration reference |
| IMU suit | Gait events, stride time, speed, loading summaries | Individual force waveforms, impact peak | Primary platform for scalable assessment |
| EMG (envelope) | Muscle timing, amplitude scaling, synergies | Stand-alone classification; spectral analysis (raw signal not released) | Optional module; capture raw EMG |
| Dry EEG while walking | Nothing beyond its artefact content in this release | Cortical inference | Defer; gate on validated mobile-EEG artefact removal |

## Repository structure

```
src/mgait/            analysis package
  io.py               loaders for every modality, with release-specific fixes
  audit.py            per-trial quality checks
  gait.py             force-plate and IMU gait events, strides, features, alignment
  emg.py              gait-cycle EMG and NMF synergies
  eeg.py              spectra, step-harmonic and head-coherence artefact indices
  features.py         trial-level feature table
  stats.py            mixed models, Holm, cluster permutation test, ICC, Bland-Altman
  models.py           participant-wise model evaluation
  plotting.py         chart style
scripts/              run_audit.py, build_cache.py, build_notebooks.py
notebooks/            01-06 executed notebooks (sources in notebooks/source)
results/              figures/ and tables/ produced by the notebooks
tests/                unit tests on synthetic signals
```

## Board presentation

A 16-slide board paper, *Gait sensing: what to deploy, and what to defer*, turns the findings into four
decisions (adopt the IMU suit as the primary platform, retain force plates as the reference, defer walking EEG
behind a validation gate, govern gait data as biometric personal data), with a phased plan whose gates are
set from this study's results, a risk register and a method appendix. Every figure and number on the slides
is produced by the notebooks in this repository.

## Notebooks

1. `01_data_audit` - coverage, metadata integrity, per-modality quality, synchronisation, issue register
2. `02_gait_biomechanics` - force-plate events, speed effects (mixed models), GRF waveforms (cluster permutation tests), BMI
3. `03_wearable_validation` - IMU vs force plates on held-out participants; speed estimate; kinematics; virtual force plate
4. `04_muscle_activity` - activation profiles, speed scaling, muscle synergies
5. `05_eeg_signal_integrity` - step-locked spectra, head-motion coherence vs surrogate null, heel-strike transients, artefact-adjusted effects
6. `06_machine_learning` - speed classification by modality, leakage demonstration, EEG artefact test, re-identification, attribute inference

## Methods principles

* **No tuning on evaluation data.** The only tuned processing setting (IMU contact threshold) was chosen on ten
  development participants and fixed in `config.py` before held-out evaluation.
* **Participants are the unit of validation.** Every model is cross-validated with participants held out as
  whole groups; intervals resample participants.
* **Repeated measures are modelled.** Speed effects use linear mixed models with a random intercept per
  participant; families of tests are Holm-adjusted.
* **Corrections are evidenced, not assumed.** Each data fix is supported by an independent check (for example,
  the S5 plate swap by both centre-of-pressure geometry and IMU contact alignment).
* **Negative results are reported.** The EEG and waveform-prediction results are presented with the same weight
  as the positive ones.

## Reproduce

Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
```

Obtain the dataset and set the paths as described in [DATASET.md](DATASET.md), then:

```bash
.venv\Scripts\python scripts\run_audit.py
.venv\Scripts\python scripts\build_cache.py
.venv\Scripts\python scripts\build_notebooks.py
.venv\Scripts\python -m pytest
```

The audit and cache take under a minute on a 16-core machine; executing all notebooks takes about five minutes.

## Limitations

* Healthy young adults (median age 22) on a treadmill at slow to comfortable speeds; results do not transfer
  to older, clinical or overground populations without re-validation.
* One minute per condition and a single session: re-identification is shown across speeds, not across days.
* EMG is available only as a 16 Hz envelope; EEG uses a dry headset with no motion reference channels.
* Force-plate and IMU clocks were aligned post hoc; 11 of 168 trials could not be aligned within 0.1 s and are
  excluded from event-level comparisons.

## Data and licence

Data: Katmah et al., *A Multimodal Gait Dataset of Brain Activity, Muscle Activity, Kinematics and Ground Forces
in Young Adults*, v1.0.0, CC BY 4.0. The data are not redistributed here; see [DATASET.md](DATASET.md).
