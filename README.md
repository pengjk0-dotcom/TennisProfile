# TennisProfile

Official reproducibility materials for:

**TennisProfile: Reliability-Aware Groundstroke Profiling from Monocular 3D Human Recovery**

Accepted at the MoCha workshop at ECCV 2026.

TennisProfile converts monocular human-recovery outputs into stroke-specific,
expert-referenced tennis motion profiles. The release contains the reviewed
event table, fixed study split, derived features, statistical and robustness
results, figure assets, and the scripts used for the paper-facing analyses.

## Repository Scope

This repository is intentionally limited to material that can be redistributed:

- reviewed `start`, `contact`, and `end` event annotations;
- the fixed subject and trial split;
- derived feature tables and aggregate results;
- cohort, viewpoint, and racket-hand audit tables;
- analysis, robustness, and figure-generation scripts;
- the recorded HMR2 and analysis environments.

Raw THETIS or SportsPose media, SMPL assets, HMR2 checkpoints, and other
third-party resources are not included. See
[`THIRD_PARTY_ASSETS.md`](THIRD_PARTY_ASSETS.md) for acquisition and licensing
boundaries.

## Quick Verification

The integrity check uses only the Python standard library:

```bash
python3 scripts/verify_release.py
```

It verifies the 36-subject, 72-stroke cohort, event ordering, table alignment,
finite paper-facing features, and the absence of redistributed raw media or
model weights.

File-level SHA-256 checksums for the recorded release are listed in
`MANIFEST.sha256`.

For the statistical analyses:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r reproducibility/requirements.txt
python scripts/run_camera_ready_robustness.py --skip-sportspose
```

The last command regenerates the analyses that depend only on the released
feature and split tables. To recompute the SportsPose frame-rate check, set
`SPORTSPOSE_ROOT` to a licensed SportsPose checkout and omit
`--skip-sportspose`.

## Contents

- `data/`: final split, event annotations, derived features, reference
  summaries, and group summaries.
- `results/`: statistical tests, perturbation checks, exhaustive expert
  holdouts, phase maps, and cross-source summaries.
- `figures/`: paper and supplementary figures generated from the recorded run.
- `scripts/`: feature extraction, analysis, robustness, HMR2 conversion, and
  figure-generation code.
- `reproducibility/`: dependency pins, runtime record, cohort audit, and the
  Apple MPS compatibility patch used for HMR2 inference.

## Reproduction Levels

**Released-table reproduction.** The quick verification and camera-ready
robustness scripts run from the included annotations, split, and derived
features.

**Feature recomputation.** Recomputing body-line and wrist features requires the
licensed source videos and the recorded HMR2 skeleton outputs. The expected
event and feature schemas are represented by `data/final_events.csv` and
`data/final_all_assigned_features.csv`.

**Full HMR2 inference.** The recorded commit, checkpoint hash, runtime, and
compatibility patch are documented in
`reproducibility/HMR2_RUNTIME.md`. Users must obtain the HMR2 checkpoint and
SMPL assets from their original providers.

## Main Analysis Cohort

The fixed release contains 18 expert-reference subjects, 3 held-out expert
subjects, and 15 beginner subjects. Each subject contributes one flat forehand
and one one-handed backhand, yielding 72 reviewed strokes. Cohort membership was
fixed independently of the final feature values; the selection record is in
`reproducibility/subject_selection_audit.csv`.

## Licensing

Code written for TennisProfile is released under the MIT License. Original
annotations and derived tabular outputs are released under CC BY 4.0, to the
extent that the authors hold rights in those materials. No license is granted
for third-party videos, body models, checkpoints, or software.

Citation metadata will be added once the camera-ready proceedings record is
available.
