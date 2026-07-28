# HMR2 Inference Runtime

This record fixes the runtime used to produce the paper-facing HMR2 outputs.
It contains no source videos, SMPL model files, or model weights.

## Base implementation

- 4D-Humans base commit: `efe18deff163b29dff87ddbd575fa29b716a356c`
- checkpoint filename: `epoch=35-step=1000000.ckpt`
- checkpoint SHA-256: `c9ea09c834161468638c9de7a782c614e300c622709bf52967274c543ec72fcf`
- local compatibility patch: `reproducibility/4dhumans_mps_compatibility.patch`

The compatibility patch enables device selection and macOS rendering support. It
does not modify model weights or feature definitions.

## Inference environment

- Python 3.10.20
- PyTorch 2.12.0
- torchvision 0.27.0
- PyTorch Lightning 2.6.1
- NumPy 2.2.6
- pandas 2.3.3
- SciPy 1.15.3
- OpenCV 4.13.0
- device used for the recorded inference: Apple MPS

## Analysis environment

The separate feature-extraction and figure environment is pinned in
`requirements.txt`. It uses Python 3.14.5 in the recorded run.

## Reproduction sequence

1. Obtain the HMR2 checkpoint and SMPL model from their original providers.
2. Check out the base 4D-Humans commit and apply the included compatibility
   patch when needed for the recorded macOS environment.
3. Run `scripts/run_4dhumans_frame_inference.py` on an extracted-frame manifest.
4. Convert predictions with `scripts/convert_4dhumans_npz_to_skeleton_csv.py`.
5. Use `data/final_events.csv` to define the reviewed stroke windows and run
   `scripts/extract_kinetic_chain_features.py` on the converted skeleton
   manifest.
6. Install the pinned analysis requirements, then run
   `scripts/run_camera_ready_robustness.py`. The released-table path can be
   checked without licensed SportsPose files by adding `--skip-sportspose`.

The paper release includes derived event tables and aggregate features. Raw
videos, SMPL assets, and model weights remain subject to their original terms.
