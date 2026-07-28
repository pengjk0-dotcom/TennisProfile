#!/usr/bin/env python3
"""Convert frame-level 4DHumans/HMR2 npz predictions to skeleton CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BODY25_JOINT_NAMES = [
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]

DEFAULT_KEEP_JOINTS = [
    "pelvis",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def resolve_prediction_path(value: str, root: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path


def prediction_to_rows(
    manifest_row: pd.Series,
    prediction: dict[str, np.ndarray],
    keep_joints: set[str] | None,
) -> list[dict[str, object]]:
    keypoints = prediction["pred_keypoints_3d"]
    if keypoints.shape[0] < len(BODY25_JOINT_NAMES):
        raise ValueError(f"Expected at least 25 joints, got {keypoints.shape}")

    detector_score = float(manifest_row.get("detector_score", prediction.get("detector_score", 1.0)))
    frame_index = int(manifest_row["frame_index"])
    trial_id = str(manifest_row["trial_id"])
    rows: list[dict[str, object]] = []
    for joint_index, joint_name in enumerate(BODY25_JOINT_NAMES):
        if keep_joints is not None and joint_name not in keep_joints:
            continue
        x, y, z = keypoints[joint_index]
        rows.append(
            {
                "frame": frame_index,
                "joint": joint_name,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "confidence": detector_score,
                "trial_id": trial_id,
                "subject_id": manifest_row.get("subject_id", ""),
                "skill_level": manifest_row.get("skill_level", ""),
                "stroke_type": manifest_row.get("stroke_type", ""),
                "action_type": manifest_row.get("action_type", ""),
                "timestamp_sec": manifest_row.get("timestamp_sec", ""),
                "person_rank": manifest_row.get("person_rank", 0),
                "source_joint_index": joint_index,
                "prediction_path": manifest_row.get("prediction_path", ""),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--all-body25", action="store_true", help="Keep all BODY_25 joints instead of the core set.")
    args = parser.parse_args()

    root = Path.cwd()
    predictions_manifest = pd.read_csv(args.predictions_manifest)
    ok_rows = predictions_manifest[predictions_manifest["status"] == "ok"].copy()
    if ok_rows.empty:
        raise ValueError("No ok prediction rows found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_joints = None if args.all_body25 else set(DEFAULT_KEEP_JOINTS)

    sequence_manifest_rows: list[dict[str, object]] = []
    for trial_id, trial_df in ok_rows.groupby("trial_id", sort=True):
        long_rows: list[dict[str, object]] = []
        trial_df = trial_df.sort_values(["frame_index", "person_rank"]).copy()
        for _, row in trial_df.iterrows():
            prediction_path = resolve_prediction_path(str(row["prediction_path"]), root)
            prediction = load_prediction(prediction_path)
            long_rows.extend(prediction_to_rows(row, prediction, keep_joints))

        out_path = out_dir / f"{trial_id}_skeleton.csv.gz"
        sequence = pd.DataFrame(long_rows)
        sequence.to_csv(out_path, index=False, compression="gzip")
        sequence_manifest_rows.append(
            {
                "trial_id": trial_id,
                "subject_id": trial_df.iloc[0].get("subject_id", ""),
                "skill_level": trial_df.iloc[0].get("skill_level", ""),
                "stroke_type": trial_df.iloc[0].get("stroke_type", ""),
                "action_type": trial_df.iloc[0].get("action_type", ""),
                "frames": int(trial_df["frame_index"].nunique()),
                "joints_per_frame": len(BODY25_JOINT_NAMES) if args.all_body25 else len(DEFAULT_KEEP_JOINTS),
                "sequence_path": str(out_path),
            }
        )
        print(f"{trial_id}: wrote {len(sequence)} rows to {out_path}")

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sequence_manifest_rows).to_csv(out_manifest, index=False)
    print(f"Wrote {len(sequence_manifest_rows)} sequence rows to {out_manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
