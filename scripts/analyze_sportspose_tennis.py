#!/usr/bin/env python3
"""Analyze the downloaded SportsPose tennis subset.

The script treats SportsPose as an external 3D-pose source rather than as a
skill-label dataset. It builds a manifest and extracts the same body-level
motion cues used by the paper where the joint definitions are available.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


COCO17 = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

OUTDOOR_VIEW_TO_CAMERA = {
    "right": "CAM3.avi",
    "front_left": "CAM4.avi",
    "back": "CAM2.avi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/external/sportspose/SportsPose"),
        help="Root directory of the downloaded SportsPose subset.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/sportspose_tennis_external"),
        help="Directory for manifest and feature tables.",
    )
    parser.add_argument(
        "--paper-results-dir",
        type=Path,
        default=Path("paper/results"),
        help="Directory for compact paper-facing summaries.",
    )
    parser.add_argument("--fps", type=float, default=90.0)
    parser.add_argument(
        "--half-window-frames",
        type=int,
        default=45,
        help="Frames before and after the wrist-speed anchor for feature extraction.",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def iter_tennis_trials(dataset_root: Path) -> Iterable[Path]:
    data_root = dataset_root / "data" / "outdoors"
    yield from sorted(data_root.glob("S*/tennis/tennis*.npy"))


def infer_trial_id(path: Path) -> str:
    subject_id = path.parents[1].name
    return f"sportspose_outdoors_{subject_id}_{path.stem}"


def nan_interpolate(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected a one-dimensional signal.")
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if finite.sum() == 0:
        return np.zeros_like(arr)
    idx = np.arange(arr.size)
    return np.interp(idx, idx[finite], arr[finite])


def smooth_signal(values: np.ndarray, fps: float, cutoff_hz: float = 6.0) -> np.ndarray:
    arr = nan_interpolate(values)
    if arr.size < 5:
        return arr
    nyquist = fps / 2.0
    cutoff = min(cutoff_hz / nyquist, 0.95)
    order = 4
    padlen = 3 * (order + 1)
    while order > 1 and arr.size <= padlen:
        order -= 1
        padlen = 3 * (order + 1)
    if arr.size <= padlen:
        return arr
    b, a = butter(order, cutoff, btype="low")
    return filtfilt(b, a, arr)


def horizontal_line_angle(
    sequence: np.ndarray,
    left_joint: str,
    right_joint: str,
) -> np.ndarray:
    """Return unwrapped line angle in the SportsPose horizontal x-y plane.

    SportsPose outdoor 3D poses use the third coordinate as the vertical axis in
    the downloaded subset, so pelvis and shoulder rotation are projected to x-y.
    """

    left = sequence[:, COCO17[left_joint], :]
    right = sequence[:, COCO17[right_joint], :]
    vec = right - left
    angle = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
    return np.degrees(np.unwrap(np.radians(nan_interpolate(angle))))


def derivative(values: np.ndarray, fps: float) -> np.ndarray:
    if values.size < 2:
        return np.zeros_like(values)
    return np.gradient(values) * fps


def joint_speed(sequence: np.ndarray, joint: str, fps: float) -> np.ndarray:
    coords = sequence[:, COCO17[joint], :]
    coords = np.apply_along_axis(nan_interpolate, 0, coords)
    if coords.shape[0] < 2:
        return np.zeros(coords.shape[0])
    diff = np.diff(coords, axis=0, prepend=coords[0:1])
    return np.linalg.norm(diff, axis=1) * fps


def safe_peak_abs(values: np.ndarray) -> Tuple[int, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return 0, float("nan")
    idx = int(np.nanargmax(np.abs(arr)))
    return idx, float(arr[idx])


def range_deg(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))


def build_manifest_row(
    dataset_root: Path,
    pose_path: Path,
    timing: Dict[str, Any],
    fps: float,
) -> Dict[str, Any]:
    rel_pose = pose_path.relative_to(dataset_root)
    subject_id = pose_path.parents[1].name
    arr = np.load(pose_path, mmap_mode="r")
    video_path_tokens = timing.get("video_path", [])
    if isinstance(video_path_tokens, np.ndarray):
        video_path_tokens = video_path_tokens.tolist()
    video_rel_dir = Path("videos").joinpath(*map(str, video_path_tokens))
    camera_paths = {
        f"{view}_video_rel_path": str(video_rel_dir / camera)
        for view, camera in OUTDOOR_VIEW_TO_CAMERA.items()
    }
    return {
        "dataset_id": "SportsPose",
        "environment": "outdoors",
        "subject_id": subject_id,
        "activity": "tennis",
        "trial_id": infer_trial_id(pose_path),
        "pose_rel_path": str(rel_pose),
        "timing_rel_path": str(pose_path.with_name(f"{pose_path.stem}_timing.pkl").relative_to(dataset_root)),
        "calib_rel_path": str((pose_path.parents[1] / "calib.pkl").relative_to(dataset_root)),
        "fps": fps,
        "n_frames": int(arr.shape[0]),
        "n_joints": int(arr.shape[1]),
        "coordinate_vertical_axis": "z",
        "video_folder": "/".join(map(str, video_path_tokens)),
        "has_local_right_video": (dataset_root / camera_paths["right_video_rel_path"]).exists(),
        **camera_paths,
    }


def compute_trial_features(
    pose_path: Path,
    timing: Dict[str, Any],
    fps: float,
    half_window_frames: int,
) -> Dict[str, Any]:
    sequence = np.load(pose_path)
    n_frames = int(sequence.shape[0])

    left_wrist_speed = smooth_signal(joint_speed(sequence, "left_wrist", fps), fps)
    right_wrist_speed = smooth_signal(joint_speed(sequence, "right_wrist", fps), fps)
    left_peak = float(np.nanmax(left_wrist_speed))
    right_peak = float(np.nanmax(right_wrist_speed))
    active_side = "left" if left_peak >= right_peak else "right"
    active_speed = left_wrist_speed if active_side == "left" else right_wrist_speed
    anchor_frame = int(np.nanargmax(active_speed))

    start_frame = max(0, anchor_frame - half_window_frames)
    end_frame = min(n_frames - 1, anchor_frame + half_window_frames)
    window = sequence[start_frame : end_frame + 1]
    window_n = int(window.shape[0])

    pelvis_angle = smooth_signal(
        horizontal_line_angle(window, "left_hip", "right_hip"), fps
    )
    shoulder_angle = smooth_signal(
        horizontal_line_angle(window, "left_shoulder", "right_shoulder"), fps
    )
    shoulder_vs_pelvis = nan_interpolate(shoulder_angle - pelvis_angle)

    pelvis_velocity = derivative(pelvis_angle, fps)
    shoulder_velocity = derivative(shoulder_angle, fps)

    active_wrist_joint = f"{active_side}_wrist"
    active_wrist_speed_window = smooth_signal(joint_speed(window, active_wrist_joint, fps), fps)

    pelvis_peak_idx, pelvis_peak_velocity = safe_peak_abs(pelvis_velocity)
    shoulder_peak_idx, shoulder_peak_velocity = safe_peak_abs(shoulder_velocity)
    wrist_peak_idx, wrist_peak_speed = safe_peak_abs(active_wrist_speed_window)

    anchor_in_window = anchor_frame - start_frame
    video_path_tokens = timing.get("video_path", [])
    if isinstance(video_path_tokens, np.ndarray):
        video_path_tokens = video_path_tokens.tolist()

    return {
        "dataset_id": "SportsPose",
        "environment": "outdoors",
        "subject_id": pose_path.parents[1].name,
        "activity": "tennis",
        "trial_id": infer_trial_id(pose_path),
        "pose_file": pose_path.name,
        "fps": fps,
        "n_frames": n_frames,
        "window_start_frame": int(start_frame),
        "anchor_frame": int(anchor_frame),
        "window_end_frame": int(end_frame),
        "window_n_frames": window_n,
        "anchor_definition": "maximum active-wrist 3D speed",
        "active_wrist_side": active_side,
        "anchor_near_edge": bool(start_frame == 0 or end_frame == n_frames - 1),
        "video_folder": "/".join(map(str, video_path_tokens)),
        "pelvis_rotation_range_deg": round(range_deg(pelvis_angle), 4),
        "shoulder_rotation_range_deg": round(range_deg(shoulder_angle), 4),
        "shoulder_vs_pelvis_rotation_range_deg": round(range_deg(shoulder_vs_pelvis), 4),
        "pelvis_peak_abs_velocity_deg_s": round(abs(pelvis_peak_velocity), 4),
        "shoulder_peak_abs_velocity_deg_s": round(abs(shoulder_peak_velocity), 4),
        "active_wrist_peak_speed_units_s": round(abs(wrist_peak_speed), 4),
        "pelvis_fastest_frame_relative_to_anchor": int(pelvis_peak_idx - anchor_in_window),
        "shoulder_fastest_frame_relative_to_anchor": int(shoulder_peak_idx - anchor_in_window),
        "wrist_fastest_frame_relative_to_anchor": int(wrist_peak_idx - anchor_in_window),
        "pelvis_fastest_ms_relative_to_anchor": round(
            (pelvis_peak_idx - anchor_in_window) * 1000.0 / fps, 2
        ),
        "shoulder_fastest_ms_relative_to_anchor": round(
            (shoulder_peak_idx - anchor_in_window) * 1000.0 / fps, 2
        ),
        "wrist_fastest_ms_relative_to_anchor": round(
            (wrist_peak_idx - anchor_in_window) * 1000.0 / fps, 2
        ),
        "shoulder_after_pelvis_fastest_ms": round(
            (shoulder_peak_idx - pelvis_peak_idx) * 1000.0 / fps, 2
        ),
        "all_features_finite": bool(
            np.isfinite(
                [
                    range_deg(pelvis_angle),
                    range_deg(shoulder_angle),
                    range_deg(shoulder_vs_pelvis),
                    abs(pelvis_peak_velocity),
                    abs(shoulder_peak_velocity),
                    abs(wrist_peak_speed),
                ]
            ).all()
        ),
    }


def summarize(features: pd.DataFrame, manifest: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = [
        "pelvis_rotation_range_deg",
        "shoulder_rotation_range_deg",
        "shoulder_vs_pelvis_rotation_range_deg",
        "pelvis_peak_abs_velocity_deg_s",
        "shoulder_peak_abs_velocity_deg_s",
        "active_wrist_peak_speed_units_s",
        "pelvis_fastest_ms_relative_to_anchor",
        "shoulder_fastest_ms_relative_to_anchor",
        "shoulder_after_pelvis_fastest_ms",
    ]
    numeric_summary: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        values = features[col].dropna()
        numeric_summary[col] = {
            "median": round(float(values.median()), 3),
            "iqr": round(float(values.quantile(0.75) - values.quantile(0.25)), 3),
            "min": round(float(values.min()), 3),
            "max": round(float(values.max()), 3),
        }

    return {
        "dataset": "SportsPose",
        "subset": "outdoor tennis",
        "n_sequences": int(len(features)),
        "n_subjects": int(features["subject_id"].nunique()),
        "subjects": sorted(features["subject_id"].unique().tolist()),
        "fps": float(features["fps"].iloc[0]) if len(features) else None,
        "frames_per_sequence": sorted(features["n_frames"].unique().tolist()),
        "all_sequences_have_17_joints": bool((manifest["n_joints"] == 17).all()),
        "all_features_finite": bool(features["all_features_finite"].all()),
        "anchors_not_near_edge_count": int((~features["anchor_near_edge"]).sum()),
        "anchors_not_near_edge_percent": round(float((~features["anchor_near_edge"]).mean() * 100), 1),
        "active_wrist_counts": features["active_wrist_side"].value_counts().to_dict(),
        "local_video_files_present": int(manifest["has_local_right_video"].sum()),
        "numeric_summary": numeric_summary,
        "interpretation": (
            "This subset is used only as an external 3D-pose check. It has no "
            "skill labels or reviewed ball-contact labels in this analysis."
        ),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.paper_results_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    for pose_path in iter_tennis_trials(args.dataset_root):
        timing_path = pose_path.with_name(f"{pose_path.stem}_timing.pkl")
        if not timing_path.exists():
            raise FileNotFoundError(f"Missing timing file for {pose_path}")
        timing = load_pickle(timing_path)
        manifest_rows.append(build_manifest_row(args.dataset_root, pose_path, timing, args.fps))
        feature_rows.append(
            compute_trial_features(pose_path, timing, args.fps, args.half_window_frames)
        )

    manifest = pd.DataFrame(manifest_rows)
    features = pd.DataFrame(feature_rows)
    summary = summarize(features, manifest)

    manifest_path = args.out_dir / "sportspose_tennis_manifest.csv"
    features_path = args.out_dir / "sportspose_tennis_features.csv"
    summary_json_path = args.paper_results_dir / "sportspose_external_validation_summary.json"
    summary_csv_path = args.paper_results_dir / "sportspose_external_validation_summary.csv"

    manifest.to_csv(manifest_path, index=False)
    features.to_csv(features_path, index=False)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_rows = []
    for metric, values in summary["numeric_summary"].items():
        summary_rows.append({"metric": metric, **values})
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote features: {features_path}")
    print(f"Wrote summary: {summary_json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
