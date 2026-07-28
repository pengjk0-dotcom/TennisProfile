#!/usr/bin/env python3
"""Compare HMR2 features with SportsPose 3D-pose features on matched video clips."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


COCO17 = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}

FEATURE_COLUMNS = [
    "pelvis_rotation_range_deg",
    "shoulder_rotation_range_deg",
    "shoulder_vs_pelvis_rotation_range_deg",
    "pelvis_peak_abs_velocity_deg_s",
    "shoulder_peak_abs_velocity_deg_s",
    "active_wrist_peak_speed_units_s",
    "pelvis_fastest_ms_relative_to_anchor",
    "shoulder_fastest_ms_relative_to_anchor",
    "wrist_fastest_ms_relative_to_anchor",
    "shoulder_after_pelvis_fastest_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sports-root",
        type=Path,
        default=Path("data/external/sportspose/SportsPose"),
    )
    parser.add_argument(
        "--sportspose-features",
        type=Path,
        default=Path("data/processed/sportspose_tennis_external/sportspose_tennis_features.csv"),
    )
    parser.add_argument(
        "--sportspose-manifest",
        type=Path,
        default=Path("data/processed/sportspose_tennis_external/sportspose_tennis_manifest.csv"),
    )
    parser.add_argument(
        "--hmr2-skeleton-manifest",
        type=Path,
        default=Path("data/processed/sportspose_hmr2_validation/hmr2_skeleton_manifest.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/sportspose_hmr2_validation"),
    )
    parser.add_argument(
        "--paper-results-dir",
        type=Path,
        default=Path("paper/results"),
    )
    return parser.parse_args()


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def nan_interpolate(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if finite.sum() == 0:
        return np.zeros_like(arr)
    idx = np.arange(arr.size)
    return np.interp(idx, idx[finite], arr[finite])


def smooth_signal(values: np.ndarray, fps: float, cutoff_hz: float = 6.0) -> np.ndarray:
    arr = nan_interpolate(np.asarray(values, dtype=float))
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


def derivative(values: np.ndarray, fps: float) -> np.ndarray:
    if values.size < 2:
        return np.zeros_like(values)
    return np.gradient(values) * fps


def range_deg(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))


def safe_peak_abs(values: np.ndarray) -> tuple[int, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return 0, float("nan")
    idx = int(np.nanargmax(np.abs(arr)))
    return idx, float(arr[idx])


def sports_pose_line_angle(sequence: np.ndarray, left_joint: str, right_joint: str) -> np.ndarray:
    left = sequence[:, COCO17[left_joint], :]
    right = sequence[:, COCO17[right_joint], :]
    vec = right - left
    angle = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
    return np.degrees(np.unwrap(np.radians(nan_interpolate(angle))))


def sports_pose_joint_speed(sequence: np.ndarray, joint: str, fps: float) -> np.ndarray:
    coords = sequence[:, COCO17[joint], :]
    coords = np.apply_along_axis(nan_interpolate, 0, coords)
    diff = np.diff(coords, axis=0, prepend=coords[0:1])
    return np.linalg.norm(diff, axis=1) * fps


def hmr2_to_sequence(skeleton_path: Path) -> tuple[np.ndarray, list[int], list[str]]:
    df = pd.read_csv(skeleton_path)
    frames = sorted(df["frame"].unique().tolist())
    joints = sorted(df["joint"].unique().tolist())
    frame_index = {frame: i for i, frame in enumerate(frames)}
    joint_index = {joint: i for i, joint in enumerate(joints)}
    sequence = np.full((len(frames), len(joints), 3), np.nan, dtype=float)
    for row in df.itertuples(index=False):
        sequence[
            frame_index[int(row.frame)],
            joint_index[str(row.joint)],
            :,
        ] = [float(row.x), float(row.y), float(row.z)]
    return sequence, frames, joints


def hmr2_line_angle(sequence: np.ndarray, joints: list[str], left_joint: str, right_joint: str) -> np.ndarray:
    joint_index = {joint: i for i, joint in enumerate(joints)}
    left = sequence[:, joint_index[left_joint], :]
    right = sequence[:, joint_index[right_joint], :]
    vec = right - left
    # HMR2 body joints use y as the vertical axis; rotation is measured in x-z.
    angle = np.degrees(np.arctan2(vec[:, 2], vec[:, 0]))
    return np.degrees(np.unwrap(np.radians(nan_interpolate(angle))))


def hmr2_joint_speed(sequence: np.ndarray, joints: list[str], joint: str, fps: float) -> np.ndarray:
    joint_index = {name: i for i, name in enumerate(joints)}
    coords = sequence[:, joint_index[joint], :]
    coords = np.apply_along_axis(nan_interpolate, 0, coords)
    diff = np.diff(coords, axis=0, prepend=coords[0:1])
    return np.linalg.norm(diff, axis=1) * fps


def compute_features_from_signals(
    pelvis_angle: np.ndarray,
    shoulder_angle: np.ndarray,
    wrist_speed: np.ndarray,
    fps: float,
    anchor_in_window: int,
) -> dict[str, float]:
    pelvis_angle = smooth_signal(pelvis_angle, fps)
    shoulder_angle = smooth_signal(shoulder_angle, fps)
    shoulder_vs_pelvis = nan_interpolate(shoulder_angle - pelvis_angle)
    wrist_speed = smooth_signal(wrist_speed, fps)

    pelvis_velocity = derivative(pelvis_angle, fps)
    shoulder_velocity = derivative(shoulder_angle, fps)
    pelvis_peak_idx, pelvis_peak_velocity = safe_peak_abs(pelvis_velocity)
    shoulder_peak_idx, shoulder_peak_velocity = safe_peak_abs(shoulder_velocity)
    wrist_peak_idx, wrist_peak_speed = safe_peak_abs(wrist_speed)

    return {
        "pelvis_rotation_range_deg": range_deg(pelvis_angle),
        "shoulder_rotation_range_deg": range_deg(shoulder_angle),
        "shoulder_vs_pelvis_rotation_range_deg": range_deg(shoulder_vs_pelvis),
        "pelvis_peak_abs_velocity_deg_s": abs(pelvis_peak_velocity),
        "shoulder_peak_abs_velocity_deg_s": abs(shoulder_peak_velocity),
        "active_wrist_peak_speed_units_s": abs(wrist_peak_speed),
        "pelvis_fastest_ms_relative_to_anchor": (pelvis_peak_idx - anchor_in_window) * 1000.0 / fps,
        "shoulder_fastest_ms_relative_to_anchor": (shoulder_peak_idx - anchor_in_window) * 1000.0 / fps,
        "wrist_fastest_ms_relative_to_anchor": (wrist_peak_idx - anchor_in_window) * 1000.0 / fps,
        "shoulder_after_pelvis_fastest_ms": (shoulder_peak_idx - pelvis_peak_idx) * 1000.0 / fps,
    }


def compute_sportspose_window_features(
    pose_path: Path,
    window_start: int,
    window_end: int,
    anchor_frame: int,
    active_side: str,
    fps: float,
) -> dict[str, float]:
    sequence = np.load(pose_path)
    window = sequence[window_start : window_end + 1]
    pelvis_angle = sports_pose_line_angle(window, "left_hip", "right_hip")
    shoulder_angle = sports_pose_line_angle(window, "left_shoulder", "right_shoulder")
    wrist_speed = sports_pose_joint_speed(window, f"{active_side}_wrist", fps)
    return compute_features_from_signals(
        pelvis_angle,
        shoulder_angle,
        wrist_speed,
        fps,
        anchor_frame - window_start,
    )


def compute_hmr2_window_features(
    skeleton_path: Path,
    anchor_frame: int,
    active_side: str,
    fps: float,
) -> dict[str, float]:
    sequence, frames, joints = hmr2_to_sequence(skeleton_path)
    if anchor_frame not in frames:
        raise ValueError(f"Anchor frame {anchor_frame} not found in {skeleton_path}")
    pelvis_angle = hmr2_line_angle(sequence, joints, "left_hip", "right_hip")
    shoulder_angle = hmr2_line_angle(sequence, joints, "left_shoulder", "right_shoulder")
    wrist_speed = hmr2_joint_speed(sequence, joints, f"{active_side}_wrist", fps)
    return compute_features_from_signals(
        pelvis_angle,
        shoulder_angle,
        wrist_speed,
        fps,
        frames.index(anchor_frame),
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.paper_results_dir.mkdir(parents=True, exist_ok=True)

    sportspose_features = pd.read_csv(args.sportspose_features)
    sportspose_manifest = pd.read_csv(args.sportspose_manifest)
    hmr2_manifest = pd.read_csv(args.hmr2_skeleton_manifest)

    comparison_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for row in hmr2_manifest.itertuples(index=False):
        trial_id = str(row.trial_id)
        sp_feature = sportspose_features[sportspose_features["trial_id"] == trial_id].iloc[0]
        sp_manifest = sportspose_manifest[sportspose_manifest["trial_id"] == trial_id].iloc[0]
        pose_path = args.sports_root / str(sp_manifest["pose_rel_path"])
        skeleton_path = Path(str(row.sequence_path))
        if not skeleton_path.is_absolute():
            skeleton_path = Path.cwd() / skeleton_path

        fps = float(sp_feature["fps"])
        window_start = int(sp_feature["window_start_frame"])
        window_end = int(sp_feature["window_end_frame"])
        anchor_frame = int(sp_feature["anchor_frame"])
        active_side = str(sp_feature["active_wrist_side"])

        recomputed_sp = compute_sportspose_window_features(
            pose_path,
            window_start,
            window_end,
            anchor_frame,
            active_side,
            fps,
        )
        hmr2 = compute_hmr2_window_features(skeleton_path, anchor_frame, active_side, fps)

        for feature in FEATURE_COLUMNS:
            sp_value = float(recomputed_sp[feature])
            hmr2_value = float(hmr2[feature])
            abs_difference = abs(hmr2_value - sp_value)
            is_timing_feature = "_ms_" in feature or feature.endswith("_ms")
            relative_difference = (
                np.nan if is_timing_feature or abs(sp_value) < 1e-8 else abs_difference / abs(sp_value)
            )
            comparison_rows.append(
                {
                    "trial_id": trial_id,
                    "subject_id": str(row.subject_id),
                    "feature": feature,
                    "sportspose_value": round(sp_value, 4),
                    "hmr2_value": round(hmr2_value, 4),
                    "absolute_difference": round(abs_difference, 4),
                    "relative_difference": round(relative_difference, 4)
                    if np.isfinite(relative_difference)
                    else "",
                }
            )
        paired_rows.append(
            {
                "trial_id": trial_id,
                "subject_id": str(row.subject_id),
                "fps": fps,
                "window_start_frame": window_start,
                "anchor_frame": anchor_frame,
                "window_end_frame": window_end,
                "active_wrist_side": active_side,
                "n_window_frames": int(window_end - window_start + 1),
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    paired = pd.DataFrame(paired_rows)
    comparison_path = args.out_dir / "hmr2_vs_sportspose_feature_comparison.csv"
    paired_path = args.out_dir / "hmr2_vs_sportspose_paired_trials.csv"
    summary_path = args.paper_results_dir / "hmr2_vs_sportspose_feature_comparison.json"
    comparison.to_csv(comparison_path, index=False)
    paired.to_csv(paired_path, index=False)

    timing_features = comparison[
        comparison["feature"].str.contains("_ms_") | comparison["feature"].str.endswith("_ms")
    ]
    range_features = comparison[comparison["feature"].str.contains("_range_deg")]
    speed_features = comparison[comparison["feature"].str.contains("_velocity|wrist_peak")]
    summary = {
        "n_trials": int(paired["trial_id"].nunique()),
        "n_feature_pairs": int(len(comparison)),
        "paired_trials_csv": str(paired_path),
        "comparison_csv": str(comparison_path),
        "median_abs_timing_difference_ms": round(float(timing_features["absolute_difference"].median()), 3)
        if len(timing_features)
        else None,
        "median_abs_rotation_range_difference_deg": round(float(range_features["absolute_difference"].median()), 3)
        if len(range_features)
        else None,
        "median_relative_speed_difference": round(float(speed_features["relative_difference"].median()), 3)
        if len(speed_features)
        else None,
        "note": (
            "This first-pass comparison uses matched frames from the same SportsPose RGB clip. "
            "Wrist-speed values are scale-sensitive and should be interpreted as dataset-internal cues."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote paired trials: {paired_path}")
    print(f"Wrote comparison: {comparison_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
