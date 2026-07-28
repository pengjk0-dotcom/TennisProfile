#!/usr/bin/env python3
"""Extract kinetic chain timing features from 4DHumans skeleton sequences.

This script:
1. Loads skeleton CSV sequences (3D joint coordinates)
2. Computes pelvis and trunk rotation angles from joint positions
3. Calculates angular velocity
4. Extracts timing features: Δt_pelvis_trunk, Peak_order, ROM_ratio

Usage:
    python scripts/extract_kinetic_chain_features.py \
        --skeleton-manifest data/processed/thetis_groundstroke/expert_dev_skeleton_manifest.csv \
        --out-dir data/processed/thetis_groundstroke/expert_features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[1]


def load_skeleton_sequence(path: Path) -> pd.DataFrame:
    """Load a skeleton CSV.gz file."""
    return pd.read_csv(path)


def compute_pelvis_orientation(joints_df: pd.DataFrame) -> np.ndarray:
    """Compute pelvis orientation from left/right hip joints.

    Returns the rotation angle (in degrees) of the pelvis around the vertical axis.
    The angle is computed from the projection of the left-right hip vector onto the
    horizontal plane (x-z plane, assuming y is up).

    Args:
        joints_df: DataFrame with columns [frame, joint, x, y, z] for one frame.

    Returns:
        Pelvis orientation angle in degrees (0-360).
    """
    # Get left and right hip positions
    left_hip = joints_df[joints_df['joint'] == 'left_hip'][['x', 'y', 'z']].values[0]
    right_hip = joints_df[joints_df['joint'] == 'right_hip'][['x', 'y', 'z']].values[0]

    # Compute hip vector (left to right)
    hip_vector = right_hip - left_hip

    # Project onto horizontal plane (ignore y component)
    hip_vector_horizontal = np.array([hip_vector[0], 0, hip_vector[2]])

    # Compute angle relative to x-axis
    angle = np.arctan2(hip_vector_horizontal[2], hip_vector_horizontal[0])
    return np.degrees(angle)


def compute_trunk_orientation(joints_df: pd.DataFrame) -> np.ndarray:
    """Compute trunk orientation from shoulder joints.

    Returns the rotation angle (in degrees) of the trunk around the vertical axis.
    Similar to pelvis orientation but using shoulder joints.

    Args:
        joints_df: DataFrame with columns [frame, joint, x, y, z] for one frame.

    Returns:
        Trunk orientation angle in degrees (0-360).
    """
    # Get left and right shoulder positions
    left_shoulder = joints_df[joints_df['joint'] == 'left_shoulder'][['x', 'y', 'z']].values[0]
    right_shoulder = joints_df[joints_df['joint'] == 'right_shoulder'][['x', 'y', 'z']].values[0]

    # Compute shoulder vector (left to right)
    shoulder_vector = right_shoulder - left_shoulder

    # Project onto horizontal plane
    shoulder_vector_horizontal = np.array([shoulder_vector[0], 0, shoulder_vector[2]])

    # Compute angle relative to x-axis
    angle = np.arctan2(shoulder_vector_horizontal[2], shoulder_vector_horizontal[0])
    return np.degrees(angle)


def compute_joint_angle(joints_df: pd.DataFrame, joint_name: str) -> float:
    """Compute the angle of a specific joint from its parent joints.

    For pelvis and trunk, we compute the rotation around the vertical axis.
    This is a simplified version that uses the orientation of the joint's local frame.
    """
    if joint_name == 'pelvis':
        return compute_pelvis_orientation(joints_df)
    elif joint_name == 'trunk':
        return compute_trunk_orientation(joints_df)
    else:
        raise ValueError(f"Unsupported joint: {joint_name}")


def unwrap_angle_sequence(angles: np.ndarray) -> np.ndarray:
    """Unwrap angle sequence to avoid 360-degree jumps."""
    return np.unwrap(np.radians(angles))


def compute_angular_velocity(angles_rad: np.ndarray, fps: float) -> np.ndarray:
    """Compute angular velocity from angle sequence.

    Args:
        angles_rad: Angle sequence in radians (already unwrapped).
        fps: Frames per second.

    Returns:
        Angular velocity in degrees per second.
    """
    if len(angles_rad) < 2:
        return np.zeros_like(angles_rad, dtype=float)
    # Finite difference
    velocity = np.gradient(angles_rad) * fps
    return np.degrees(velocity)


def effective_cutoff_hz(fps: float, requested_cutoff_hz: float = 6.0) -> float:
    """Return a per-clip cutoff that stays below the Nyquist frequency."""
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return float(min(requested_cutoff_hz, 0.45 * fps))


def butterworth_lowpass(signal: np.ndarray, cutoff: float, fps: float, order: int = 4) -> np.ndarray:
    """Apply Butterworth low-pass filter.

    Args:
        signal: Input signal.
        cutoff: Cutoff frequency in Hz.
        fps: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        Filtered signal.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 4 or not np.isfinite(signal).any():
        return signal

    filled = pd.Series(signal).interpolate(limit_direction="both").to_numpy(dtype=float)
    nyquist = 0.5 * fps
    normal_cutoff = cutoff / nyquist
    if not 0.0 < normal_cutoff < 1.0:
        raise ValueError(
            f"Cutoff must fall below Nyquist: cutoff={cutoff}, fps={fps}, Nyquist={nyquist}"
        )

    # Short tennis stroke windows are common after event alignment. The default
    # order=4 filtfilt pad length is too long for 10-frame smoke tests, so reduce
    # the order and pad length when needed instead of failing.
    filter_order = min(order, max(1, (len(filled) - 2) // 3))
    b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)
    padlen = min(3 * max(len(a), len(b)), len(filled) - 1)
    try:
        return filtfilt(b, a, filled, padlen=padlen)
    except ValueError:
        return filled


def find_peaks_simple(signal: np.ndarray, min_prominence: float = 0.1) -> np.ndarray:
    """Find peaks in a signal using simple derivative-based method.

    Args:
        signal: Input signal.
        min_prominence: Minimum prominence for a peak.

    Returns:
        Array of peak indices.
    """
    signal = np.nan_to_num(np.asarray(signal, dtype=float), nan=0.0)
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            # Check prominence
            left_min = np.min(signal[:i]) if i > 0 else signal[i]
            right_min = np.min(signal[i+1:]) if i < len(signal) - 1 else signal[i]
            prominence = signal[i] - max(left_min, right_min)
            if prominence >= min_prominence:
                peaks.append(i)
    return np.array(peaks)


def dominant_peak_index(signal: np.ndarray, min_prominence: float = 10.0) -> int | None:
    """Return the dominant absolute peak index for a short kinematic signal."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) == 0 or not np.isfinite(signal).any():
        return None
    magnitude = np.abs(np.nan_to_num(signal, nan=0.0))
    adaptive_prominence = max(min_prominence, 0.10 * float(np.nanmax(magnitude) - np.nanmin(magnitude)))
    peaks = find_peaks_simple(magnitude, min_prominence=adaptive_prominence)
    if len(peaks) == 0:
        return int(np.nanargmax(magnitude))
    return int(peaks[np.argmax(magnitude[peaks])])


def joint_xyz_series(skeleton_df: pd.DataFrame, frames: list[int], joint_name: str) -> np.ndarray | None:
    """Return frame-aligned xyz coordinates for a named joint."""
    joint_df = skeleton_df[skeleton_df["joint"] == joint_name][["frame", "x", "y", "z"]]
    if joint_df.empty:
        return None
    values = joint_df.drop_duplicates("frame").set_index("frame").reindex(frames)[["x", "y", "z"]]
    values = values.interpolate(limit_direction="both")
    return values.to_numpy(dtype=float)


def joint_speed(points: np.ndarray | None, fps: float) -> np.ndarray | None:
    if points is None or len(points) == 0:
        return None
    diff = np.diff(points, axis=0, prepend=points[[0]]) * float(fps)
    return np.linalg.norm(diff, axis=1)


def compute_rom(angles: np.ndarray) -> float:
    """Compute Range of Motion (ROM) from angle sequence."""
    return np.max(angles) - np.min(angles)


def process_sequence(
    skeleton_df: pd.DataFrame,
    fps: float = 18.0,
    requested_cutoff_hz: float = 6.0,
) -> dict:
    """Process a single skeleton sequence to extract kinetic chain features.

    Args:
        skeleton_df: DataFrame with skeleton data.
        fps: Frame rate of this clip.
        requested_cutoff_hz: Maximum low-pass cutoff in Hertz.

    Returns:
        Dictionary with extracted features.
    """
    frames = sorted(skeleton_df['frame'].unique())

    pelvis_angles = []
    trunk_angles = []

    for frame in frames:
        frame_data = skeleton_df[skeleton_df['frame'] == frame]
        pelvis_angle = compute_pelvis_orientation(frame_data)
        trunk_angle = compute_trunk_orientation(frame_data)
        pelvis_angles.append(pelvis_angle)
        trunk_angles.append(trunk_angle)

    pelvis_angles = np.array(pelvis_angles)
    trunk_angles = np.array(trunk_angles)

    # Unwrap angles to handle 360-degree jumps
    pelvis_angles_unwrapped = unwrap_angle_sequence(pelvis_angles)
    trunk_angles_unwrapped = unwrap_angle_sequence(trunk_angles)

    # Convert back to degrees for easier interpretation
    pelvis_angles_unwrapped = np.degrees(pelvis_angles_unwrapped)
    trunk_angles_unwrapped = np.degrees(trunk_angles_unwrapped)

    # The effective cutoff is adapted per clip to remain below its Nyquist limit.
    cutoff_hz = effective_cutoff_hz(fps, requested_cutoff_hz)
    pelvis_filtered = butterworth_lowpass(pelvis_angles_unwrapped, cutoff=cutoff_hz, fps=fps)
    trunk_filtered = butterworth_lowpass(trunk_angles_unwrapped, cutoff=cutoff_hz, fps=fps)

    # Compute angular velocity
    pelvis_velocity = compute_angular_velocity(np.radians(pelvis_filtered), fps)
    trunk_velocity = compute_angular_velocity(np.radians(trunk_filtered), fps)

    # Find dominant peaks in absolute angular velocity. Direction can flip with
    # camera/viewpoint, while peak timing is the quantity we care about here.
    pelvis_peak_idx = dominant_peak_index(pelvis_velocity, min_prominence=10.0)
    trunk_peak_idx = dominant_peak_index(trunk_velocity, min_prominence=10.0)
    pelvis_peaks = np.array([] if pelvis_peak_idx is None else [pelvis_peak_idx], dtype=int)
    trunk_peaks = np.array([] if trunk_peak_idx is None else [trunk_peak_idx], dtype=int)

    # Compute ROM
    pelvis_rom = compute_rom(pelvis_filtered)
    trunk_rom = compute_rom(trunk_filtered)
    rom_ratio = trunk_rom / pelvis_rom if pelvis_rom > 0 else 0.0

    separation = trunk_filtered - pelvis_filtered

    left_wrist = joint_xyz_series(skeleton_df, frames, "left_wrist")
    right_wrist = joint_xyz_series(skeleton_df, frames, "right_wrist")
    left_wrist_speed = joint_speed(left_wrist, fps)
    right_wrist_speed = joint_speed(right_wrist, fps)
    left_wrist_peak_idx = dominant_peak_index(left_wrist_speed, min_prominence=0.01) if left_wrist_speed is not None else None
    right_wrist_peak_idx = dominant_peak_index(right_wrist_speed, min_prominence=0.01) if right_wrist_speed is not None else None
    left_wrist_peak_speed = (
        float(left_wrist_speed[left_wrist_peak_idx])
        if left_wrist_speed is not None and left_wrist_peak_idx is not None
        else np.nan
    )
    right_wrist_peak_speed = (
        float(right_wrist_speed[right_wrist_peak_idx])
        if right_wrist_speed is not None and right_wrist_peak_idx is not None
        else np.nan
    )
    if np.nan_to_num(right_wrist_peak_speed, nan=-np.inf) >= np.nan_to_num(left_wrist_peak_speed, nan=-np.inf):
        active_wrist_side = "right"
        active_wrist_speed = right_wrist_speed
        active_wrist_peak_idx = right_wrist_peak_idx
    else:
        active_wrist_side = "left"
        active_wrist_speed = left_wrist_speed
        active_wrist_peak_idx = left_wrist_peak_idx

    # Compute timing features
    features = {
        'pelvis_angles': pelvis_filtered,
        'trunk_angles': trunk_filtered,
        'trunk_pelvis_separation': separation,
        'pelvis_velocity': pelvis_velocity,
        'trunk_velocity': trunk_velocity,
        'left_wrist_speed': left_wrist_speed,
        'right_wrist_speed': right_wrist_speed,
        'active_wrist_speed': active_wrist_speed,
        'pelvis_peaks': pelvis_peaks,
        'trunk_peaks': trunk_peaks,
        'pelvis_rom': pelvis_rom,
        'trunk_rom': trunk_rom,
        'separation_rom': compute_rom(separation),
        'separation_at_pelvis_peak': np.nan,
        'separation_at_trunk_peak': np.nan,
        'rom_ratio': rom_ratio,
        'frames': frames,
        'fps': fps,
        'filter_cutoff_hz': cutoff_hz,
        'active_wrist_side': active_wrist_side,
        'left_wrist_peak_speed': left_wrist_peak_speed,
        'right_wrist_peak_speed': right_wrist_peak_speed,
    }

    # Compute Δt_pelvis_trunk (timing difference between pelvis and trunk peak velocities)
    if pelvis_peak_idx is not None and trunk_peak_idx is not None:
        # Convert frame difference to time difference
        dt_frames = trunk_peak_idx - pelvis_peak_idx
        dt_seconds = dt_frames / fps
        dt_milliseconds = dt_seconds * 1000

        features['dt_pelvis_trunk_ms'] = dt_milliseconds
        features['dt_pelvis_trunk_frames'] = int(dt_frames)
        features['peak_order'] = 'P-D' if dt_frames > 0 else 'D-P' if dt_frames < 0 else 'simultaneous'
        features['pelvis_peak_frame'] = int(frames[pelvis_peak_idx])
        features['trunk_peak_frame'] = int(frames[trunk_peak_idx])
        features['pelvis_peak_velocity_dps'] = float(pelvis_velocity[pelvis_peak_idx])
        features['trunk_peak_velocity_dps'] = float(trunk_velocity[trunk_peak_idx])
        features['pelvis_peak_abs_velocity_dps'] = float(abs(pelvis_velocity[pelvis_peak_idx]))
        features['trunk_peak_abs_velocity_dps'] = float(abs(trunk_velocity[trunk_peak_idx]))
        features['separation_at_pelvis_peak'] = float(separation[pelvis_peak_idx])
        features['separation_at_trunk_peak'] = float(separation[trunk_peak_idx])
    else:
        features['dt_pelvis_trunk_ms'] = np.nan
        features['dt_pelvis_trunk_frames'] = np.nan
        features['peak_order'] = 'unknown'
        features['pelvis_peak_frame'] = np.nan
        features['trunk_peak_frame'] = np.nan
        features['pelvis_peak_velocity_dps'] = np.nan
        features['trunk_peak_velocity_dps'] = np.nan
        features['pelvis_peak_abs_velocity_dps'] = np.nan
        features['trunk_peak_abs_velocity_dps'] = np.nan

    if active_wrist_peak_idx is not None:
        features['active_wrist_peak_frame'] = int(frames[active_wrist_peak_idx])
        features['active_wrist_peak_speed'] = float(active_wrist_speed[active_wrist_peak_idx])
        if trunk_peak_idx is not None:
            features['dt_trunk_active_wrist_ms'] = float((active_wrist_peak_idx - trunk_peak_idx) / fps * 1000)
        else:
            features['dt_trunk_active_wrist_ms'] = np.nan
    else:
        features['active_wrist_peak_frame'] = np.nan
        features['active_wrist_peak_speed'] = np.nan
        features['dt_trunk_active_wrist_ms'] = np.nan

    return features


def load_manual_events(events_csv: Path | None) -> dict[str, dict]:
    if events_csv is None:
        return {}
    events = pd.read_csv(events_csv)
    required = ["trial_id", "swing_start_frame", "ball_contact_frame", "swing_end_frame", "label_quality"]
    missing = [col for col in required if col not in events.columns]
    if missing:
        raise ValueError(f"Events file is missing required columns: {missing}")
    out = {}
    for row in events.itertuples(index=False):
        if pd.isna(row.swing_start_frame) or pd.isna(row.ball_contact_frame) or pd.isna(row.swing_end_frame):
            continue
        out[str(row.trial_id)] = {
            "swing_start_frame": int(row.swing_start_frame),
            "ball_contact_frame": int(row.ball_contact_frame),
            "swing_end_frame": int(row.swing_end_frame),
            "label_quality": str(row.label_quality),
            "notes": str(getattr(row, "notes", "")),
        }
    return out


def cut_to_manual_window(skeleton_df: pd.DataFrame, event: dict | None) -> tuple[pd.DataFrame, dict]:
    if event is None:
        return skeleton_df.copy(), {
            "manual_window_used": False,
            "swing_start_frame": np.nan,
            "ball_contact_frame": np.nan,
            "swing_end_frame": np.nan,
            "label_quality": "",
        }

    start = int(event["swing_start_frame"])
    contact = int(event["ball_contact_frame"])
    end = int(event["swing_end_frame"])
    if not (start <= contact <= end):
        raise ValueError(f"Manual event frames must satisfy start <= contact <= end: {event}")

    window = skeleton_df[(skeleton_df["frame"] >= start) & (skeleton_df["frame"] <= end)].copy()
    if window.empty:
        raise ValueError(f"No skeleton rows found between frames {start} and {end}")

    return window, {
        "manual_window_used": True,
        "swing_start_frame": start,
        "ball_contact_frame": contact,
        "swing_end_frame": end,
        "label_quality": event.get("label_quality", ""),
    }


def frame_minus_contact(frame: float | int, contact: float | int) -> float:
    if pd.isna(frame) or pd.isna(contact):
        return np.nan
    return float(frame) - float(contact)


def process_manifest(
    manifest_path: Path,
    out_dir: Path,
    fps: float = 18.0,
    cutoff_hz: float = 6.0,
    events_csv: Path | None = None,
) -> pd.DataFrame:
    """Process all sequences in a manifest file.

    Args:
        manifest_path: Path to skeleton manifest CSV.
        out_dir: Output directory for feature files.

    Returns:
        DataFrame with features for all sequences.
    """
    manifest = pd.read_csv(manifest_path)
    manual_events = load_manual_events(events_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_features = []

    for _, row in manifest.iterrows():
        trial_id = row['trial_id']
        sequence_path = Path(row['sequence_path'])
        if not sequence_path.is_absolute():
            sequence_path = ROOT / sequence_path
        skill_level = row['skill_level']
        action_type = row['action_type']

        print(f"Processing {trial_id}...")

        # Load skeleton sequence
        skeleton_df = load_skeleton_sequence(sequence_path)
        window_df, window_meta = cut_to_manual_window(skeleton_df, manual_events.get(trial_id))

        # Extract features
        row_fps = row.get("fps", fps)
        row_fps = fps if pd.isna(row_fps) or row_fps == "" else float(row_fps)
        features = process_sequence(
            window_df,
            fps=row_fps,
            requested_cutoff_hz=cutoff_hz,
        )

        ball_contact_frame = window_meta["ball_contact_frame"]
        pelvis_minus_contact = frame_minus_contact(features["pelvis_peak_frame"], ball_contact_frame)
        shoulder_minus_contact = frame_minus_contact(features["trunk_peak_frame"], ball_contact_frame)
        wrist_minus_contact = frame_minus_contact(features["active_wrist_peak_frame"], ball_contact_frame)
        shoulder_minus_pelvis = frame_minus_contact(features["trunk_peak_frame"], features["pelvis_peak_frame"])
        wrist_minus_shoulder = frame_minus_contact(features["active_wrist_peak_frame"], features["trunk_peak_frame"])

        # Store results
        feature_row = {
            'trial_id': trial_id,
            'subject_id': row['subject_id'],
            'skill_level': skill_level,
            'action_type': action_type,
            'frames': len(features['frames']),
            'fps': features['fps'],
            'filter_cutoff_hz': features['filter_cutoff_hz'],
            'manual_window_used': window_meta['manual_window_used'],
            'swing_start_frame': window_meta['swing_start_frame'],
            'ball_contact_frame': ball_contact_frame,
            'swing_end_frame': window_meta['swing_end_frame'],
            'label_quality': window_meta['label_quality'],
            'pelvis_rom': features['pelvis_rom'],
            'trunk_rom': features['trunk_rom'],
            'separation_rom': features['separation_rom'],
            'rom_ratio': features['rom_ratio'],
            'dt_pelvis_trunk_ms': features['dt_pelvis_trunk_ms'],
            'dt_pelvis_trunk_frames': features['dt_pelvis_trunk_frames'],
            'dt_trunk_active_wrist_ms': features['dt_trunk_active_wrist_ms'],
            'peak_order': features['peak_order'],
            'pelvis_peak_frame': features['pelvis_peak_frame'],
            'trunk_peak_frame': features['trunk_peak_frame'],
            'active_wrist_side': features['active_wrist_side'],
            'active_wrist_peak_frame': features['active_wrist_peak_frame'],
            'active_wrist_peak_speed': features['active_wrist_peak_speed'],
            'pelvis_fastest_frame': features['pelvis_peak_frame'],
            'shoulder_fastest_frame': features['trunk_peak_frame'],
            'wrist_fastest_frame': features['active_wrist_peak_frame'],
            'pelvis_fastest_minus_contact_frames': pelvis_minus_contact,
            'shoulder_fastest_minus_contact_frames': shoulder_minus_contact,
            'wrist_fastest_minus_contact_frames': wrist_minus_contact,
            'shoulder_minus_pelvis_fastest_frames': shoulder_minus_pelvis,
            'wrist_minus_shoulder_fastest_frames': wrist_minus_shoulder,
            'pelvis_fastest_minus_contact_ms': pelvis_minus_contact / row_fps * 1000 if np.isfinite(pelvis_minus_contact) else np.nan,
            'shoulder_fastest_minus_contact_ms': shoulder_minus_contact / row_fps * 1000 if np.isfinite(shoulder_minus_contact) else np.nan,
            'wrist_fastest_minus_contact_ms': wrist_minus_contact / row_fps * 1000 if np.isfinite(wrist_minus_contact) else np.nan,
            'pelvis_peak_velocity_dps': features['pelvis_peak_velocity_dps'],
            'trunk_peak_velocity_dps': features['trunk_peak_velocity_dps'],
            'pelvis_peak_abs_velocity_dps': features['pelvis_peak_abs_velocity_dps'],
            'trunk_peak_abs_velocity_dps': features['trunk_peak_abs_velocity_dps'],
            'separation_at_pelvis_peak': features['separation_at_pelvis_peak'],
            'separation_at_trunk_peak': features['separation_at_trunk_peak'],
        }
        all_features.append(feature_row)

        # Save detailed time series
        timeseries_df = pd.DataFrame({
            'frame': features['frames'],
            'frame_from_ball_contact': [
                frame_minus_contact(frame, ball_contact_frame) for frame in features['frames']
            ],
            'pelvis_angle_deg': features['pelvis_angles'],
            'trunk_angle_deg': features['trunk_angles'],
            'trunk_pelvis_separation_deg': features['trunk_pelvis_separation'],
            'pelvis_velocity_dps': features['pelvis_velocity'],
            'trunk_velocity_dps': features['trunk_velocity'],
            'left_wrist_speed': features['left_wrist_speed'],
            'right_wrist_speed': features['right_wrist_speed'],
            'active_wrist_speed': features['active_wrist_speed'],
        })
        timeseries_path = out_dir / f"{trial_id}_timeseries.csv"
        timeseries_df.to_csv(timeseries_path, index=False)

        # Save peaks
        peak_indices = np.concatenate([features['pelvis_peaks'], features['trunk_peaks']])
        peak_frames = [features['frames'][int(idx)] for idx in peak_indices]
        peaks_df = pd.DataFrame({
            'joint': ['pelvis'] * len(features['pelvis_peaks']) + ['trunk'] * len(features['trunk_peaks']),
            'peak_frame': peak_frames,
            'frame_from_ball_contact': [
                frame_minus_contact(frame, ball_contact_frame) for frame in peak_frames
            ],
            'peak_velocity': np.concatenate([
                features['pelvis_velocity'][features['pelvis_peaks']],
                features['trunk_velocity'][features['trunk_peaks']]
            ]),
            'peak_abs_velocity': np.concatenate([
                np.abs(features['pelvis_velocity'][features['pelvis_peaks']]),
                np.abs(features['trunk_velocity'][features['trunk_peaks']])
            ]),
        })
        peaks_path = out_dir / f"{trial_id}_peaks.csv"
        peaks_df.to_csv(peaks_path, index=False)

    # Create summary DataFrame
    features_df = pd.DataFrame(all_features)

    # Save features
    features_path = out_dir / "kinetic_chain_features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"Saved features to {features_path}")

    return features_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract kinetic chain timing features")
    parser.add_argument("--skeleton-manifest", required=True, help="Path to skeleton manifest CSV")
    parser.add_argument("--out-dir", required=True, help="Output directory for features")
    parser.add_argument("--fps", type=float, default=18.0, help="Frames per second")
    parser.add_argument("--cutoff-hz", type=float, default=6.0, help="Maximum low-pass cutoff in Hertz")
    parser.add_argument("--events-csv", default=None, help="Optional manual stroke event CSV.")
    args = parser.parse_args()

    manifest_path = Path(args.skeleton_manifest)
    out_dir = Path(args.out_dir)

    events_csv = Path(args.events_csv) if args.events_csv else None
    features_df = process_manifest(
        manifest_path,
        out_dir,
        fps=args.fps,
        cutoff_hz=args.cutoff_hz,
        events_csv=events_csv,
    )

    # Print summary
    print("\n" + "="*60)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total sequences: {len(features_df)}")
    print(f"\nBy skill level:")
    print(features_df['skill_level'].value_counts())
    print(f"\nFeature statistics:")
    print(features_df[['pelvis_rom', 'trunk_rom', 'rom_ratio', 'dt_pelvis_trunk_ms']].describe())

    return 0


if __name__ == "__main__":
    sys.exit(main())
