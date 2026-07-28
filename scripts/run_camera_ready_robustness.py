#!/usr/bin/env python3
"""Camera-ready robustness and cohort-audit analyses for TennisProfile.

The script extends the submitted analysis without changing the final cohort or
the feature definitions. It produces:

1. Benjamini-Hochberg adjusted group-comparison results and effect sizes;
2. exhaustive three-subject expert hold-out sensitivity checks;
3. a same-capture-setting hold-out sensitivity check;
4. stroke-duration and pooled-reference supplementary baselines;
5. SportsPose 90-to-18/10 fps trajectory-resampling sensitivity; and
6. subject-selection, capture-setting, and racket-hand audit tables.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu


def find_release_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (
            (parent / "data/final_all_assigned_features.csv").is_file()
            and (parent / "data/final_master_split.csv").is_file()
            and (parent / "scripts").is_dir()
        ):
            return parent
    raise RuntimeError("Could not locate the TennisProfile release root.")


ROOT = find_release_root()
SCRIPT_ROOT = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from analyze_sportspose_tennis import (  # noqa: E402
    COCO17,
    derivative,
    horizontal_line_angle,
    joint_speed,
    nan_interpolate,
    range_deg,
    safe_peak_abs,
    smooth_signal,
)
from run_paper_sanity_experiments import (  # noqa: E402
    FEATURE_LABELS,
    PROFILE_FEATURES,
    cliffs_delta,
)


FEATURES_PATH = ROOT / "data/final_all_assigned_features.csv"
SPLIT_PATH = ROOT / "data/final_master_split.csv"
SPORTSPOSE_ROOT = Path(
    os.environ.get(
        "SPORTSPOSE_ROOT",
        str(ROOT / "external/SportsPose"),
    )
)
RESULTS = ROOT / "results"
AUDIT = ROOT / "reproducibility"

ROTATION_FEATURES = ["pelvis_rom", "trunk_rom", "separation_rom"]
ROOM_SUBJECTS = {"p47", "p48", "p49", "p50", "p51", "p53"}
REMOVED_SUBJECTS = {
    "p1": (
        "removed_after_manual_review",
        "visual_identity_overlap_with_retained_p50",
    ),
    "p52": (
        "removed_after_manual_review",
        "nominal_forehand_and_backhand_files_not_independent_stroke_clips",
    ),
}


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjustment."""

    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def build_adjusted_statistics(features: pd.DataFrame) -> pd.DataFrame:
    expert = features[features["final_role"] == "expert_reference"]
    beginner = features[features["final_role"] == "beginner_test"]
    rows: list[dict[str, Any]] = []

    for action_type in sorted(expert["action_type"].unique()):
        e_action = expert[expert["action_type"] == action_type]
        b_action = beginner[beginner["action_type"] == action_type]
        for feature in PROFILE_FEATURES:
            e = e_action[feature].astype(float).to_numpy()
            b = b_action[feature].astype(float).to_numpy()
            statistic, p_value = mannwhitneyu(b, e, alternative="two-sided")
            expert_mean = float(np.mean(e))
            below = int(np.sum(b < expert_mean))
            above = int(np.sum(b > expert_mean))
            dominant = max(below, above)
            rows.append(
                {
                    "action_type": action_type,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "expert_n": len(e),
                    "beginner_n": len(b),
                    "expert_mean": expert_mean,
                    "beginner_mean": float(np.mean(b)),
                    "mann_whitney_u": float(statistic),
                    "mann_whitney_u_p": float(p_value),
                    "cliffs_delta_beginner_vs_expert": float(cliffs_delta(b, e)),
                    "beginner_below_expert_mean": below,
                    "beginner_above_expert_mean": above,
                    "direction_consistency_p_one_sided": float(
                        binomtest(
                            dominant, len(b), 0.5, alternative="greater"
                        ).pvalue
                    ),
                }
            )

    result = pd.DataFrame(rows)
    result["mann_whitney_q_bh_global_22"] = bh_adjust(
        result["mann_whitney_u_p"].to_numpy()
    )
    result["mann_whitney_q_bh_within_stroke_11"] = np.nan
    result["direction_consistency_q_bh_global_22"] = bh_adjust(
        result["direction_consistency_p_one_sided"].to_numpy()
    )
    result["direction_consistency_q_bh_within_stroke_11"] = np.nan
    for _, indices in result.groupby("action_type").groups.items():
        result.loc[indices, "mann_whitney_q_bh_within_stroke_11"] = bh_adjust(
            result.loc[indices, "mann_whitney_u_p"].to_numpy()
        )
        result.loc[indices, "direction_consistency_q_bh_within_stroke_11"] = (
            bh_adjust(
                result.loc[
                    indices, "direction_consistency_p_one_sided"
                ].to_numpy()
            )
        )
    return result.sort_values(["action_type", "mann_whitney_u_p"])


def reference_parameters(reference: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    parameters: dict[str, dict[str, np.ndarray]] = {}
    for action_type, group in reference.groupby("action_type"):
        values = group[PROFILE_FEATURES].astype(float)
        parameters[action_type] = {
            "mean": values.mean(axis=0).to_numpy(),
            "std": values.std(axis=0, ddof=1).to_numpy(),
            "min": values.min(axis=0).to_numpy(),
            "max": values.max(axis=0).to_numpy(),
        }
    return parameters


def reference_metrics(
    samples: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, float]:
    params = reference_parameters(reference)
    all_abs_z: list[float] = []
    all_outside: list[bool] = []
    stroke_scores: list[tuple[str, float]] = []
    rotation_indices = [PROFILE_FEATURES.index(feature) for feature in ROTATION_FEATURES]

    for action_type, group in samples.groupby("action_type"):
        current = params[action_type]
        values = group[PROFILE_FEATURES].astype(float).to_numpy()
        std = current["std"].copy()
        std[~np.isfinite(std) | (std <= 1e-12)] = np.nan
        z = (values - current["mean"]) / std
        outside = (values < current["min"]) | (values > current["max"])
        all_abs_z.extend(np.abs(z[np.isfinite(z)]).tolist())
        all_outside.extend(outside.ravel().tolist())
        rotation_score = np.sqrt(np.nanmean(np.square(z[:, rotation_indices]), axis=1))
        stroke_scores.extend(zip(group["subject_id"].astype(str), rotation_score))

    subject_scores = (
        pd.DataFrame(stroke_scores, columns=["subject_id", "score"])
        .groupby("subject_id")["score"]
        .mean()
    )
    return {
        "mean_abs_z": float(np.mean(all_abs_z)),
        "outside_rate": float(np.mean(all_outside)),
        "mean_subject_rotation_score": float(subject_scores.mean()),
    }


def exhaustive_holdout(
    expert_pool: pd.DataFrame,
    beginner: pd.DataFrame,
    holdout_size: int,
    setting: str,
) -> pd.DataFrame:
    subjects = sorted(expert_pool["subject_id"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for held_subjects in itertools.combinations(subjects, holdout_size):
        held_set = set(held_subjects)
        reference = expert_pool[~expert_pool["subject_id"].isin(held_set)]
        heldout = expert_pool[expert_pool["subject_id"].isin(held_set)]
        hold_metrics = reference_metrics(heldout, reference)
        beginner_metrics = reference_metrics(beginner, reference)
        rows.append(
            {
                "setting": setting,
                "held_out_subjects": ";".join(held_subjects),
                "reference_n_subjects": reference["subject_id"].nunique(),
                "heldout_n_subjects": heldout["subject_id"].nunique(),
                "beginner_n_subjects": beginner["subject_id"].nunique(),
                "heldout_mean_abs_z": hold_metrics["mean_abs_z"],
                "beginner_mean_abs_z": beginner_metrics["mean_abs_z"],
                "mean_abs_z_gap_beginner_minus_holdout": (
                    beginner_metrics["mean_abs_z"] - hold_metrics["mean_abs_z"]
                ),
                "heldout_outside_rate": hold_metrics["outside_rate"],
                "beginner_outside_rate": beginner_metrics["outside_rate"],
                "outside_rate_gap_beginner_minus_holdout": (
                    beginner_metrics["outside_rate"] - hold_metrics["outside_rate"]
                ),
                "heldout_mean_subject_rotation_score": hold_metrics[
                    "mean_subject_rotation_score"
                ],
                "beginner_mean_subject_rotation_score": beginner_metrics[
                    "mean_subject_rotation_score"
                ],
                "rotation_score_gap_beginner_minus_holdout": (
                    beginner_metrics["mean_subject_rotation_score"]
                    - hold_metrics["mean_subject_rotation_score"]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_holdout(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for setting, group in detail.groupby("setting"):
        row: dict[str, Any] = {
            "setting": setting,
            "n_splits": len(group),
            "reference_n_subjects": int(group["reference_n_subjects"].iloc[0]),
            "heldout_n_subjects": int(group["heldout_n_subjects"].iloc[0]),
            "beginner_n_subjects": int(group["beginner_n_subjects"].iloc[0]),
        }
        for metric in [
            "mean_abs_z_gap_beginner_minus_holdout",
            "outside_rate_gap_beginner_minus_holdout",
            "rotation_score_gap_beginner_minus_holdout",
        ]:
            values = group[metric].to_numpy()
            row[f"{metric}_positive_fraction"] = float(np.mean(values > 0))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            row[f"{metric}_q975"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_reference_metrics(
    samples: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, float]:
    ref_values = reference[PROFILE_FEATURES].astype(float)
    values = samples[PROFILE_FEATURES].astype(float).to_numpy()
    mean = ref_values.mean(axis=0).to_numpy()
    std = ref_values.std(axis=0, ddof=1).to_numpy(copy=True)
    minimum = ref_values.min(axis=0).to_numpy()
    maximum = ref_values.max(axis=0).to_numpy()
    std[~np.isfinite(std) | (std <= 1e-12)] = np.nan
    z = (values - mean) / std
    outside = (values < minimum) | (values > maximum)
    rotation_indices = [PROFILE_FEATURES.index(feature) for feature in ROTATION_FEATURES]
    stroke_scores = np.sqrt(np.nanmean(np.square(z[:, rotation_indices]), axis=1))
    subject_scores = (
        pd.DataFrame(
            {
                "subject_id": samples["subject_id"].astype(str).to_numpy(),
                "score": stroke_scores,
            }
        )
        .groupby("subject_id")["score"]
        .mean()
    )
    return {
        "mean_abs_z": float(np.nanmean(np.abs(z))),
        "outside_rate": float(np.mean(outside)),
        "mean_subject_rotation_score": float(subject_scores.mean()),
    }


def build_reference_baseline(features: pd.DataFrame) -> pd.DataFrame:
    reference = features[features["final_role"] == "expert_reference"]
    holdout = features[features["final_role"] == "expert_holdout"]
    beginner = features[features["final_role"] == "beginner_test"]
    rows: list[dict[str, Any]] = []

    for reference_type in ["stroke_specific", "pooled_strokes"]:
        if reference_type == "stroke_specific":
            hold_metrics = reference_metrics(holdout, reference)
            beginner_metrics = reference_metrics(beginner, reference)
        else:
            hold_metrics = pooled_reference_metrics(holdout, reference)
            beginner_metrics = pooled_reference_metrics(beginner, reference)
        rows.append(
            {
                "reference_type": reference_type,
                **{f"heldout_{k}": v for k, v in hold_metrics.items()},
                **{f"beginner_{k}": v for k, v in beginner_metrics.items()},
                "mean_abs_z_gap_beginner_minus_holdout": (
                    beginner_metrics["mean_abs_z"] - hold_metrics["mean_abs_z"]
                ),
                "outside_rate_gap_beginner_minus_holdout": (
                    beginner_metrics["outside_rate"] - hold_metrics["outside_rate"]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_duration_summary(
    split: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    duration = split.copy()
    duration["duration_frames_between_anchors"] = (
        duration["swing_end_frame"] - duration["swing_start_frame"]
    )
    duration["duration_ms"] = (
        duration["duration_frames_between_anchors"] * 1000.0 / duration["fps"]
    )
    summary = (
        duration.groupby(["final_role", "action_type"])
        .agg(
            n=("trial_id", "size"),
            duration_frames_mean=("duration_frames_between_anchors", "mean"),
            duration_frames_sd=("duration_frames_between_anchors", "std"),
            duration_ms_mean=("duration_ms", "mean"),
            duration_ms_sd=("duration_ms", "std"),
            duration_ms_median=("duration_ms", "median"),
        )
        .reset_index()
    )
    test_rows: list[dict[str, Any]] = []
    for action_type in sorted(duration["action_type"].unique()):
        expert = duration.loc[
            (duration["action_type"] == action_type)
            & (duration["final_role"] == "expert_reference"),
            "duration_ms",
        ].to_numpy()
        beginner = duration.loc[
            (duration["action_type"] == action_type)
            & (duration["final_role"] == "beginner_test"),
            "duration_ms",
        ].to_numpy()
        statistic, p_value = mannwhitneyu(
            beginner, expert, alternative="two-sided"
        )
        test_rows.append(
            {
                "action_type": action_type,
                "expert_n": len(expert),
                "beginner_n": len(beginner),
                "expert_duration_ms_mean": float(np.mean(expert)),
                "beginner_duration_ms_mean": float(np.mean(beginner)),
                "mann_whitney_u": float(statistic),
                "mann_whitney_u_p": float(p_value),
                "cliffs_delta_beginner_vs_expert": float(
                    cliffs_delta(beginner, expert)
                ),
            }
        )
    detail = duration[
        [
            "trial_id",
            "subject_id",
            "final_role",
            "action_type",
            "fps",
            "duration_frames_between_anchors",
            "duration_ms",
        ]
    ]
    return detail, summary, pd.DataFrame(test_rows)


def compute_sportspose_features(sequence: np.ndarray, fps: float) -> dict[str, float]:
    cutoff = min(6.0, 0.45 * fps)
    half_window_frames = int(round(0.5 * fps))
    left_speed = smooth_signal(
        joint_speed(sequence, "left_wrist", fps), fps, cutoff_hz=cutoff
    )
    right_speed = smooth_signal(
        joint_speed(sequence, "right_wrist", fps), fps, cutoff_hz=cutoff
    )
    active_side = "left" if np.nanmax(left_speed) >= np.nanmax(right_speed) else "right"
    active_speed = left_speed if active_side == "left" else right_speed
    anchor = int(np.nanargmax(active_speed))
    start = max(0, anchor - half_window_frames)
    end = min(len(sequence) - 1, anchor + half_window_frames)
    window = sequence[start : end + 1]
    anchor_in_window = anchor - start

    pelvis = smooth_signal(
        horizontal_line_angle(window, "left_hip", "right_hip"),
        fps,
        cutoff_hz=cutoff,
    )
    shoulder = smooth_signal(
        horizontal_line_angle(window, "left_shoulder", "right_shoulder"),
        fps,
        cutoff_hz=cutoff,
    )
    relative = nan_interpolate(shoulder - pelvis)
    pelvis_velocity = derivative(pelvis, fps)
    shoulder_velocity = derivative(shoulder, fps)
    wrist_velocity = smooth_signal(
        joint_speed(window, f"{active_side}_wrist", fps),
        fps,
        cutoff_hz=cutoff,
    )

    pelvis_peak_idx, pelvis_peak = safe_peak_abs(pelvis_velocity)
    shoulder_peak_idx, shoulder_peak = safe_peak_abs(shoulder_velocity)
    wrist_peak_idx, wrist_peak = safe_peak_abs(wrist_velocity)
    return {
        "pelvis_rotation_range_deg": range_deg(pelvis),
        "shoulder_rotation_range_deg": range_deg(shoulder),
        "relative_rotation_range_deg": range_deg(relative),
        "pelvis_peak_speed": abs(pelvis_peak),
        "shoulder_peak_speed": abs(shoulder_peak),
        "wrist_peak_speed": abs(wrist_peak),
        "pelvis_fastest_ms": (pelvis_peak_idx - anchor_in_window) * 1000.0 / fps,
        "shoulder_fastest_ms": (shoulder_peak_idx - anchor_in_window)
        * 1000.0
        / fps,
        "wrist_fastest_ms": (wrist_peak_idx - anchor_in_window) * 1000.0 / fps,
    }


def sports_pose_rate_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    pose_paths = sorted(
        (SPORTSPOSE_ROOT / "data/outdoors").glob("S*/tennis/tennis*.npy")
    )
    if not pose_paths:
        raise FileNotFoundError("SportsPose outdoor tennis poses were not found.")

    feature_groups = {
        "rotation_range": [
            "pelvis_rotation_range_deg",
            "shoulder_rotation_range_deg",
            "relative_rotation_range_deg",
        ],
        "peak_speed": [
            "pelvis_peak_speed",
            "shoulder_peak_speed",
            "wrist_peak_speed",
        ],
        "fastest_motion_timing": [
            "pelvis_fastest_ms",
            "shoulder_fastest_ms",
            "wrist_fastest_ms",
        ],
    }
    rows: list[dict[str, Any]] = []

    for pose_path in pose_paths:
        source = np.load(pose_path)
        baseline = compute_sportspose_features(source, 90.0)
        for target_fps in [18.0, 10.0]:
            step = int(round(90.0 / target_fps))
            sampled = source[::step]
            candidate = compute_sportspose_features(sampled, 90.0 / step)
            for group, names in feature_groups.items():
                for feature in names:
                    base = float(baseline[feature])
                    value = float(candidate[feature])
                    absolute = abs(value - base)
                    relative = absolute / max(abs(base), 1e-12)
                    rows.append(
                        {
                            "trial_id": f"{pose_path.parents[1].name}_{pose_path.stem}",
                            "subject_id": pose_path.parents[1].name,
                            "source_fps": 90.0,
                            "target_fps": target_fps,
                            "feature_group": group,
                            "feature": feature,
                            "baseline_value_90fps": base,
                            "resampled_value": value,
                            "absolute_difference": absolute,
                            "relative_difference": relative,
                        }
                    )

    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for (target_fps, group), values in detail.groupby(
        ["target_fps", "feature_group"]
    ):
        summary_rows.append(
            {
                "source_fps": 90.0,
                "target_fps": target_fps,
                "feature_group": group,
                "n_feature_pairs": len(values),
                "n_trials": values["trial_id"].nunique(),
                "n_subjects": values["subject_id"].nunique(),
                "median_absolute_difference": float(
                    values["absolute_difference"].median()
                ),
                "median_relative_difference": float(
                    values["relative_difference"].median()
                ),
                "q25_relative_difference": float(
                    values["relative_difference"].quantile(0.25)
                ),
                "q75_relative_difference": float(
                    values["relative_difference"].quantile(0.75)
                ),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def build_selection_and_trial_audits(
    split: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_subjects = set(split["subject_id"].astype(str))
    role_by_subject = (
        split[["subject_id", "final_role"]]
        .drop_duplicates()
        .set_index("subject_id")["final_role"]
        .to_dict()
    )
    subject_rows: list[dict[str, Any]] = []
    for index in range(1, 56):
        subject = f"p{index}"
        if subject in final_subjects:
            status = "included"
            reason = (
                "paired_flat_forehand_and_one_handed_backhand_with_completed_"
                "event_review_and_successful_feature_extraction"
            )
        elif subject in REMOVED_SUBJECTS:
            status, reason = REMOVED_SUBJECTS[subject]
        else:
            status = "not_sampled"
            reason = "not_in_fixed_manually_reviewed_cohort"
        subject_rows.append(
            {
                "subject_id": subject,
                "selection_status": status,
                "final_role": role_by_subject.get(subject, ""),
                "selection_or_exclusion_record": reason,
                "feature_values_used_for_selection": False,
                "capture_setting_if_included": (
                    "small_indoor_room"
                    if subject in ROOM_SUBJECTS
                    else ("indoor_gym" if subject in final_subjects else "")
                ),
            }
        )

    active_side = features[["trial_id", "active_wrist_side"]].copy()
    trial = split.merge(active_side, on="trial_id", how="left")
    trial["capture_setting"] = np.where(
        trial["subject_id"].isin(ROOM_SUBJECTS),
        "small_indoor_room",
        "indoor_gym",
    )
    trial["observed_racket_hand"] = "right"
    trial["hand_audit_method"] = "event_frame_montage_and_source_video_review"
    trial["feature_direction_handling"] = (
        "body_line_range_and_absolute_angular_speed_are_sign_invariant;"
        "wrist_descriptor_uses_higher_speed_side_not_anatomical_hand_label"
    )
    trial["view_description"] = "fixed_monocular_indoor_view"
    return pd.DataFrame(subject_rows), trial


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate TennisProfile robustness analyses from the released "
            "tables. SportsPose resampling additionally requires a licensed "
            "SportsPose checkout."
        )
    )
    parser.add_argument(
        "--skip-sportspose",
        action="store_true",
        help=(
            "Reuse the released SportsPose frame-rate summary instead of "
            "recomputing it from licensed 3D pose files."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(FEATURES_PATH)
    split = pd.read_csv(SPLIT_PATH)

    stats = build_adjusted_statistics(features)
    stats.to_csv(RESULTS / "exploratory_statistics_bh_effect_sizes.csv", index=False)

    expert_pool = features[
        features["final_role"].isin(["expert_reference", "expert_holdout"])
    ].copy()
    beginner = features[features["final_role"] == "beginner_test"].copy()
    all_setting = exhaustive_holdout(
        expert_pool,
        beginner,
        holdout_size=3,
        setting="all_selected_capture_settings",
    )

    gym_expert = expert_pool[~expert_pool["subject_id"].isin(ROOM_SUBJECTS)].copy()
    gym_beginner = beginner[~beginner["subject_id"].isin(ROOM_SUBJECTS)].copy()
    gym_setting = exhaustive_holdout(
        gym_expert,
        gym_beginner,
        holdout_size=3,
        setting="indoor_gym_only",
    )
    holdout_detail = pd.concat([all_setting, gym_setting], ignore_index=True)
    holdout_summary = summarize_holdout(holdout_detail)
    holdout_detail.to_csv(
        RESULTS / "exhaustive_three_expert_holdout_detail.csv", index=False
    )
    holdout_summary.to_csv(
        RESULTS / "exhaustive_three_expert_holdout_summary.csv", index=False
    )

    baseline = build_reference_baseline(features)
    baseline.to_csv(RESULTS / "stroke_specific_vs_pooled_reference.csv", index=False)

    duration_detail, duration_summary, duration_tests = build_duration_summary(split)
    duration_detail.to_csv(RESULTS / "stroke_duration_detail.csv", index=False)
    duration_summary.to_csv(RESULTS / "stroke_duration_summary.csv", index=False)
    duration_tests.to_csv(
        RESULTS / "stroke_duration_group_tests.csv", index=False
    )

    rate_summary_path = RESULTS / "sportspose_frame_rate_sensitivity_summary.csv"
    if args.skip_sportspose:
        if not rate_summary_path.exists():
            raise FileNotFoundError(
                "--skip-sportspose requires the released SportsPose summary."
            )
        rate_summary = pd.read_csv(rate_summary_path)
        print("Reusing released SportsPose frame-rate summary.")
    else:
        rate_detail, rate_summary = sports_pose_rate_sensitivity()
        rate_detail.to_csv(
            RESULTS / "sportspose_frame_rate_sensitivity_detail.csv", index=False
        )
        rate_summary.to_csv(rate_summary_path, index=False)

    subject_audit, trial_audit = build_selection_and_trial_audits(split, features)
    subject_audit.to_csv(AUDIT / "subject_selection_audit.csv", index=False)
    trial_audit.to_csv(AUDIT / "trial_view_hand_audit.csv", index=False)

    current_holdout = tuple(
        sorted(
            features.loc[
                features["final_role"] == "expert_holdout", "subject_id"
            ].unique()
        )
    )
    current_key = ";".join(current_holdout)
    current_row = all_setting[
        all_setting["held_out_subjects"] == current_key
    ].iloc[0]

    summary = {
        "cohort": {
            "selected_subjects": int(split["subject_id"].nunique()),
            "selected_trials": int(len(split)),
            "expert_reference_subjects": int(
                split.loc[
                    split["final_role"] == "expert_reference", "subject_id"
                ].nunique()
            ),
            "expert_holdout_subjects": int(
                split.loc[
                    split["final_role"] == "expert_holdout", "subject_id"
                ].nunique()
            ),
            "beginner_subjects": int(
                split.loc[
                    split["final_role"] == "beginner_test", "subject_id"
                ].nunique()
            ),
            "documented_post_review_removals": sorted(REMOVED_SUBJECTS),
            "not_sampled_subjects": int(
                (subject_audit["selection_status"] == "not_sampled").sum()
            ),
            "all_retained_trials_observed_right_handed": bool(
                (trial_audit["observed_racket_hand"] == "right").all()
            ),
            "capture_setting_subject_counts": (
                trial_audit[["subject_id", "capture_setting"]]
                .drop_duplicates()["capture_setting"]
                .value_counts()
                .to_dict()
            ),
        },
        "multiple_comparisons": {
            "tests_total": int(len(stats)),
            "tests_per_stroke": 11,
            "minimum_uncorrected_p": float(stats["mann_whitney_u_p"].min()),
            "minimum_bh_q_within_stroke": float(
                stats["mann_whitney_q_bh_within_stroke_11"].min()
            ),
            "minimum_bh_q_global": float(
                stats["mann_whitney_q_bh_global_22"].min()
            ),
            "discoveries_q_lt_0_05_within_stroke": int(
                (stats["mann_whitney_q_bh_within_stroke_11"] < 0.05).sum()
            ),
            "discoveries_q_lt_0_05_global": int(
                (stats["mann_whitney_q_bh_global_22"] < 0.05).sum()
            ),
            "direction_minimum_bh_q_within_stroke": float(
                stats["direction_consistency_q_bh_within_stroke_11"].min()
            ),
            "direction_discoveries_q_lt_0_05_within_stroke": int(
                (
                    stats["direction_consistency_q_bh_within_stroke_11"]
                    < 0.05
                ).sum()
            ),
        },
        "predefined_holdout": current_row.to_dict(),
        "exhaustive_holdout": holdout_summary.to_dict(orient="records"),
        "sports_pose_frame_rate_sensitivity": rate_summary.to_dict(
            orient="records"
        ),
        "reference_baseline": baseline.to_dict(orient="records"),
    }
    write_json(RESULTS / "camera_ready_robustness_summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
