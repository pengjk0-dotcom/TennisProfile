#!/usr/bin/env python3
"""Refresh paper sanity checks from the final master split.

This script intentionally uses the final master split as the single source of
truth, so the paper text, figures, and robustness checks share the same cohort
definition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_kinetic_chain_features import process_manifest  # noqa: E402
from run_paper_sanity_experiments import (  # noqa: E402
    FEATURE_LABELS,
    PROFILE_FEATURES,
    compare_to_reference,
    reference_table,
    top_features,
    top_overlap,
)

DATA = ROOT / "data/processed/thetis_groundstroke"
ANALYSIS = DATA / "final_master_analysis"
RESULTS = ROOT / "paper/results"
TEMP = DATA / "final_master_sensitivity"
ROTATION_FEATURES = ["pelvis_rom", "trunk_rom", "separation_rom"]

MANIFEST_CANDIDATES = [
    DATA / "reannotation_v2_manifests/expert_reference_manifest.csv",
    DATA / "reannotation_v2_manifests/holdout_manifest.csv",
    DATA / "reannotation_v2_manifests/beginner_manifest.csv",
    DATA / "annotation_expansion_v3_skeleton_manifest.csv",
]


def load_final_sets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = pd.read_csv(DATA / "final_master_split.csv")
    features = pd.read_csv(ANALYSIS / "final_all_assigned_features.csv")
    expert = features[features["final_role"] == "expert_reference"].copy()
    beginner = features[features["final_role"] == "beginner_test"].copy()
    holdout = features[features["final_role"] == "expert_holdout"].copy()
    return split, expert, beginner, holdout


def build_final_manifests(split: pd.DataFrame) -> tuple[Path, Path, Path, Path]:
    rows = []
    for path in MANIFEST_CANDIDATES:
        if path.exists():
            rows.append(pd.read_csv(path))
    if not rows:
        raise FileNotFoundError("No skeleton manifests found")

    manifest = pd.concat(rows, ignore_index=True)
    manifest = manifest.drop_duplicates("trial_id", keep="last")
    keep_cols = [
        "trial_id",
        "subject_id",
        "skill_level",
        "stroke_type",
        "action_type",
        "frames",
        "joints_per_frame",
        "sequence_path",
        "fps",
    ]
    manifest = manifest[[c for c in keep_cols if c in manifest.columns]]
    merged = split.merge(
        manifest,
        on=["trial_id", "subject_id", "skill_level", "action_type"],
        how="left",
        suffixes=("", "_manifest"),
    )
    missing = merged.loc[merged["sequence_path"].isna(), "trial_id"].tolist()
    if missing:
        raise ValueError(f"Missing skeleton sequence_path for final trials: {missing}")

    merged["stroke_type"] = merged.get("stroke_type", "groundstroke")
    for col in ["frames", "fps"]:
        manifest_col = f"{col}_manifest"
        if manifest_col in merged.columns:
            merged[col] = merged[col].fillna(merged[manifest_col])
    out_manifest = ANALYSIS / "final_skeleton_manifest.csv"
    event_cols = [
        "trial_id",
        "swing_start_frame",
        "ball_contact_frame",
        "swing_end_frame",
        "label_quality",
    ]
    out_events = ANALYSIS / "final_events.csv"
    merged[keep_cols].to_csv(out_manifest, index=False)
    split[event_cols].to_csv(out_events, index=False)

    role_paths = []
    for role in ["expert_reference", "beginner_test", "expert_holdout"]:
        role_manifest = merged[merged["final_role"] == role][keep_cols].copy()
        role_path = ANALYSIS / f"final_{role}_skeleton_manifest.csv"
        role_manifest.to_csv(role_path, index=False)
        role_paths.append(role_path)
    return out_events, role_paths[0], role_paths[1], role_paths[2]


def rotation_score(compare: pd.DataFrame) -> dict[str, float]:
    out = {}
    subset = compare[compare["feature"].isin(ROTATION_FEATURES)]
    for trial_id, group in subset.groupby("trial_id"):
        z = group["z_score"].astype(float).to_numpy()
        out[str(trial_id)] = float(np.sqrt(np.nanmean(np.square(z))))
    return out


def run_direction_and_stats(expert: pd.DataFrame, beginner: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for action_type in sorted(expert["action_type"].unique()):
        e_action = expert[expert["action_type"] == action_type]
        b_action = beginner[beginner["action_type"] == action_type]
        for feature in PROFILE_FEATURES:
            e = e_action[feature].astype(float).to_numpy()
            b = b_action[feature].astype(float).to_numpy()
            _, p_mw = mannwhitneyu(b, e, alternative="two-sided")
            e_mean = float(np.nanmean(e))
            below = int(np.sum(b < e_mean))
            above = int(np.sum(b > e_mean))
            dominant = max(below, above)
            direction = "below_expert_mean" if below >= above else "above_expert_mean"
            rows.append(
                {
                    "action_type": action_type,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "expert_n": int(len(e)),
                    "beginner_n": int(len(b)),
                    "expert_mean": e_mean,
                    "beginner_mean": float(np.nanmean(b)),
                    "mann_whitney_u_p": float(p_mw),
                    "beginner_below_expert_mean": below,
                    "beginner_above_expert_mean": above,
                    "dominant_direction": direction,
                    "direction_consistency_p_one_sided": float(binomtest(dominant, len(b), 0.5, alternative="greater").pvalue),
                }
            )
    return pd.DataFrame(rows).sort_values(["action_type", "mann_whitney_u_p"])


def run_contact_perturbation(expert: pd.DataFrame, beginner: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    contact_relative = [
        "pelvis_fastest_minus_contact_frames",
        "shoulder_fastest_minus_contact_frames",
        "wrist_fastest_minus_contact_frames",
    ]
    ref = reference_table(expert)
    base_compare = compare_to_reference(beginner, ref)
    base_top = top_features(base_compare)
    rows = []
    for delta in [-2, -1, 1, 2]:
        perturbed = beginner.copy()
        for feature in contact_relative:
            perturbed[feature] = perturbed[feature].astype(float) - float(delta)
        comp = compare_to_reference(perturbed, ref)
        pert_top = top_features(comp)
        for sample in beginner.itertuples(index=False):
            trial_id = str(sample.trial_id)
            rows.append(
                {
                    "trial_id": trial_id,
                    "subject_id": sample.subject_id,
                    "action_type": sample.action_type,
                    "contact_delta_frames": delta,
                    "top1_stable": base_top[trial_id][0] == pert_top[trial_id][0],
                    "top3_overlap": top_overlap(base_top[trial_id], pert_top[trial_id]),
                    "baseline_top3": ";".join(base_top[trial_id]),
                    "perturbed_top3": ";".join(pert_top[trial_id]),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["action_type", "contact_delta_frames"])
        .agg(n=("trial_id", "count"), top1_stability_rate=("top1_stable", "mean"), mean_top3_overlap=("top3_overlap", "mean"))
        .reset_index()
    )
    return detail, summary


def run_leave_one_expert(expert: pd.DataFrame, beginner: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_top = top_features(compare_to_reference(beginner, reference_table(expert)))
    rows = []
    for held_subject in sorted(expert["subject_id"].unique()):
        ref = reference_table(expert[expert["subject_id"] != held_subject].copy())
        loo_top = top_features(compare_to_reference(beginner, ref))
        for sample in beginner.itertuples(index=False):
            trial_id = str(sample.trial_id)
            rows.append(
                {
                    "held_out_expert_subject": held_subject,
                    "trial_id": trial_id,
                    "subject_id": sample.subject_id,
                    "action_type": sample.action_type,
                    "top1_stable": base_top[trial_id][0] == loo_top[trial_id][0],
                    "top3_overlap": top_overlap(base_top[trial_id], loo_top[trial_id]),
                    "baseline_top3": ";".join(base_top[trial_id]),
                    "loo_top3": ";".join(loo_top[trial_id]),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["action_type", "held_out_expert_subject"])
        .agg(n=("trial_id", "count"), top1_stability_rate=("top1_stable", "mean"), mean_top3_overlap=("top3_overlap", "mean"))
        .reset_index()
    )
    return detail, summary


def max_frame_for_sequence(sequence_path: Path) -> int:
    return int(pd.read_csv(sequence_path, usecols=["frame"])["frame"].max())


def build_swing_end_events(delta: int, manifest_path: Path, base_events: Path, out_path: Path) -> None:
    manifest = pd.read_csv(manifest_path)
    events = pd.read_csv(base_events)
    events = events[events["trial_id"].astype(str).isin(set(manifest["trial_id"].astype(str)))].copy()
    max_frames = {}
    for row in manifest.itertuples(index=False):
        seq = Path(row.sequence_path)
        if not seq.is_absolute():
            seq = ROOT / seq
        max_frames[str(row.trial_id)] = max_frame_for_sequence(seq)
    adjusted = []
    for row in events.itertuples(index=False):
        end = int(row.swing_end_frame) + delta
        end = max(int(row.ball_contact_frame), end)
        end = min(max_frames[str(row.trial_id)], end)
        adjusted.append(end)
    events["swing_end_frame"] = adjusted
    events["label_quality"] = f"swing_end_delta_{delta:+d}"
    events.to_csv(out_path, index=False)


def run_swing_end_perturbation(expert: pd.DataFrame, beginner: pd.DataFrame, events: Path, beginner_manifest: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ref = reference_table(expert)
    base_compare = compare_to_reference(beginner, ref)
    base_top = top_features(base_compare)
    base_score = rotation_score(base_compare)
    rows = []
    tmp = TEMP / "swing_end"
    tmp.mkdir(parents=True, exist_ok=True)
    for delta in [-4, -2, 2, 4]:
        events_path = tmp / f"events_swing_end_delta_{delta:+d}.csv"
        out_dir = tmp / f"features_swing_end_delta_{delta:+d}"
        build_swing_end_events(delta, beginner_manifest, events, events_path)
        perturbed = process_manifest(beginner_manifest, out_dir, events_csv=events_path)
        comp = compare_to_reference(perturbed, ref)
        pert_top = top_features(comp)
        pert_score = rotation_score(comp)
        for sample in beginner.itertuples(index=False):
            trial_id = str(sample.trial_id)
            rows.append(
                {
                    "trial_id": trial_id,
                    "subject_id": sample.subject_id,
                    "action_type": sample.action_type,
                    "swing_end_delta_frames": delta,
                    "top1_stable": base_top[trial_id][0] == pert_top[trial_id][0],
                    "top3_overlap": top_overlap(base_top[trial_id], pert_top[trial_id]),
                    "baseline_top3": ";".join(base_top[trial_id]),
                    "perturbed_top3": ";".join(pert_top[trial_id]),
                    "baseline_rotation_range_score": base_score[trial_id],
                    "perturbed_rotation_range_score": pert_score[trial_id],
                    "abs_rotation_range_score_change": abs(pert_score[trial_id] - base_score[trial_id]),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["action_type", "swing_end_delta_frames"])
        .agg(
            n=("trial_id", "count"),
            top1_stability_rate=("top1_stable", "mean"),
            mean_top3_overlap=("top3_overlap", "mean"),
            median_abs_rotation_score_change=("abs_rotation_range_score_change", "median"),
        )
        .reset_index()
    )
    overall = (
        detail.groupby(["swing_end_delta_frames"])
        .agg(
            n=("trial_id", "count"),
            top1_stability_rate=("top1_stable", "mean"),
            mean_top3_overlap=("top3_overlap", "mean"),
            median_abs_rotation_score_change=("abs_rotation_range_score_change", "median"),
        )
        .reset_index()
    )
    return detail, summary, overall


def run_filter_sensitivity(events: Path, expert_manifest: Path, beginner_manifest: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profiles: dict[float, dict[str, list[str]]] = {}
    scores: dict[float, dict[str, float]] = {}
    cutoff_records = []
    tmp = TEMP / "filter_cutoff"
    tmp.mkdir(parents=True, exist_ok=True)
    for cutoff in [4.0, 5.0, 6.0]:
        tag = f"cutoff_{cutoff:.0f}hz"
        expert = process_manifest(expert_manifest, tmp / tag / "expert", cutoff_hz=cutoff, events_csv=events)
        beginner = process_manifest(beginner_manifest, tmp / tag / "beginner", cutoff_hz=cutoff, events_csv=events)
        comp = compare_to_reference(beginner, reference_table(expert))
        profiles[cutoff] = top_features(comp)
        scores[cutoff] = rotation_score(comp)
        cutoff_records.append(
            pd.concat([expert, beginner], ignore_index=True)[["trial_id", "subject_id", "action_type", "fps", "filter_cutoff_hz"]].assign(requested_cutoff_hz=cutoff)
        )
    rows = []
    baseline = profiles[6.0]
    baseline_scores = scores[6.0]
    for cutoff in [4.0, 5.0, 6.0]:
        for trial_id, base_top3 in baseline.items():
            candidate = profiles[cutoff][trial_id]
            rows.append(
                {
                    "trial_id": trial_id,
                    "requested_cutoff_hz": cutoff,
                    "top1_stable_vs_6hz": base_top3[0] == candidate[0],
                    "top3_overlap_vs_6hz": top_overlap(base_top3, candidate),
                    "baseline_top3_6hz": ";".join(base_top3),
                    "candidate_top3": ";".join(candidate),
                    "baseline_rotation_range_score_6hz": baseline_scores[trial_id],
                    "candidate_rotation_range_score": scores[cutoff][trial_id],
                    "abs_rotation_range_score_change": abs(baseline_scores[trial_id] - scores[cutoff][trial_id]),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("requested_cutoff_hz", sort=True)
        .agg(
            n=("trial_id", "count"),
            top1_stability_rate=("top1_stable_vs_6hz", "mean"),
            mean_top3_overlap=("top3_overlap_vs_6hz", "mean"),
            median_abs_rotation_score_change=("abs_rotation_range_score_change", "median"),
        )
        .reset_index()
    )
    effective = pd.concat(cutoff_records, ignore_index=True)
    return detail, summary, effective


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)

    split, expert, beginner, holdout = load_final_sets()
    events, expert_manifest, beginner_manifest, holdout_manifest = build_final_manifests(split)

    contact_detail, contact_summary = run_contact_perturbation(expert, beginner)
    loo_detail, loo_summary = run_leave_one_expert(expert, beginner)
    stats = run_direction_and_stats(expert, beginner)
    swing_detail, swing_summary, swing_overall = run_swing_end_perturbation(expert, beginner, events, beginner_manifest)
    filter_detail, filter_summary, filter_effective = run_filter_sensitivity(events, expert_manifest, beginner_manifest)

    outputs = {
        "contact_frame_perturbation_detail.csv": contact_detail,
        "contact_frame_perturbation_summary.csv": contact_summary,
        "leave_one_expert_out_detail.csv": loo_detail,
        "leave_one_expert_out_summary.csv": loo_summary,
        "exploratory_stat_tests.csv": stats,
        "swing_end_perturbation_detail.csv": swing_detail,
        "swing_end_perturbation_summary.csv": swing_summary,
        "swing_end_perturbation_overall.csv": swing_overall,
        "filter_cutoff_sensitivity_detail.csv": filter_detail,
        "filter_cutoff_sensitivity_summary.csv": filter_summary,
        "filter_cutoff_sensitivity_effective_cutoffs.csv": filter_effective,
    }
    for name, df in outputs.items():
        df.to_csv(RESULTS / name, index=False)

    report = {
        "final_split": {
            "expert_reference_subjects": int(expert["subject_id"].nunique()),
            "beginner_subjects": int(beginner["subject_id"].nunique()),
            "held_out_subjects": int(holdout["subject_id"].nunique()),
        },
        "contact": {
            "top1": float(contact_detail["top1_stable"].mean()),
            "top3": float(contact_detail["top3_overlap"].mean()),
        },
        "swing_end": {
            "top1": float(swing_detail["top1_stable"].mean()),
            "top3": float(swing_detail["top3_overlap"].mean()),
            "median_score_change": float(swing_detail["abs_rotation_range_score_change"].median()),
        },
        "filter_5hz": filter_summary[filter_summary["requested_cutoff_hz"] == 5.0].iloc[0].to_dict(),
        "filter_4hz": filter_summary[filter_summary["requested_cutoff_hz"] == 4.0].iloc[0].to_dict(),
        "loo": {
            "top1": float(loo_detail["top1_stable"].mean()),
            "top3": float(loo_detail["top3_overlap"].mean()),
        },
    }
    (RESULTS / "paper_sanity_experiments_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
