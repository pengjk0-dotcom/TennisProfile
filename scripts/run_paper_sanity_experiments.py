#!/usr/bin/env python3
"""Run derived-table sanity experiments for TennisProfile.

These checks operate on the released paper-facing feature table:

1. ball-contact frame perturbation (+/- 1 and +/- 2 frames);
2. leave-one-expert-out reference stability;
3. exploratory group statistics;
4. an auxiliary subject-level classifier check retained for provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

PROFILE_FEATURES = [
    "pelvis_fastest_minus_contact_frames",
    "shoulder_fastest_minus_contact_frames",
    "wrist_fastest_minus_contact_frames",
    "shoulder_minus_pelvis_fastest_frames",
    "wrist_minus_shoulder_fastest_frames",
    "pelvis_rom",
    "trunk_rom",
    "separation_rom",
    "active_wrist_peak_speed",
    "pelvis_peak_abs_velocity_dps",
    "trunk_peak_abs_velocity_dps",
]

CONTACT_RELATIVE_FEATURES = [
    "pelvis_fastest_minus_contact_frames",
    "shoulder_fastest_minus_contact_frames",
    "wrist_fastest_minus_contact_frames",
]

FEATURE_LABELS = {
    "pelvis_fastest_minus_contact_frames": "Pelvis fastest-motion time",
    "shoulder_fastest_minus_contact_frames": "Shoulder fastest-motion time",
    "wrist_fastest_minus_contact_frames": "Wrist fastest-motion time",
    "shoulder_minus_pelvis_fastest_frames": "Shoulder-pelvis fastest-motion gap",
    "wrist_minus_shoulder_fastest_frames": "Wrist-shoulder fastest-motion gap",
    "pelvis_rom": "Pelvis rotation range",
    "trunk_rom": "Shoulder rotation range",
    "separation_rom": "Shoulder-vs-pelvis rotation range",
    "active_wrist_peak_speed": "Wrist peak speed",
    "pelvis_peak_abs_velocity_dps": "Pelvis peak rotation speed",
    "trunk_peak_abs_velocity_dps": "Shoulder peak rotation speed",
}


def load_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_df = pd.read_csv(DATA / "final_all_assigned_features.csv")
    expert = all_df[all_df["final_role"] == "expert_reference"].copy()
    beginner = all_df[all_df["final_role"] == "beginner_test"].copy()
    return expert, beginner, all_df


def reference_table(expert: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for action_type, group in expert.groupby("action_type"):
        for feature in PROFILE_FEATURES:
            values = group[feature].astype(float)
            rows.append(
                {
                    "action_type": action_type,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "n": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def compare_to_reference(samples: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, sample in samples.iterrows():
        action_ref = ref[ref["action_type"] == sample["action_type"]]
        for _, r in action_ref.iterrows():
            feature = r["feature"]
            value = float(sample[feature])
            std = float(r["std"])
            z = np.nan if not np.isfinite(std) or std <= 1e-12 else (value - float(r["mean"])) / std
            rows.append(
                {
                    "trial_id": sample["trial_id"],
                    "subject_id": sample["subject_id"],
                    "skill_level": sample["skill_level"],
                    "action_type": sample["action_type"],
                    "feature": feature,
                    "feature_label": r["feature_label"],
                    "value": value,
                    "reference_mean": float(r["mean"]),
                    "reference_std": std,
                    "reference_min": float(r["min"]),
                    "reference_max": float(r["max"]),
                    "z_score": z,
                    "abs_z_score": abs(z) if np.isfinite(z) else np.nan,
                    "outside_reference_minmax": not (float(r["min"]) <= value <= float(r["max"])),
                }
            )
    return pd.DataFrame(rows)


def top_features(compare: pd.DataFrame, k: int = 3) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for trial_id, group in compare.groupby("trial_id"):
        ranked = group.sort_values("abs_z_score", ascending=False)
        out[trial_id] = ranked["feature"].head(k).tolist()
    return out


def top_overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return np.nan
    return len(set(a).intersection(b)) / min(len(a), len(b))


def run_contact_perturbation(expert: pd.DataFrame, beginner: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = reference_table(expert)
    base_compare = compare_to_reference(beginner, ref)
    base_top = top_features(base_compare)
    rows = []
    for delta in [-2, -1, 1, 2]:
        perturbed = beginner.copy()
        for feature in CONTACT_RELATIVE_FEATURES:
            perturbed[feature] = perturbed[feature].astype(float) - float(delta)
        perturbed["ball_contact_perturbation_frames"] = delta
        comp = compare_to_reference(perturbed, ref)
        pert_top = top_features(comp)
        for _, sample in beginner.iterrows():
            trial_id = sample["trial_id"]
            overlap = top_overlap(base_top[trial_id], pert_top[trial_id])
            rows.append(
                {
                    "trial_id": trial_id,
                    "subject_id": sample["subject_id"],
                    "action_type": sample["action_type"],
                    "contact_delta_frames": delta,
                    "top1_stable": base_top[trial_id][0] == pert_top[trial_id][0],
                    "top3_overlap": overlap,
                    "baseline_top3": ";".join(base_top[trial_id]),
                    "perturbed_top3": ";".join(pert_top[trial_id]),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["action_type", "contact_delta_frames"])
        .agg(
            n=("trial_id", "count"),
            top1_stability_rate=("top1_stable", "mean"),
            mean_top3_overlap=("top3_overlap", "mean"),
        )
        .reset_index()
    )
    return detail, summary


def run_leave_one_expert(expert: pd.DataFrame, beginner: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_ref = reference_table(expert)
    base_top = top_features(compare_to_reference(beginner, full_ref))
    rows = []
    for held_subject in sorted(expert["subject_id"].unique()):
        sub_expert = expert[expert["subject_id"] != held_subject].copy()
        ref = reference_table(sub_expert)
        loo_top = top_features(compare_to_reference(beginner, ref))
        for _, sample in beginner.iterrows():
            trial_id = sample["trial_id"]
            rows.append(
                {
                    "held_out_expert_subject": held_subject,
                    "trial_id": trial_id,
                    "subject_id": sample["subject_id"],
                    "action_type": sample["action_type"],
                    "top1_stable": base_top[trial_id][0] == loo_top[trial_id][0],
                    "top3_overlap": top_overlap(base_top[trial_id], loo_top[trial_id]),
                    "baseline_top3": ";".join(base_top[trial_id]),
                    "loo_top3": ";".join(loo_top[trial_id]),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["action_type", "held_out_expert_subject"])
        .agg(
            n=("trial_id", "count"),
            top1_stability_rate=("top1_stable", "mean"),
            mean_top3_overlap=("top3_overlap", "mean"),
        )
        .reset_index()
    )
    return detail, summary


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = sum(float(a > b) for a in x for b in y)
    lt = sum(float(a < b) for a in x for b in y)
    return (gt - lt) / (len(x) * len(y))


def run_exploratory_stats(expert: pd.DataFrame, beginner: pd.DataFrame) -> pd.DataFrame:
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
            same_direction = max(below, above)
            direction = "below_expert_mean" if below >= above else "above_expert_mean"
            p_direction = binomtest(same_direction, len(b), 0.5, alternative="greater").pvalue
            rows.append(
                {
                    "action_type": action_type,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "expert_n": len(e),
                    "beginner_n": len(b),
                    "expert_mean": float(np.nanmean(e)),
                    "beginner_mean": float(np.nanmean(b)),
                    "mann_whitney_u_p": float(p_mw),
                    "cliffs_delta_beginner_vs_expert": cliffs_delta(b, e),
                    "beginner_below_expert_mean": below,
                    "beginner_above_expert_mean": above,
                    "dominant_direction": direction,
                    "direction_consistency_p_one_sided": float(p_direction),
                }
            )
    return pd.DataFrame(rows).sort_values(["action_type", "mann_whitney_u_p"])


@dataclass
class ClassificationResult:
    model: str
    row_accuracy: float
    subject_accuracy: float
    n_rows: int
    n_subjects: int


def run_classifier_sanity(all_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        from sklearn.neighbors import NearestCentroid
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.tree import DecisionTreeClassifier
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("scikit-learn is required for classifier sanity check") from exc

    feature_cols = PROFILE_FEATURES + ["action_type"]
    model_df = all_df[["trial_id", "subject_id", "skill_level"] + feature_cols].dropna(
        subset=["subject_id", "skill_level"]
    )
    x = model_df[feature_cols]
    y = model_df["skill_level"].map({"beginner": 0, "expert": 1}).astype(int)
    subjects = model_df["subject_id"].to_numpy()
    unique_subjects = sorted(model_df["subject_id"].unique())

    numeric_features = PROFILE_FEATURES
    categorical_features = ["action_type"]
    preprocess_scaled = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    preprocess_tree = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models = {
        "nearest_centroid": Pipeline([("pre", preprocess_scaled), ("clf", NearestCentroid())]),
        "logistic_regression_l2": Pipeline(
            [
                ("pre", preprocess_scaled),
                ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=13)),
            ]
        ),
        "decision_tree_depth2": Pipeline(
            [("pre", preprocess_tree), ("clf", DecisionTreeClassifier(max_depth=2, random_state=13))]
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost_depth1_optional"] = Pipeline(
            [
                ("pre", preprocess_tree),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=20,
                        max_depth=1,
                        learning_rate=0.1,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        eval_metric="logloss",
                        random_state=13,
                    ),
                ),
            ]
        )
    except Exception:
        pass

    pred_rows = []
    summary_rows = []
    for model_name, model in models.items():
        row_true = []
        row_pred = []
        subject_true = []
        subject_pred = []
        for test_subject in unique_subjects:
            train_mask = subjects != test_subject
            test_mask = subjects == test_subject
            model.fit(x.loc[train_mask], y.loc[train_mask])
            pred = model.predict(x.loc[test_mask])
            true = y.loc[test_mask].to_numpy()
            row_true.extend(true.tolist())
            row_pred.extend(pred.tolist())
            votes = int(np.round(np.mean(pred)))
            subject_pred.append(votes)
            subject_true.append(int(np.round(np.mean(true))))
            for trial_id, t, p in zip(model_df.loc[test_mask, "trial_id"], true, pred):
                pred_rows.append(
                    {
                        "model": model_name,
                        "held_out_subject": test_subject,
                        "trial_id": trial_id,
                        "true_label": "expert" if t == 1 else "beginner",
                        "predicted_label": "expert" if p == 1 else "beginner",
                        "correct": bool(t == p),
                    }
                )
        summary_rows.append(
            {
                "model": model_name,
                "row_accuracy": float(accuracy_score(row_true, row_pred)),
                "row_balanced_accuracy": float(balanced_accuracy_score(row_true, row_pred)),
                "row_macro_f1": float(f1_score(row_true, row_pred, average="macro")),
                "subject_accuracy": float(accuracy_score(subject_true, subject_pred)),
                "subject_balanced_accuracy": float(balanced_accuracy_score(subject_true, subject_pred)),
                "n_rows": len(row_true),
                "n_subjects": len(subject_true),
            }
        )

    return pd.DataFrame(pred_rows), pd.DataFrame(summary_rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    expert, beginner, all_df = load_features()

    contact_detail, contact_summary = run_contact_perturbation(expert, beginner)
    loo_detail, loo_summary = run_leave_one_expert(expert, beginner)
    stats = run_exploratory_stats(expert, beginner)
    cls_detail, cls_summary = run_classifier_sanity(all_df)

    contact_detail.to_csv(OUT / "contact_frame_perturbation_detail.csv", index=False)
    contact_summary.to_csv(OUT / "contact_frame_perturbation_summary.csv", index=False)
    loo_detail.to_csv(OUT / "leave_one_expert_out_detail.csv", index=False)
    loo_summary.to_csv(OUT / "leave_one_expert_out_summary.csv", index=False)
    stats.to_csv(OUT / "exploratory_stat_tests.csv", index=False)
    cls_detail.to_csv(OUT / "skill_classifier_loso_detail.csv", index=False)
    cls_summary.to_csv(OUT / "skill_classifier_loso_summary.csv", index=False)

    summary = {
        "contact_perturbation": {
            "overall_top1_stability_rate": float(contact_detail["top1_stable"].mean()),
            "overall_mean_top3_overlap": float(contact_detail["top3_overlap"].mean()),
        },
        "leave_one_expert_out": {
            "overall_top1_stability_rate": float(loo_detail["top1_stable"].mean()),
            "overall_mean_top3_overlap": float(loo_detail["top3_overlap"].mean()),
        },
        "best_classifier_by_subject_balanced_accuracy": cls_summary.sort_values(
            "subject_balanced_accuracy", ascending=False
        ).iloc[0].to_dict(),
    }
    (OUT / "paper_sanity_experiments_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote results to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
