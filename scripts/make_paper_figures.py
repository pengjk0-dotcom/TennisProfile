#!/usr/bin/env python3
"""Generate paper-ready figures for the tennis skill profiling study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/thetis_groundstroke"
OUT = ROOT / "paper/figures"

PALETTE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "dark": "#111827",
}

def cohort_subject_n(df: pd.DataFrame, action_type: str | None = None) -> int:
    subset = df if action_type is None else df[df["action_type"] == action_type]
    return int(subset["subject_id"].nunique())


def cohort_trial_n(df: pd.DataFrame, action_type: str | None = None) -> int:
    subset = df if action_type is None else df[df["action_type"] == action_type]
    return int(subset["trial_id"].nunique())

FEATURE_LABELS = {
    "pelvis_fastest_minus_contact_frames": "Pelvis fastest time",
    "shoulder_fastest_minus_contact_frames": "Shoulder fastest time",
    "wrist_fastest_minus_contact_frames": "Wrist fastest time",
    "shoulder_minus_pelvis_fastest_frames": "Shoulder - pelvis time",
    "wrist_minus_shoulder_fastest_frames": "Wrist - shoulder time",
    "pelvis_rom": "Pelvis turn range",
    "trunk_rom": "Shoulder turn range",
    "separation_rom": "Shoulder relative-to-pelvis turn range",
    "active_wrist_peak_speed": "Wrist peak speed",
    "pelvis_peak_abs_velocity_dps": "Pelvis peak speed",
    "trunk_peak_abs_velocity_dps": "Shoulder peak speed",
}

def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def z_score(value: float, mean: float, std: float) -> float:
    if not np.isfinite(std) or std <= 1e-9:
        return np.nan
    return (value - mean) / std


def compare_features(features: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in features.iterrows():
        ref = reference[reference["action_type"] == row["action_type"]]
        for _, r in ref.iterrows():
            feature = r["feature"]
            if feature not in features.columns:
                continue
            value = float(row[feature])
            mean = float(r["mean"])
            std = float(r["std"])
            rows.append(
                {
                    "trial_id": row["trial_id"],
                    "subject_id": row["subject_id"],
                    "action_type": row["action_type"],
                    "feature": feature,
                    "label": FEATURE_LABELS.get(feature, feature),
                    "value": value,
                    "mean": mean,
                    "std": std,
                    "z": z_score(value, mean, std),
                    "min": float(r["min"]),
                    "max": float(r["max"]),
                    "outside": not (float(r["min"]) <= value <= float(r["max"])),
                }
            )
    return pd.DataFrame(rows)


def load_timeseries(trial_id: str, *directories: Path) -> pd.DataFrame:
    for directory in directories:
        path = directory / f"{trial_id}_timeseries.csv"
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(trial_id)


def load_marked_timeseries(trial_id: str) -> pd.DataFrame:
    """Load the reviewed timeseries for a trial from the known processed folders."""
    return load_timeseries(
        trial_id,
        DATA / "kinetic_chain_reannotation_v2_expert_reference",
        DATA / "kinetic_chain_reannotation_v2_beginner",
        DATA / "kinetic_chain_reannotation_v2_holdout",
        DATA / "kinetic_chain_expert_reference_pilot_marked",
        DATA / "kinetic_chain_expert_reference_batch2_marked",
        DATA / "kinetic_chain_next_annotation_batch_marked",
        DATA / "kinetic_chain_beginner_dev_batch2_marked",
        DATA / "kinetic_chain_smoke_beginner_marked",
        DATA / "kinetic_chain_smoke_beginner_balanced20",
        DATA / "kinetic_chain_holdout_marked",
        DATA / "kinetic_chain_annotation_expansion_v3",
    )


def existing_data_path(*names: str) -> Path:
    for name in names:
        path = DATA / name
        if path.exists():
            return path
    raise FileNotFoundError(names[0])


def figure_reference_ranges() -> None:
    expert = pd.read_csv(existing_data_path("final_expert_reference_features.csv", "expert_reference_12_subject_features.csv", "expert_reference_8_subject_features.csv"))
    beginner = pd.read_csv(existing_data_path("final_beginner_test_features.csv", "beginner_dev_all_10_subject_features.csv", "beginner_dev_all_6_subject_features.csv"))
    reference = pd.read_csv(existing_data_path("final_expert_reference_summary.csv", "expert_reference_12_subject_summary.csv", "expert_reference_8_subject_summary.csv"))

    selected = [
        "pelvis_rom",
        "trunk_rom",
        "separation_rom",
        "active_wrist_peak_speed",
        "pelvis_peak_abs_velocity_dps",
        "trunk_peak_abs_velocity_dps",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.1), sharex=True)
    for ax, action_type, title in [
        (
            axes[0],
            "forehand_flat",
            f"Forehand flat: expert n={cohort_subject_n(expert, 'forehand_flat')}, beginner n={cohort_subject_n(beginner, 'forehand_flat')}",
        ),
        (
            axes[1],
            "backhand",
            f"Backhand: expert n={cohort_subject_n(expert, 'backhand')}, beginner n={cohort_subject_n(beginner, 'backhand')}",
        ),
    ]:
        y_positions = np.arange(len(selected))[::-1]
        ax.axvspan(-1, 1, color=PALETTE["light_gray"], alpha=0.45, label="expert mean ± 1 SD")
        ax.axvline(0, color=PALETTE["dark"], linewidth=1)

        for yi, feature in zip(y_positions, selected):
            ref_row = reference[(reference["action_type"] == action_type) & (reference["feature"] == feature)].iloc[0]
            mean = float(ref_row["mean"])
            std = float(ref_row["std"])
            exp_values = expert.loc[expert["action_type"] == action_type, feature].astype(float)
            beg_values = beginner.loc[beginner["action_type"] == action_type, feature].astype(float)

            exp_z = [(v - mean) / std for v in exp_values]
            beg_z = [(v - mean) / std for v in beg_values]
            ax.hlines(yi, min(exp_z), max(exp_z), color=PALETTE["blue"], linewidth=4, alpha=0.45)
            ax.scatter(exp_z, np.full(len(exp_z), yi + 0.11), s=28, color=PALETTE["blue"], alpha=0.8)
            ax.scatter(beg_z, np.full(len(beg_z), yi - 0.11), s=34, color=PALETTE["vermillion"], alpha=0.82)

        ax.set_title(title)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([FEATURE_LABELS[f] for f in selected])
        ax.set_xlabel("z-score vs expert reference")
        ax.grid(axis="x", alpha=0.22)
        ax.set_xlim(-3.5, 3.5)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["blue"], markersize=7, label="expert trials defining reference"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["vermillion"], markersize=7, label="beginner deviations"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["light_gray"], alpha=0.45, label="expert mean ± 1 SD"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Expert reference ranges and beginner deviations", fontweight="bold", y=1.02)
    fig.text(
        0.5,
        0.045,
        "z = (value - expert mean) / expert SD. Rotation range: degrees; rotation speed: deg/s; wrist speed: HMR2 coordinate units/s, THETIS internal only.",
        ha="center",
        fontsize=8.5,
        color=PALETTE["gray"],
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.98))
    save_figure(fig, "fig2_expert_reference_ranges")


def figure_individual_deviation_profile() -> None:
    compare = pd.read_csv(
        existing_data_path(
            "final_beginner_vs_expert_reference.csv",
            "beginner_dev_all_10_subject_vs_expert_reference_12_subject.csv",
            "beginner_dev_all_6_subject_vs_expert_reference_8_subject.csv",
        )
    )
    trial_id = "p30_forehand_flat_s1"
    trial = compare[compare["trial_id"] == trial_id].copy()
    trial["label"] = trial["feature"].map(FEATURE_LABELS).fillna(trial["feature_name_zh"])
    trial = trial.sort_values("abs_z_score", ascending=True)

    colors = [PALETTE["vermillion"] if outside else PALETTE["sky"] for outside in trial["inside_reference_minmax"].map(lambda x: not x)]

    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    ax.axvspan(-1, 1, color=PALETTE["light_gray"], alpha=0.45)
    ax.axvline(0, color=PALETTE["dark"], linewidth=1)
    ax.barh(trial["label"], trial["z_score"], color=colors, alpha=0.9)
    ax.set_xlabel("feature deviation z-score vs expert forehand reference")
    ax.set_title("p30 forehand profile (z-scores)", fontweight="bold")
    ax.grid(axis="x", alpha=0.22)

    for y, (_, row) in enumerate(trial.iterrows()):
        x = float(row["z_score"])
        label = "outside" if not bool(row["inside_reference_minmax"]) else "inside"
        ax.text(
            x + (0.07 if x >= 0 else -0.07),
            y,
            label,
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8,
            color=PALETTE["dark"],
        )

    legend = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["vermillion"], label="outside expert min-max"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["sky"], label="inside expert min-max"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["light_gray"], alpha=0.45, label="expert mean ± 1 SD"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=True)
    ax.text(
        0.01,
        0.02,
        "Reference: final expert forehand set. Positive z = above expert mean.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=PALETTE["gray"],
        ha="left",
        va="bottom",
    )
    fig.tight_layout()
    save_figure(fig, "fig3_individual_deviation_profile_p30_forehand")


def figure_forehand_reference_to_profile() -> None:
    expert = pd.read_csv(existing_data_path("final_expert_reference_features.csv", "expert_reference_12_subject_features.csv", "expert_reference_8_subject_features.csv"))
    beginner = pd.read_csv(existing_data_path("final_beginner_test_features.csv", "beginner_dev_all_10_subject_features.csv", "beginner_dev_all_6_subject_features.csv"))
    reference = pd.read_csv(existing_data_path("final_expert_reference_summary.csv", "expert_reference_12_subject_summary.csv", "expert_reference_8_subject_summary.csv"))
    compare = pd.read_csv(
        existing_data_path(
            "final_beginner_vs_expert_reference.csv",
            "beginner_dev_all_10_subject_vs_expert_reference_12_subject.csv",
            "beginner_dev_all_6_subject_vs_expert_reference_8_subject.csv",
        )
    )

    group_features = [
        "pelvis_rom",
        "trunk_rom",
        "separation_rom",
        "active_wrist_peak_speed",
        "pelvis_peak_abs_velocity_dps",
        "trunk_peak_abs_velocity_dps",
    ]
    action_type = "forehand_flat"
    trial_id = "p30_forehand_flat_s1"
    xlim = (-4.6, 4.6)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 5.15),
        gridspec_kw={"width_ratios": [1.03, 1.07], "wspace": 0.22},
    )

    ax = axes[0]
    y_positions = np.arange(len(group_features))[::-1]
    ax.axvspan(-1, 1, color=PALETTE["light_gray"], alpha=0.45)
    ax.axvline(0, color=PALETTE["dark"], linewidth=1)
    for yi, feature in zip(y_positions, group_features):
        ref_row = reference[(reference["action_type"] == action_type) & (reference["feature"] == feature)].iloc[0]
        mean = float(ref_row["mean"])
        std = float(ref_row["std"])
        exp_values = expert.loc[expert["action_type"] == action_type, feature].astype(float)
        beg_values = beginner.loc[beginner["action_type"] == action_type, feature].astype(float)
        exp_z = [(v - mean) / std for v in exp_values]
        beg_z = [(v - mean) / std for v in beg_values]
        ax.hlines(yi, min(exp_z), max(exp_z), color=PALETTE["blue"], linewidth=4, alpha=0.45)
        ax.scatter(exp_z, np.full(len(exp_z), yi + 0.11), s=26, color=PALETTE["blue"], alpha=0.8)
        ax.scatter(beg_z, np.full(len(beg_z), yi - 0.11), s=32, color=PALETTE["vermillion"], alpha=0.82)
    ax.set_title(
        "A. Forehand reference distribution\n"
        f"expert n={cohort_subject_n(expert, action_type)}, beginner n={cohort_subject_n(beginner, action_type)}"
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([FEATURE_LABELS[f] for f in group_features])
    ax.set_xlabel("z-score vs expert forehand reference")
    ax.grid(axis="x", alpha=0.22)
    ax.set_xlim(*xlim)

    ax = axes[1]
    trial = compare[compare["trial_id"] == trial_id].copy()
    trial["label"] = trial["feature"].map(FEATURE_LABELS).fillna(trial["feature_name_zh"])
    trial = trial.sort_values("abs_z_score", ascending=True)
    colors = [PALETTE["vermillion"] if outside else PALETTE["sky"] for outside in trial["inside_reference_minmax"].map(lambda x: not x)]
    ax.axvspan(-1, 1, color=PALETTE["light_gray"], alpha=0.45)
    ax.axvline(0, color=PALETTE["dark"], linewidth=1)
    ax.barh(trial["label"], trial["z_score"], color=colors, alpha=0.9)
    ax.set_xlabel("z-score vs expert forehand reference")
    ax.set_title("B. p30 forehand stroke profile\ncomplete 11-feature output")
    ax.grid(axis="x", alpha=0.22)
    ax.set_xlim(*xlim)
    for y, (_, row) in enumerate(trial.iterrows()):
        x = float(row["z_score"])
        if abs(x) < 1.25:
            continue
        label = "outside" if not bool(row["inside_reference_minmax"]) else "inside"
        ax.text(
            x + (0.08 if x >= 0 else -0.08),
            y,
            label,
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8,
            color=PALETTE["dark"],
        )

    fig.suptitle("Forehand reference distribution and individual movement profile", fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save_figure(fig, "fig2_forehand_reference_to_profile")


def figure_start_review_effect() -> None:
    reference = pd.read_csv(existing_data_path("final_expert_reference_summary.csv", "expert_reference_12_subject_summary.csv", "expert_reference_8_subject_summary.csv"))
    before = pd.read_csv(DATA / "kinetic_chain_event_review_example_before/kinetic_chain_features.csv")
    after = pd.read_csv(DATA / "kinetic_chain_reannotation_v2_beginner/kinetic_chain_features.csv")
    trial_id = "p8_backhand_s1"
    before_trial = before[before["trial_id"] == trial_id].copy()
    after_trial = after[after["trial_id"] == trial_id].copy()
    before_cmp = compare_features(before_trial, reference)
    after_cmp = compare_features(after_trial, reference)

    ts_after = load_timeseries(trial_id, DATA / "kinetic_chain_reannotation_v2_beginner")
    old_start = int(before_trial["swing_start_frame"].iloc[0] - before_trial["ball_contact_frame"].iloc[0])
    reviewed_start = int(after_trial["swing_start_frame"].iloc[0] - after_trial["ball_contact_frame"].iloc[0])
    features = [
        "shoulder_fastest_minus_contact_frames",
        "wrist_minus_shoulder_fastest_frames",
        "shoulder_minus_pelvis_fastest_frames",
        "pelvis_fastest_minus_contact_frames",
    ]
    labels = [FEATURE_LABELS[f] for f in features]
    before_z = [float(before_cmp[before_cmp["feature"] == f]["z"].iloc[0]) for f in features]
    after_z = [float(after_cmp[after_cmp["feature"] == f]["z"].iloc[0]) for f in features]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.1, 1.0]})

    x = ts_after["frame_from_ball_contact"]
    axes[0].plot(x, ts_after["pelvis_velocity_dps"].abs(), label="pelvis speed", color=PALETTE["blue"], linewidth=2)
    axes[0].plot(x, ts_after["trunk_velocity_dps"].abs(), label="shoulder speed", color=PALETTE["orange"], linewidth=2)
    axes[0].plot(x, ts_after["active_wrist_speed"] * 35, label="wrist speed ×35", color=PALETTE["pink"], linewidth=2)
    axes[0].axvline(old_start, color=PALETTE["gray"], linestyle="--", linewidth=1.5, label="old start")
    axes[0].axvline(reviewed_start, color=PALETTE["green"], linestyle="--", linewidth=1.5, label="reviewed start")
    axes[0].axvline(0, color=PALETTE["vermillion"], linestyle="-", linewidth=1.5, label="ball contact")
    axes[0].set_title("p8 backhand speed traces")
    axes[0].set_xlabel("frames from ball contact (0 = contact)")
    axes[0].set_ylabel("speed (pelvis/shoulder deg/s; wrist ×35)")
    axes[0].grid(alpha=0.22)
    axes[0].legend(loc="upper right", frameon=True)

    y = np.arange(len(features))
    height = 0.36
    axes[1].axvspan(-1, 1, color=PALETTE["light_gray"], alpha=0.45)
    axes[1].axvline(0, color=PALETTE["dark"], linewidth=1)
    axes[1].barh(y + height / 2, before_z, height=height, color=PALETTE["vermillion"], label="before review")
    axes[1].barh(y - height / 2, after_z, height=height, color=PALETTE["green"], label="after review")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("feature deviation z-score vs expert backhand reference")
    axes[1].set_title("Profile change after event review")
    axes[1].grid(axis="x", alpha=0.22)
    axes[1].legend(loc="lower right", frameon=True)

    fig.suptitle("Event review for p8 backhand (final expert reference)", fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, "fig4_p8_start_review_effect")


def angle_change_deg(values: pd.Series) -> np.ndarray:
    radians = np.unwrap(np.deg2rad(values.astype(float).to_numpy()))
    return np.rad2deg(radians - radians[0])


def phase_bins_for_trial(ts: pd.DataFrame) -> pd.Series:
    contact = ts["frame_from_ball_contact"].astype(float)
    pre = abs(float(contact.min())) or 1.0
    post = abs(float(contact.max())) or 1.0
    phase = np.where(contact < 0, contact / pre, contact / post)
    bins = [-1.01, -0.60, -0.20, 0.20, 0.60, 1.01]
    labels = [
        "Early\npre-contact",
        "Late\npre-contact",
        "Contact\nwindow",
        "Early\nfollow-through",
        "Late\nfollow-through",
    ]
    return pd.cut(phase, bins=bins, labels=labels, include_lowest=True)


def trial_phase_feature_table(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, meta in features.iterrows():
        trial_id = str(meta["trial_id"])
        ts = load_marked_timeseries(trial_id).copy()
        if "frame_from_ball_contact" not in ts.columns:
            ts["frame_from_ball_contact"] = ts["frame"] - int(meta["ball_contact_frame"])
        ts["phase_bin"] = phase_bins_for_trial(ts)
        ts["Pelvis speed"] = ts["pelvis_velocity_dps"].abs()
        ts["Shoulder speed"] = ts["trunk_velocity_dps"].abs()
        ts["Wrist speed"] = ts["active_wrist_speed"]
        ts["Pelvis turn"] = np.abs(angle_change_deg(ts["pelvis_angle_deg"]))
        ts["Shoulder turn"] = np.abs(angle_change_deg(ts["trunk_angle_deg"]))
        ts["Shoulder relative-to-pelvis turn"] = np.abs(angle_change_deg(ts["trunk_pelvis_separation_deg"]))
        for phase, group in ts.groupby("phase_bin", observed=False):
            if pd.isna(phase) or group.empty:
                continue
            for name in [
                "Pelvis speed",
                "Shoulder speed",
                "Wrist speed",
                "Pelvis turn",
                "Shoulder turn",
                "Shoulder relative-to-pelvis turn",
            ]:
                rows.append(
                    {
                        "trial_id": trial_id,
                        "subject_id": meta["subject_id"],
                        "skill_level": meta["skill_level"],
                        "action_type": meta["action_type"],
                        "phase": str(phase),
                        "feature": name,
                        "value": float(group[name].mean()),
                    }
                )
    return pd.DataFrame(rows)


def figure_joint_time_deviation_map() -> None:
    expert = pd.read_csv(existing_data_path("final_expert_reference_features.csv", "expert_reference_12_subject_features.csv", "expert_reference_8_subject_features.csv"))
    beginner = pd.read_csv(existing_data_path("final_beginner_test_features.csv", "beginner_dev_all_10_subject_features.csv", "beginner_dev_all_6_subject_features.csv"))
    phase_values = pd.concat(
        [trial_phase_feature_table(expert), trial_phase_feature_table(beginner)],
        ignore_index=True,
    )

    reference = (
        phase_values[phase_values["skill_level"] == "expert"]
        .groupby(["action_type", "phase", "feature"], as_index=False)
        .agg(reference_mean=("value", "mean"), reference_std=("value", "std"))
    )
    beginner_values = phase_values[phase_values["skill_level"] == "beginner"].merge(
        reference, on=["action_type", "phase", "feature"], how="left"
    )
    beginner_values["z_score"] = (beginner_values["value"] - beginner_values["reference_mean"]) / beginner_values[
        "reference_std"
    ].replace(0, np.nan)
    summary = (
        beginner_values.groupby(["action_type", "phase", "feature"], as_index=False)
        .agg(mean_z_score=("z_score", "mean"), mean_abs_z_score=("z_score", lambda x: float(np.nanmean(np.abs(x)))))
    )
    results_path = ROOT / "paper/results/joint_time_deviation_map.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_path, index=False)
    print(f"Wrote {results_path}")

    action_titles = [("forehand_flat", "Forehand"), ("backhand", "Backhand")]
    phase_order = [
        "Early\npre-contact",
        "Late\npre-contact",
        "Contact\nwindow",
        "Early\nfollow-through",
        "Late\nfollow-through",
    ]
    phase_short_labels = [
        "Early\npre",
        "Late\npre",
        "Contact",
        "Early\nfollow",
        "Late\nfollow",
    ]
    feature_order = [
        "Pelvis speed",
        "Shoulder speed",
        "Wrist speed",
        "Pelvis turn",
        "Shoulder turn",
        "Shoulder relative-to-pelvis turn",
    ]

    # Keep the native figure close to the manuscript column width so labels
    # remain legible after LaTeX scales the vector graphic.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5), sharey=True)
    image = None
    for ax, (action_type, title) in zip(axes, action_titles):
        panel = summary[summary["action_type"] == action_type]
        matrix = (
            panel.pivot(index="feature", columns="phase", values="mean_z_score")
            .reindex(index=feature_order, columns=phase_order)
            .to_numpy()
        )
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(phase_order)))
        ax.set_xticklabels(phase_short_labels)
        ax.set_yticks(np.arange(len(feature_order)))
        ax.set_yticklabels(
            [
                "Pelvis speed",
                "Shoulder speed",
                "Wrist speed",
                "Pelvis turn",
                "Shoulder turn",
                "Shoulder relative\nto pelvis turn",
            ]
        )
        ax.set_xlabel("stroke phase")
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                value = matrix[y, x]
                if np.isfinite(value):
                    ax.text(x, y, f"{value:+.1f}", ha="center", va="center", fontsize=9.5, color=PALETTE["dark"])
        ax.set_xticks(np.arange(-0.5, len(phase_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(feature_order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
    axes[0].set_ylabel("movement feature")
    fig.subplots_adjust(left=0.19, right=0.98, top=0.82, bottom=0.30, wspace=0.08)
    if image is not None:
        cax = fig.add_axes([0.30, 0.10, 0.40, 0.055])
        cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
        cbar.set_label("beginner deviation from expert reference (mean z-score)")
    fig.suptitle("Joint-Time Deviation Map", fontweight="bold", y=0.96)
    save_figure(fig, "fig5_joint_time_deviation_map")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
    selected_rows = []
    for ax, (action_type, title) in zip(axes, action_titles):
        panel = summary[summary["action_type"] == action_type].copy()
        panel["abs_mean_z"] = panel["mean_z_score"].abs()
        panel = panel.sort_values("abs_mean_z", ascending=False).head(4)
        panel = panel.sort_values("mean_z_score")
        selected_rows.append(panel.assign(panel_title=title))

        labels = [f"{row.feature}\n{row.phase.replace(chr(10), ' ')}" for row in panel.itertuples()]
        values = panel["mean_z_score"].to_numpy()
        colors = [PALETTE["vermillion"] if v > 0 else PALETTE["blue"] for v in values]
        y = np.arange(len(panel))

        ax.axvline(0, color=PALETTE["dark"], linewidth=1)
        ax.barh(y, values, color=colors, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_title(f"{title}: largest phase-level deviations")
        ax.set_xlabel("beginner mean z-score vs expert reference")
        ax.grid(axis="x", alpha=0.22)
        ax.set_xlim(-1.8, 1.8)
        for yi, v in zip(y, values):
            ax.text(
                v + (0.05 if v >= 0 else -0.05),
                yi,
                f"{v:+.1f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8.5,
                color=PALETTE["dark"],
            )

    selected = pd.concat(selected_rows, ignore_index=True)
    selected_path = ROOT / "paper/results/key_phase_deviation_summary.csv"
    selected.to_csv(selected_path, index=False)
    print(f"Wrote {selected_path}")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["vermillion"], label="beginner above expert reference"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["blue"], label="beginner below expert reference"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Key phase-level deviations from expert reference", fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    save_figure(fig, "fig5_key_phase_deviation_summary")


def main() -> int:
    configure_matplotlib()
    figure_forehand_reference_to_profile()
    figure_start_review_effect()
    figure_joint_time_deviation_map()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
