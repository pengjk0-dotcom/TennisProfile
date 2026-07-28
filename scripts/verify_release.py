#!/usr/bin/env python3
"""Verify the self-contained portion of the TennisProfile release."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

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

FORBIDDEN_SUFFIXES = {
    ".avi",
    ".ckpt",
    ".mkv",
    ".mov",
    ".mp4",
    ".onnx",
    ".pth",
    ".pt",
    ".webm",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    split = read_csv(DATA / "final_master_split.csv")
    events = read_csv(DATA / "final_events.csv")
    features = read_csv(DATA / "final_all_assigned_features.csv")

    require(len(split) == 72, f"expected 72 split rows, found {len(split)}")
    require(len(events) == 72, f"expected 72 event rows, found {len(events)}")
    require(len(features) == 72, f"expected 72 feature rows, found {len(features)}")

    subjects_by_role: dict[str, set[str]] = defaultdict(set)
    role_action_counts: Counter[tuple[str, str]] = Counter()
    for row in split:
        subjects_by_role[row["final_role"]].add(row["subject_id"])
        role_action_counts[(row["final_role"], row["action_type"])] += 1
        start = int(row["swing_start_frame"])
        contact = int(row["ball_contact_frame"])
        end = int(row["swing_end_frame"])
        require(start <= contact <= end, f"invalid event order for {row['trial_id']}")

    expected_subjects = {
        "expert_reference": 18,
        "expert_holdout": 3,
        "beginner_test": 15,
    }
    observed_subjects = {
        role: len(subjects) for role, subjects in subjects_by_role.items()
    }
    require(
        observed_subjects == expected_subjects,
        f"unexpected role counts: {observed_subjects}",
    )

    for role, n_subjects in expected_subjects.items():
        for action in ("backhand", "forehand_flat"):
            require(
                role_action_counts[(role, action)] == n_subjects,
                f"unexpected {role}/{action} trial count",
            )

    split_ids = {row["trial_id"] for row in split}
    require(split_ids == {row["trial_id"] for row in events}, "event IDs differ")
    require(split_ids == {row["trial_id"] for row in features}, "feature IDs differ")

    for row in features:
        for feature in PROFILE_FEATURES:
            value = float(row[feature])
            require(math.isfinite(value), f"non-finite {feature} in {row['trial_id']}")

    forbidden = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    require(not forbidden, f"forbidden source assets found: {forbidden}")

    print("TennisProfile release verification passed.")
    print("  subjects: 36 (18 reference, 3 held-out expert, 15 beginner)")
    print("  strokes: 72 (one forehand and one backhand per subject)")
    print("  event ordering, table alignment, and feature finiteness: valid")
    print("  raw media and model-weight files: absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
