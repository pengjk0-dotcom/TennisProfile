#!/usr/bin/env python3
"""Build the TennisProfile overview figure.

The figure follows the visual logic used in sports motion-capture papers:
RGB evidence, recovered mesh overlay, clean mesh rendering, and the resulting
profile output are shown in one compact flow.  It uses the local HMR2 outputs
and the official 4D-Humans renderer.

Run with:

    PYTHONPATH=external/4D-Humans ./.micromamba/envs/4dhumans/bin/python \
        scripts/make_tennisprofile_overview_figure.py
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hmr2.configs import get_config
from hmr2.models.smpl_wrapper import SMPL
from hmr2.utils.renderer import Renderer, cam_crop_to_full


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/thetis_groundstroke"
ROSTER = ROOT / "reproducibility/study_roster.csv"
OUT = ROOT / "paper/figures"
MODEL_CONFIG = Path.home() / ".cache/4DHumans/logs/train/multiruns/hmr2/0/model_config.yaml"

TRIALS = [
    ("Expert reference", "p34_forehand_flat_s1"),
    ("Beginner query", "p5_forehand_flat_s1"),
]
EVENTS = [
    ("Start", "swing_start_frame"),
    ("Contact", "ball_contact_frame"),
    ("End", "swing_end_frame"),
]
PROFILE_TRIAL = "p5_forehand_flat_s1"

FEATURE_LABELS = {
    "pelvis_fastest_minus_contact_frames": "Pelvis timing",
    "shoulder_fastest_minus_contact_frames": "Shoulder timing",
    "wrist_fastest_minus_contact_frames": "Wrist timing",
    "shoulder_minus_pelvis_fastest_frames": "Shoulder-pelvis gap",
    "wrist_minus_shoulder_fastest_frames": "Wrist-shoulder gap",
    "pelvis_peak_abs_velocity_dps": "Pelvis angular velocity",
    "trunk_peak_abs_velocity_dps": "Shoulder-line angular velocity",
    "active_wrist_peak_speed": "Higher-speed wrist speed",
    "pelvis_rom": "Pelvis range",
    "trunk_rom": "Shoulder-line range",
    "separation_rom": "Shoulder-pelvis range",
}


def find_first(pattern: str) -> Path:
    matches = sorted(DATA.glob(pattern), key=lambda p: ("smoke" in str(p), str(p)))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


def load_roster_events() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with ROSTER.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["trial_id"]] = {
                key: int(row[key]) for _, key in EVENTS
            }
    return out


def load_prediction(trial_id: str, frame: int) -> dict[str, np.ndarray]:
    path = find_first(f"*hmr2*/npz/{trial_id}_f{frame:06d}_p00.npz")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_frame(trial_id: str, frame: int) -> np.ndarray:
    path = find_first(f"*_frames/{trial_id}/frame_{frame:06d}.jpg")
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.float32)[:, :, ::-1] / 255.0


def smpl_vertices(smpl: SMPL, prediction: dict[str, np.ndarray]) -> np.ndarray:
    with torch.no_grad():
        output = smpl(
            global_orient=torch.tensor(prediction["smpl_global_orient"]).reshape(1, 1, 3, 3).float(),
            body_pose=torch.tensor(prediction["smpl_body_pose"]).reshape(1, 23, 3, 3).float(),
            betas=torch.tensor(prediction["smpl_betas"]).reshape(1, 10).float(),
            pose2rot=False,
        )
    return output.vertices[0].detach().cpu().numpy()


def full_frame_camera(config, prediction: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    img_size = torch.tensor(prediction["img_size"]).reshape(1, 2).float()
    scaled_focal_length = config.EXTRA.FOCAL_LENGTH / config.MODEL.IMAGE_SIZE * img_size.max()
    cam_t = cam_crop_to_full(
        torch.tensor(prediction["pred_cam"]).reshape(1, 3).float(),
        torch.tensor(prediction["box_center"]).reshape(1, 2).float(),
        torch.tensor([float(prediction["box_size"])]).float(),
        img_size,
        scaled_focal_length,
    )
    return cam_t.detach().cpu().numpy()[0], float(scaled_focal_length)


def render_rgba(config, renderer: Renderer, smpl: SMPL, trial_id: str, frame: int) -> tuple[np.ndarray, np.ndarray]:
    os.environ.pop("PYOPENGL_PLATFORM", None)
    prediction = load_prediction(trial_id, frame)
    vertices = smpl_vertices(smpl, prediction)
    cam_t, focal_length = full_frame_camera(config, prediction)
    with contextlib.redirect_stdout(io.StringIO()):
        rgba = renderer.render_rgba_multiple(
            [vertices],
            [cam_t],
            render_res=prediction["img_size"].astype(int),
            focal_length=focal_length,
            mesh_base_color=(0.12, 0.70, 0.88),
            scene_bg_color=(1, 1, 1),
        )
    return rgba, prediction["img_size"].astype(int)


def crop_box(alpha_masks: list[np.ndarray], pad: int = 54) -> tuple[int, int, int, int]:
    mask = np.any(np.stack([m > 0.05 for m in alpha_masks], axis=0), axis=0)
    ys, xs = np.where(mask)
    height, width = mask.shape
    if len(xs) == 0:
        return 0, 0, width, height
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(width, int(xs.max()) + pad)
    y1 = min(height, int(ys.max()) + pad)
    return x0, y0, x1, y1


def player_crop_box(
    trial_id: str,
    frame: int,
    image_shape: tuple[int, int, int],
    scale: float = 1.95,
    aspect: float = 1.0,
) -> tuple[int, int, int, int]:
    """Crop around the recovered player, with extra room for the racket."""
    prediction = load_prediction(trial_id, frame)
    center = np.asarray(prediction["box_center"]).reshape(-1)[:2]
    size = float(prediction["box_size"])
    height, width = image_shape[:2]
    crop_w = size * scale
    crop_h = size * scale
    if crop_w / crop_h < aspect:
        crop_w = crop_h * aspect
    else:
        crop_h = crop_w / aspect

    cx, cy = center
    x0 = int(round(cx - crop_w / 2))
    x1 = int(round(cx + crop_w / 2))
    y0 = int(round(cy - crop_h / 2))
    y1 = int(round(cy + crop_h / 2))

    if x0 < 0:
        x1 = min(width, x1 - x0)
        x0 = 0
    if y0 < 0:
        y1 = min(height, y1 - y0)
        y0 = 0
    if x1 > width:
        x0 = max(0, x0 - (x1 - width))
        x1 = width
    if y1 > height:
        y0 = max(0, y0 - (y1 - height))
        y1 = height
    return x0, y0, x1, y1


def crop_image(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


def make_overlay(image: np.ndarray, rgba: np.ndarray, alpha_scale: float = 0.84) -> np.ndarray:
    alpha = rgba[:, :, 3:4] * alpha_scale
    return np.clip(image * (1 - alpha) + rgba[:, :, :3] * alpha, 0, 1)


def make_clean_mesh(rgba: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    mesh = crop_image(rgba, box)
    rgb = mesh[:, :, :3]
    alpha = mesh[:, :, 3:4]
    bg = np.ones_like(rgb) * np.array([0.965, 0.975, 0.985])
    return np.clip(rgb * alpha + bg * (1 - alpha), 0, 1)


def axis_image(ax, image: np.ndarray, title: str | None = None) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8.5, pad=2.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_edgecolor("#CBD5E1")


def load_profile_rows() -> pd.DataFrame:
    top = pd.read_csv(DATA / "final_beginner_top3_deviations.csv")
    rows = top[top["trial_id"] == PROFILE_TRIAL].copy()
    if len(rows) != 3:
        raise RuntimeError(f"Expected 3 profile rows for {PROFILE_TRIAL}, got {len(rows)}")
    return rows.sort_values("rank")


def draw_profile_card(ax, rows: pd.DataFrame) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.04, 0.94, "TennisProfile output", fontsize=8.0, fontweight="bold", color="#111827")
    ax.text(0.04, 0.86, "p5 forehand vs expert reference", fontsize=6.5, color="#475569")

    card = patches.FancyBboxPatch(
        (0.04, 0.30), 0.90, 0.48,
        boxstyle="round,pad=0.014,rounding_size=0.026",
        linewidth=0.7,
        edgecolor="#CBD5E1",
        facecolor="#F8FAFC",
    )
    ax.add_patch(card)
    ax.text(0.08, 0.72, "Top standardized deviations", fontsize=6.4, fontweight="bold", color="#0F172A")

    y0 = 0.61
    for idx, row in enumerate(rows.itertuples(index=False)):
        y = y0 - idx * 0.13
        label = FEATURE_LABELS.get(row.feature, row.feature.replace("_", " "))
        value = float(row.deviation_strength)
        outside = str(row.outside_expert_minmax) == "True"
        ax.text(0.08, y + 0.025, f"{int(row.rank)}. {label}", fontsize=6.2, color="#1F2937", va="center")
        ax.text(0.87, y + 0.025, f"{value:.1f} SD", fontsize=6.1, color="#C2410C", va="center", ha="right")
        if outside:
            ax.text(0.08, y - 0.028, "outside expert range", fontsize=5.2, color="#B45309")

    interp = patches.FancyBboxPatch(
        (0.04, 0.16), 0.90, 0.075,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        linewidth=0,
        facecolor="#FEF3C7",
    )
    ax.add_patch(interp)
    ax.text(
        0.49, 0.206,
        "Evidence level",
        fontsize=5.35,
        fontweight="bold",
        color="#92400E",
        ha="center",
        va="center",
    )
    ax.text(
        0.49, 0.181,
        "same-source timing cue",
        fontsize=5.15,
        color="#1F2937",
        ha="center",
        va="center",
    )

    ax.text(
        0.50, 0.06,
        "Named deviations with review evidence",
        fontsize=5.9,
        color="#334155",
        ha="center",
    )


def add_arrow(fig, ax_from, ax_to, color="#64748B") -> None:
    p0 = ax_from.get_position()
    p1 = ax_to.get_position()
    x0 = p0.x1 + 0.004
    y0 = (p0.y0 + p0.y1) / 2
    x1 = p1.x0 - 0.004
    y1 = (p1.y0 + p1.y1) / 2
    arrow = patches.FancyArrowPatch(
        (x0, y0), (x1, y1),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=8,
        lw=0.8,
        color=color,
        shrinkA=0,
        shrinkB=0,
    )
    fig.add_artist(arrow)


def main() -> int:
    roster = load_roster_events()
    config = get_config(str(MODEL_CONFIG), update_cachedir=True)
    smpl = SMPL(
        model_path=config.SMPL.MODEL_PATH,
        gender=config.SMPL.GENDER,
        num_body_joints=config.SMPL.NUM_BODY_JOINTS,
        joint_regressor_extra=config.SMPL.JOINT_REGRESSOR_EXTRA,
    )
    renderer = Renderer(config, faces=smpl.faces)

    rendered: dict[str, dict[str, list[np.ndarray] | tuple[int, int, int, int]]] = {}
    for stroke_name, trial_id in TRIALS:
        event_frames = [(label, roster[trial_id][column]) for label, column in EVENTS]
        rgbs: list[np.ndarray] = []
        overlays: list[np.ndarray] = []
        meshes: list[np.ndarray] = []
        for _, frame in event_frames:
            rgb = load_frame(trial_id, frame)
            rgba, _ = render_rgba(config, renderer, smpl, trial_id, frame)
            box = player_crop_box(trial_id, frame, rgb.shape, scale=1.95, aspect=1.0)
            rgbs.append(rgb)
            overlays.append(make_overlay(rgb, rgba))
            meshes.append(make_clean_mesh(rgba, box))
        rendered[stroke_name] = {
            "rgbs": [
                crop_image(img, player_crop_box(trial_id, frame, img.shape, scale=1.95, aspect=1.0))
                for img, (_, frame) in zip(rgbs, event_frames)
            ],
            "overlays": [
                crop_image(img, player_crop_box(trial_id, frame, img.shape, scale=1.95, aspect=1.0))
                for img, (_, frame) in zip(overlays, event_frames)
            ],
            "mesh": meshes[1],
            "frames": [frame for _, frame in event_frames],
        }

    profile_rows = load_profile_rows()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    fig = plt.figure(figsize=(7.15, 2.75))
    gs = gridspec.GridSpec(
        2, 6,
        figure=fig,
        width_ratios=[0.95, 0.95, 0.95, 1.08, 0.94, 1.95],
        wspace=0.09,
        hspace=0.18,
    )

    event_labels = ["Start", "Contact", "End"]
    for row, (stroke_name, _) in enumerate(TRIALS):
        item = rendered[stroke_name]
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            title = event_labels[col] if row == 0 else None
            axis_image(ax, item["rgbs"][col], title)
            if col == 0:
                ax.text(
                    -0.16, 0.50, stroke_name,
                    transform=ax.transAxes,
                    rotation=90,
                    fontsize=7.2,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    color="#111827",
                )
        ax_overlay = fig.add_subplot(gs[row, 3])
        axis_image(ax_overlay, item["overlays"][1], "HMR2 overlay" if row == 0 else None)

        ax_mesh = fig.add_subplot(gs[row, 4])
        axis_image(ax_mesh, item["mesh"], "3D mesh" if row == 0 else None)

    ax_card = fig.add_subplot(gs[:, 5])
    draw_profile_card(ax_card, profile_rows)

    # Group labels above the visual blocks.
    fig.text(0.205, 0.982, "Reviewed RGB events", fontsize=7.6, fontweight="bold", ha="center", color="#111827")
    fig.text(0.575, 0.982, "HMR2 recovery at contact", fontsize=7.6, fontweight="bold", ha="center", color="#111827")
    fig.text(0.855, 0.982, "Reference profile", fontsize=7.6, fontweight="bold", ha="center", color="#111827")

    # One visual arrow is enough to signal the transition from reviewed video
    # evidence to the recovered-body/profile representation.
    add_arrow(fig, fig.axes[2], fig.axes[3])

    fig.subplots_adjust(left=0.045, right=0.985, top=0.885, bottom=0.055)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT / f"fig1_hmr2_recovery_visualization.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        print(f"Wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
