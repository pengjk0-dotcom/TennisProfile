#!/usr/bin/env python3
"""Build supplementary visual boundary examples for TennisProfile.

This supplementary figure follows the qualitative visual style of sports
motion-capture papers: each example shows the reviewed RGB evidence and the
corresponding HMR2 mesh overlay.  The goal is not to introduce a new result in
the main text, but to give reviewers representative visual anchors for the
interpretation boundaries discussed in the paper.

Run with:

    PYTHONPATH=external/4D-Humans ./.micromamba/envs/4dhumans/bin/python \
        scripts/make_boundary_examples_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from hmr2.configs import get_config
from hmr2.models.smpl_wrapper import SMPL
from hmr2.utils.renderer import Renderer

from make_tennisprofile_overview_figure import (
    MODEL_CONFIG,
    OUT,
    crop_image,
    load_frame,
    load_roster_events,
    make_overlay,
    player_crop_box,
    render_rgba,
)


EXAMPLES = [
    {
        "title": "Multi-player background",
        "trial": "p30_forehand_flat_s1",
        "note": "visual clutter",
        "scale": 2.25,
        "aspect": 1.14,
    },
    {
        "title": "Fast wrist-racket region",
        "trial": "p5_forehand_flat_s1",
        "note": "local high-speed motion",
        "scale": 2.05,
        "aspect": 1.14,
    },
    {
        "title": "Body mesh vs. racket detail",
        "trial": "p51_forehand_flat_s1",
        "note": "body-level interpretation",
        "scale": 2.15,
        "aspect": 1.14,
    },
]


def _axis_image(ax, image, title: str | None = None) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8.3, fontweight="bold", pad=3, color="#111827")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_edgecolor("#CBD5E1")


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

    rendered = []
    for item in EXAMPLES:
        trial_id = item["trial"]
        frame = roster[trial_id]["ball_contact_frame"]
        rgb = load_frame(trial_id, frame)
        rgba, _ = render_rgba(config, renderer, smpl, trial_id, frame)
        box = player_crop_box(
            trial_id,
            frame,
            rgb.shape,
            scale=float(item["scale"]),
            aspect=float(item["aspect"]),
        )
        rendered.append(
            {
                **item,
                "frame": frame,
                "rgb": crop_image(rgb, box),
                "overlay": crop_image(make_overlay(rgb, rgba), box),
            }
        )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    fig = plt.figure(figsize=(7.1, 3.35))
    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[0.27, 1.0, 1.0],
        wspace=0.08,
        hspace=0.09,
    )

    for col, item in enumerate(rendered):
        ax_title = fig.add_subplot(gs[0, col])
        ax_title.set_axis_off()
        ax_title.text(
            0.5,
            0.70,
            item["title"],
            ha="center",
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color="#111827",
        )
        ax_title.text(
            0.5,
            0.31,
            item["note"],
            ha="center",
            va="center",
            fontsize=5.4,
            color="#475569",
        )

        ax_rgb = fig.add_subplot(gs[1, col])
        _axis_image(ax_rgb, item["rgb"])

        ax_overlay = fig.add_subplot(gs[2, col])
        _axis_image(ax_overlay, item["overlay"])

    fig.text(
        0.013,
        0.53,
        "RGB frame / HMR2 overlay",
        rotation=90,
        va="center",
        ha="center",
        fontsize=6.8,
        fontweight="bold",
        color="#111827",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT / f"fig_supp_boundary_examples.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        print(f"Wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
