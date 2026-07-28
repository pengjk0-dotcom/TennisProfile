#!/usr/bin/env python3
"""Run frame-level 4DHumans/HMR2 inference on an extracted-frame manifest.

This avoids the official PHALP tracker path because that path depends on
CUDA-only neural-renderer components on macOS. The output is intentionally
frame-level; temporal linking and smoothing are handled downstream.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
FOUR_D_HUMANS_DIR = ROOT / "external" / "4D-Humans"
sys.path.insert(0, str(FOUR_D_HUMANS_DIR))

SMPL_NEUTRAL_BASENAME = "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require_smpl_model() -> None:
    expected_paths = [
        FOUR_D_HUMANS_DIR / "data" / SMPL_NEUTRAL_BASENAME,
        Path.home() / ".cache" / "4DHumans" / "data" / "smpl" / "SMPL_NEUTRAL.pkl",
    ]
    if any(path.exists() for path in expected_paths):
        return
    expected = expected_paths[0]
    raise SystemExit(
        "Missing licensed SMPL neutral model. Download it from https://smplify.is.tue.mpg.de/ "
        f"and place it at: {expected}"
    )


def full_frame_boxes(img_cv2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = img_cv2.shape[:2]
    boxes = np.array([[0.0, 0.0, float(width - 1), float(height - 1)]], dtype=np.float32)
    scores = np.array([1.0], dtype=np.float32)
    return boxes, scores


def build_detector(detector_name: str, score_threshold: float):
    if detector_name == "full-frame":
        return None

    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy

    if detector_name == "vitdet":
        from detectron2.config import LazyConfig
        import hmr2

        cfg_path = Path(hmr2.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = score_threshold
        return DefaultPredictor_Lazy(detectron2_cfg)

    if detector_name == "regnety":
        from detectron2 import model_zoo

        detectron2_cfg = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py",
            trained=True,
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = score_threshold
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        return DefaultPredictor_Lazy(detectron2_cfg)

    raise ValueError(f"Unknown detector: {detector_name}")


def detect_person_boxes(detector, detector_name: str, img_cv2: np.ndarray, score_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if detector_name == "full-frame":
        return full_frame_boxes(img_cv2)

    det_out = detector(img_cv2)
    det_instances = det_out["instances"]
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > score_threshold)
    boxes = det_instances.pred_boxes.tensor[valid_idx].detach().cpu().numpy().astype(np.float32)
    scores = det_instances.scores[valid_idx].detach().cpu().numpy().astype(np.float32)
    return boxes, scores


def keep_largest_box(boxes: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes) <= 1:
        return boxes, scores
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    keep = int(np.argmax(widths * heights))
    return boxes[[keep]], scores[[keep]]


def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def floating_tensors_to_float32(value):
    if isinstance(value, dict):
        return {key: floating_tensors_to_float32(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [floating_tensors_to_float32(item) for item in value]
        return type(value)(converted)
    if torch.is_tensor(value) and value.is_floating_point() and value.dtype == torch.float64:
        return value.float()
    return value


def save_prediction_npz(
    out_path: Path,
    frame_row: pd.Series,
    person_rank: int,
    detector_name: str,
    detector_score: float,
    bbox_xyxy: np.ndarray,
    batch: dict,
    out: dict,
    batch_index: int,
    pred_cam_t_full: np.ndarray,
    save_vertices: bool,
) -> None:
    smpl_params = out["pred_smpl_params"]
    payload = {
        "trial_id": str(frame_row["trial_id"]),
        "subject_id": str(frame_row.get("subject_id", "")),
        "skill_level": str(frame_row.get("skill_level", "")),
        "stroke_type": str(frame_row.get("stroke_type", "")),
        "action_type": str(frame_row.get("action_type", "")),
        "frame_index": int(frame_row["frame_index"]),
        "timestamp_sec": float(frame_row["timestamp_sec"]) if str(frame_row["timestamp_sec"]) else np.nan,
        "person_rank": int(person_rank),
        "detector_name": detector_name,
        "detector_score": float(detector_score),
        "bbox_xyxy": np.asarray(bbox_xyxy, dtype=np.float32),
        "box_center": to_numpy(batch["box_center"][batch_index]).astype(np.float32),
        "box_size": np.asarray(to_numpy(batch["box_size"][batch_index]), dtype=np.float32),
        "img_size": to_numpy(batch["img_size"][batch_index]).astype(np.float32),
        "pred_keypoints_3d": to_numpy(out["pred_keypoints_3d"][batch_index]).astype(np.float32),
        "pred_keypoints_2d": to_numpy(out["pred_keypoints_2d"][batch_index]).astype(np.float32),
        "pred_cam": to_numpy(out["pred_cam"][batch_index]).astype(np.float32),
        "pred_cam_t": to_numpy(out["pred_cam_t"][batch_index]).astype(np.float32),
        "pred_cam_t_full": np.asarray(pred_cam_t_full, dtype=np.float32),
        "smpl_global_orient": to_numpy(smpl_params["global_orient"][batch_index]).astype(np.float32),
        "smpl_body_pose": to_numpy(smpl_params["body_pose"][batch_index]).astype(np.float32),
        "smpl_betas": to_numpy(smpl_params["betas"][batch_index]).astype(np.float32),
    }
    if save_vertices:
        payload["pred_vertices"] = to_numpy(out["pred_vertices"][batch_index]).astype(np.float32)
    np.savez_compressed(out_path, **payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--detector", default="full-frame", choices=["full-frame", "vitdet", "regnety"])
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--person-selection", choices=["largest", "all"], default="largest")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-vertices", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    frame_manifest = pd.read_csv(args.frame_manifest)
    if args.max_frames:
        frame_manifest = frame_manifest.head(args.max_frames).copy()

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "npz"
    pred_dir.mkdir(parents=True, exist_ok=True)

    require_smpl_model()

    from hmr2.models import DEFAULT_CHECKPOINT, load_hmr2
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.utils import recursive_to
    from hmr2.utils.renderer import cam_crop_to_full

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
    old_cwd = Path.cwd()
    os.chdir(FOUR_D_HUMANS_DIR)
    try:
        model, model_cfg = load_hmr2(checkpoint)
    finally:
        os.chdir(old_cwd)

    device = choose_device(args.device)
    model = model.to(device)
    model.eval()
    print(f"Using HMR2 device: {device}")

    detector = build_detector(args.detector, args.score_threshold)

    rows: list[dict[str, object]] = []
    for frame_row in frame_manifest.itertuples(index=False):
        row = pd.Series(frame_row._asdict())
        frame_path = Path(str(row["frame_path"]))
        if not frame_path.is_absolute():
            frame_path = ROOT / frame_path
        img_cv2 = cv2.imread(str(frame_path))
        if img_cv2 is None:
            raise FileNotFoundError(frame_path)

        boxes, scores = detect_person_boxes(detector, args.detector, img_cv2, args.score_threshold)
        if args.person_selection == "largest":
            boxes, scores = keep_largest_box(boxes, scores)

        if len(boxes) == 0:
            rows.append({
                "trial_id": row["trial_id"],
                "frame_index": int(row["frame_index"]),
                "frame_path": str(frame_path),
                "status": "no_person_detected",
                "num_people": 0,
                "prediction_path": "",
            })
            continue

        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        person_offset = 0
        for batch in dataloader:
            batch = floating_tensors_to_float32(batch)
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            pred_cam = out["pred_cam"]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full_batch = cam_crop_to_full(
                pred_cam,
                box_center,
                box_size,
                img_size,
                scaled_focal_length,
            ).detach().cpu().numpy()

            batch_count = int(batch["img"].shape[0])
            for batch_index in range(batch_count):
                person_rank = person_offset + batch_index
                pred_path = pred_dir / f"{row['trial_id']}_f{int(row['frame_index']):06d}_p{person_rank:02d}.npz"
                if not (args.skip_existing and pred_path.exists()):
                    save_prediction_npz(
                        out_path=pred_path,
                        frame_row=row,
                        person_rank=person_rank,
                        detector_name=args.detector,
                        detector_score=float(scores[person_rank]),
                        bbox_xyxy=boxes[person_rank],
                        batch=batch,
                        out=out,
                        batch_index=batch_index,
                        pred_cam_t_full=pred_cam_t_full_batch[batch_index],
                        save_vertices=args.save_vertices,
                    )

                rows.append({
                    "dataset_id": row.get("dataset_id", ""),
                    "subject_id": row.get("subject_id", ""),
                    "skill_level": row.get("skill_level", ""),
                    "stroke_type": row.get("stroke_type", ""),
                    "action_type": row.get("action_type", ""),
                    "trial_id": row["trial_id"],
                    "frame_index": int(row["frame_index"]),
                    "timestamp_sec": row.get("timestamp_sec", ""),
                    "frame_path": str(frame_path),
                    "status": "ok",
                    "num_people": len(boxes),
                    "person_rank": person_rank,
                    "detector": args.detector,
                    "detector_score": float(scores[person_rank]),
                    "bbox_x1": float(boxes[person_rank][0]),
                    "bbox_y1": float(boxes[person_rank][1]),
                    "bbox_x2": float(boxes[person_rank][2]),
                    "bbox_y2": float(boxes[person_rank][3]),
                    "prediction_path": str(pred_path),
                })
            person_offset += batch_count

        print(f"{row['trial_id']} frame {int(row['frame_index'])}: saved {len(boxes)} prediction(s)")

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_manifest, index=False)
    print(f"Wrote {len(rows)} prediction rows to {out_manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
