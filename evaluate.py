"""Evaluate license plate detection and OCR on the Kaggle validation set.

Dataset: https://www.kaggle.com/datasets/duydieunguyen/licenseplates
Format: YOLO polygon segmentation (class x1 y1 x2 y2 x3 y3 x4 y4)

This script computes detection metrics (IoU-based precision/recall) and
reports OCR results.  Note: the dataset does NOT contain ground-truth text,
so OCR accuracy cannot be measured directly — only detection can.

Usage:
    python evaluate.py

Output:
    - Detection metrics printed to console
    - JSON report saved to output/eval_report.json
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

import config
from detector import LicensePlateDetector
from ocr_engine import OCREngine
from preprocessor import Preprocessor
from utils import logger
from validator import normalize

# Paths
ARCHIVE_DIR = Path("data/archive")
IMAGE_DIR = ARCHIVE_DIR / "images/val"
LABEL_DIR = ARCHIVE_DIR / "labels/val"
OUTPUT_DIR = Path(config.OUTPUT_DIR)

# Evaluation thresholds
IOU_THRESHOLD = 0.5       # IoU >= 0.5 counts as a match
CONF_THRESHOLD = 0.5      # detection confidence threshold


def polygon_to_bbox(poly):
    """Convert YOLO-normalized polygon to (x1, y1, x2, y2) pixel bbox."""
    xs = poly[0::2]
    ys = poly[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(label_path, img_w, img_h):
    """Load YOLO polygon labels and convert to pixel bboxes."""
    gts = []
    if not label_path.exists():
        return gts
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            # Denormalize
            xs = [c * img_w for c in coords[0::2]]
            ys = [c * img_h for c in coords[1::2]]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            gts.append({"class": cls, "box": (x1, y1, x2, y2)})
    return gts


def evaluate_image(detector, ocr, preprocessor, image_path, label_path):
    """Run detection + OCR on a single image and compare with GT."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None
    h, w = frame.shape[:2]

    gts = load_ground_truth(label_path, w, h)
    dets = detector.detect(frame)

    # Filter by confidence
    dets = [d for d in dets if d[4] >= CONF_THRESHOLD]

    matches = []
    matched_gt = set()
    matched_det = set()

    # Greedy matching by IoU
    for di, det in enumerate(dets):
        best_iou = 0
        best_gi = -1
        for gi, gt in enumerate(gts):
            if gi in matched_gt:
                continue
            iou = compute_iou(det[:4], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= IOU_THRESHOLD and best_gi not in matched_gt:
            matches.append({"det_idx": di, "gt_idx": best_gi, "iou": best_iou})
            matched_gt.add(best_gi)
            matched_det.add(di)

    # OCR on matched detections
    ocr_results = []
    for mi in matches:
        det = dets[mi["det_idx"]]
        crop_raw = preprocessor.crop_plate(frame, det)
        crop_enh = preprocessor.enhance(crop_raw)
        text, score = ocr.read_with_fallback(crop_raw, crop_enh)
        text = normalize(text)
        ocr_results.append({
            "text": text,
            "score": round(score, 3),
            "iou": round(mi["iou"], 3),
        })

    return {
        "image": image_path.name,
        "total_gt": len(gts),
        "total_det": len(dets),
        "tp": len(matches),
        "fp": len(dets) - len(matched_det),
        "fn": len(gts) - len(matched_gt),
        "ocr": ocr_results,
    }


def main():
    logger.info("Loading pipeline...")
    detector = LicensePlateDetector()
    preprocessor = Preprocessor()
    ocr = OCREngine()

    image_paths = sorted(IMAGE_DIR.glob("*.png"))
    if not image_paths:
        logger.error(f"No images found in {IMAGE_DIR}")
        return

    logger.info(f"Found {len(image_paths)} validation images")

    results = []
    total_tp = total_fp = total_fn = 0

    start = time.time()
    for i, img_path in enumerate(image_paths):
        lbl_path = LABEL_DIR / (img_path.stem + ".txt")
        res = evaluate_image(detector, ocr, preprocessor, img_path, lbl_path)
        if res is None:
            continue
        results.append(res)
        total_tp += res["tp"]
        total_fp += res["fp"]
        total_fn += res["fn"]

        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(image_paths)} images...")

    elapsed = time.time() - start

    # Aggregate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # OCR stats (only on matched/cropped plates)
    all_ocr_scores = [r["score"] for img in results for r in img["ocr"]]
    avg_ocr_score = sum(all_ocr_scores) / len(all_ocr_scores) if all_ocr_scores else 0.0

    report = {
        "dataset": "LicensePlates (Kaggle)",
        "images_evaluated": len(results),
        "time_seconds": round(elapsed, 1),
        "iou_threshold": IOU_THRESHOLD,
        "conf_threshold": CONF_THRESHOLD,
        "detection": {
            "total_gt_plates": total_tp + total_fn,
            "total_detections": total_tp + total_fp,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        },
        "ocr": {
            "plates_read": len(all_ocr_scores),
            "avg_confidence": round(avg_ocr_score, 4),
        },
    }

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(f"Images evaluated : {report['images_evaluated']}")
    print(f"Time elapsed     : {report['time_seconds']}s")
    print()
    print("--- Detection ---")
    print(f"Ground-truth plates : {report['detection']['total_gt_plates']}")
    print(f"Detections          : {report['detection']['total_detections']}")
    print(f"True Positives      : {report['detection']['true_positives']}")
    print(f"False Positives     : {report['detection']['false_positives']}")
    print(f"False Negatives     : {report['detection']['false_negatives']}")
    print(f"Precision           : {report['detection']['precision']:.2%}")
    print(f"Recall              : {report['detection']['recall']:.2%}")
    print(f"F1-Score            : {report['detection']['f1_score']:.4f}")
    print()
    print("--- OCR (on matched plates) ---")
    print(f"Plates read         : {report['ocr']['plates_read']}")
    print(f"Avg OCR confidence  : {report['ocr']['avg_confidence']:.2%}")
    print("=" * 50)

    # Save JSON
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
