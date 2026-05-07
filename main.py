"""Entry point for license plate recognition.

Supports image files, video files, and webcam (source=0).
"""

import os

# Disable oneDNN/MKLDNN before any Paddle imports to avoid Windows inference bugs
os.environ["FLAGS_use_mkldnn"] = "0"

import argparse
import sys

import cv2

import config
from detector import LicensePlateDetector
from ocr_engine import OCREngine
from preprocessor import Preprocessor
from utils import draw_result, logger, resize_for_display
from validator import is_valid_plate, normalize


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")


def process_frame(frame, detector, preprocessor, ocr):
    """Run the full detection + OCR pipeline on a single frame.

    Returns:
        Annotated frame and list of result dicts.
    """
    boxes = detector.detect(frame)
    results = []

    for box in boxes:
        crop_raw = preprocessor.crop_plate(frame, box)
        crop_enhanced = preprocessor.enhance(crop_raw)
        text, score = ocr.read_with_fallback(crop_raw, crop_enhanced)
        text = normalize(text)
        valid = is_valid_plate(text)

        results.append({
            "box": box,
            "text": text,
            "score": score,
            "valid": valid,
        })
        logger.info(f"Plate: {text} | conf={score:.2f} | valid={valid}")

    annotated = draw_result(frame.copy(), results)
    return annotated, results


def process_image(source_path, detector, preprocessor, ocr, save: bool = False):
    """Process a single image and optionally save the result."""
    frame = cv2.imread(source_path)
    if frame is None:
        logger.error(f"Cannot read image: {source_path}")
        sys.exit(1)

    annotated, _ = process_frame(frame, detector, preprocessor, ocr)
    display = resize_for_display(annotated)
    cv2.imshow(config.WINDOW_NAME, display)

    if save:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_IMAGE_NAME)
        cv2.imwrite(out_path, annotated)
        logger.info(f"Saved result to {out_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(source_path, detector, preprocessor, ocr, save: bool = False):
    """Process a video file or webcam stream."""
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source_path}")
        sys.exit(1)

    writer = None
    if save:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or config.OUTPUT_VIDEO_FPS
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_FOURCC)
        out_path = os.path.join(config.OUTPUT_DIR, "result.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        logger.info(f"Saving output video to {out_path}")

    # Calculate skip step to achieve ~TARGET_PROCESS_FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS) or config.OUTPUT_VIDEO_FPS
    frame_skip = max(1, int(round(video_fps / config.TARGET_PROCESS_FPS)))
    logger.info(f"Video FPS={video_fps:.1f}, processing every {frame_skip} frame(s) (~{config.TARGET_PROCESS_FPS} FPS)")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        annotated, _ = process_frame(frame, detector, preprocessor, ocr)

        if writer is not None:
            writer.write(annotated)

        display = resize_for_display(annotated)
        cv2.imshow(config.WINDOW_NAME, display)

        if cv2.waitKey(config.WAIT_KEY_MS) & 0xFF == ord("q"):
            logger.info("Quit signal received.")
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="License Plate Recognition with YOLOv8 + PaddleOCR"
    )
    parser.add_argument(
        "--source",
        default=config.DEFAULT_SOURCE,
        help="Input source: 0=webcam, path to image, or path to video",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the annotated result(s) to the output folder",
    )
    args = parser.parse_args()

    # Resolve source
    source = int(args.source) if args.source == "0" else args.source

    # Initialize pipeline
    logger.info("Initializing pipeline components...")
    detector = LicensePlateDetector()
    preprocessor = Preprocessor()
    ocr = OCREngine()
    logger.info("Pipeline ready.")

    # Determine input type
    if isinstance(source, str):
        ext = os.path.splitext(source.lower())[1]
        if ext in IMAGE_EXTS:
            logger.info(f"Processing image: {source}")
            process_image(source, detector, preprocessor, ocr, save=args.save)
        elif ext in VIDEO_EXTS:
            logger.info(f"Processing video: {source}")
            process_video(source, detector, preprocessor, ocr, save=args.save)
        else:
            logger.warning(f"Unknown extension '{ext}', trying as video...")
            process_video(source, detector, preprocessor, ocr, save=args.save)
    else:
        logger.info("Processing webcam stream...")
        process_video(source, detector, preprocessor, ocr, save=args.save)


if __name__ == "__main__":
    main()
