"""Flask web app for license plate recognition."""

import base64
import os
import tempfile
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

import config
from detector import LicensePlateDetector
from ocr_engine import OCREngine
from preprocessor import Preprocessor
from utils import draw_result, logger
from validator import is_valid_plate, normalize

app = Flask(__name__)

# Initialize pipeline once at startup
detector = LicensePlateDetector()
preprocessor = Preprocessor()
ocr = OCREngine()

# Ensure output directory exists
os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def _process_frame_arr(frame):
    """Process a single frame (numpy array) and return annotated frame + results."""
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
            "score": round(score, 2),
            "valid": valid,
        })

    annotated = draw_result(frame.copy(), results)
    return annotated, results


def _cv2_to_base64(img):
    """Convert OpenCV BGR image to base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """Upload image and run detection + OCR."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    annotated, results = _process_frame_arr(frame)

    return jsonify({
        "success": True,
        "plates": results,
        "annotated_image": _cv2_to_base64(annotated),
    })


@app.route("/api/detect_video", methods=["POST"])
def api_detect_video():
    """Upload video and process every frame. Returns annotated video."""
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400

    file = request.files["video"]
    video_bytes = file.read()

    # Save uploaded video to temp file
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(video_bytes)
        tmp_in_path = tmp_in.name

    cap = cv2.VideoCapture(tmp_in_path)
    if not cap.isOpened():
        os.unlink(tmp_in_path)
        return jsonify({"error": "Cannot open video file"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or config.OUTPUT_VIDEO_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output path
    out_filename = f"result_{uuid.uuid4().hex[:8]}.mp4"
    out_path = os.path.join(config.OUTPUT_DIR, out_filename)

    fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_FOURCC)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    all_plates = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, results = _process_frame_arr(frame)
        writer.write(annotated)

        for r in results:
            all_plates.append({
                "frame": frame_idx,
                "text": r["text"],
                "score": r["score"],
                "valid": r["valid"],
            })

        frame_idx += 1

    cap.release()
    writer.release()
    os.unlink(tmp_in_path)

    # Deduplicate plates by text
    seen = set()
    unique_plates = []
    for p in all_plates:
        if p["text"] and p["text"] not in seen:
            seen.add(p["text"])
            unique_plates.append(p)

    return jsonify({
        "success": True,
        "download_url": f"/output/{out_filename}",
        "total_frames": frame_idx,
        "unique_plates": unique_plates,
        "plates_detected": len(unique_plates),
    })


@app.route("/output/<path:filename>")
def serve_output(filename):
    """Serve processed video files from output directory."""
    return send_file(os.path.join(config.OUTPUT_DIR, filename), as_attachment=True)


if __name__ == "__main__":
    logger.info("Starting Flask web server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
