"""Utility helpers for drawing and logging."""

import cv2
import logging

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def draw_result(frame, results):
    """Draw bounding boxes and recognized text onto the frame.

    Args:
        frame: BGR image (numpy array).
        results: List of dicts with keys "box", "text", "score", "valid".

    Returns:
        Annotated frame.
    """
    for r in results:
        x1, y1, x2, y2, conf = r["box"]
        text = r["text"]
        score = r["score"]
        valid = r["valid"]

        color = config.TEXT_COLOR if valid else config.INVALID_TEXT_COLOR
        label = f"{text} ({score:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            config.FONT,
            config.FONT_SCALE,
            color,
            config.FONT_THICKNESS,
        )

    return frame


def resize_for_display(frame, max_width=1280, max_height=720):
    """Resize frame to fit display window while keeping aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        frame = cv2.resize(frame, new_size)
    return frame
