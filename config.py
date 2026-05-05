"""Centralized configuration for the license plate recognition system."""

# YOLOv8 detector settings
YOLO_MODEL = "models/best.pt"
YOLO_CONF = 0.5
YOLO_IOU = 0.45

# Preprocessing settings
PADDING = 5
MIN_PLATE_WIDTH = 200
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
DENOISE_STRENGTH = 10
ADAPTIVE_THRESH_BLOCK_SIZE = 11
ADAPTIVE_THRESH_C = 2

# OCR settings
OCR_USE_ANGLE_CLS = True
OCR_LANG = "en"

# Video / webcam settings
DEFAULT_SOURCE = "0"
FRAME_SKIP = 1
WAIT_KEY_MS = 1

# Output settings
OUTPUT_DIR = "output"
OUTPUT_IMAGE_NAME = "result.jpg"
OUTPUT_VIDEO_FOURCC = "mp4v"
OUTPUT_VIDEO_FPS = 20.0

# Display settings
WINDOW_NAME = "License Plate Recognition"
BBOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
INVALID_TEXT_COLOR = (0, 0, 255)
FONT = 0
FONT_SCALE = 0.7
FONT_THICKNESS = 2
