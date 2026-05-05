"""License plate detection using YOLOv8."""

from ultralytics import YOLO

import config
from utils import logger


class LicensePlateDetector:
    """Wrapper around an Ultralytics YOLOv8 model for license-plate detection."""

    def __init__(self, model_path: str = config.YOLO_MODEL):
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = config.YOLO_CONF
        self.iou_threshold = config.YOLO_IOU

    def detect(self, frame):
        """Detect license plates in a frame.

        Args:
            frame: BGR image (numpy array).

        Returns:
            List of tuples (x1, y1, x2, y2, confidence).
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, conf))

        if boxes:
            logger.info(f"Detected {len(boxes)} plate(s)")
        return boxes
