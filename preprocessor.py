"""Image preprocessing pipeline for better OCR accuracy."""

import cv2
import numpy as np

import config


class Preprocessor:
    """Crop and enhance license-plate regions for OCR."""

    @staticmethod
    def crop_plate(frame, box, padding: int = config.PADDING):
        """Crop the plate region from the frame with optional padding."""
        x1, y1, x2, y2, _ = box
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def enhance(crop):
        """Apply preprocessing steps to improve OCR readability.

        Steps:
            1. Resize if width is too small.
            2. Convert to grayscale.
            3. CLAHE contrast enhancement.
            4. Non-local means denoising.
            5. Adaptive Gaussian threshold.
        """
        # 1. Resize if too small
        h, w = crop.shape[:2]
        if w < config.MIN_PLATE_WIDTH:
            scale = config.MIN_PLATE_WIDTH / w
            crop = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )

        # 2. Grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 3. CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_GRID_SIZE,
        )
        enhanced = clahe.apply(gray)

        # 4. Denoise
        denoised = cv2.fastNlMeansDenoising(
            enhanced, h=config.DENOISE_STRENGTH
        )

        # 5. Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            config.ADAPTIVE_THRESH_BLOCK_SIZE,
            config.ADAPTIVE_THRESH_C,
        )
        return thresh
