"""OCR engine using PaddleOCR with multi-attempt fallback."""

import os

# Disable oneDNN/MKLDNN to avoid Windows inference errors
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
from paddleocr import PaddleOCR

import config
from utils import logger


class OCREngine:
    """Wrapper around PaddleOCR for reading license-plate text."""

    def __init__(self):
        logger.info("Initializing PaddleOCR engine...")
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=config.OCR_USE_ANGLE_CLS,
            lang=config.OCR_LANG,
            enable_mkldnn=False,
        )

    def read(self, image):
        """Run OCR on a single image.

        Args:
            image: Numpy array (BGR or grayscale).

        Returns:
            Tuple (text, confidence).  Empty string and 0.0 if nothing found.
        """
        # Ensure 3-channel BGR/RGB because PaddleOCR text detection expects HxWxC
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        result = self.ocr.ocr(image)

        if not result:
            return "", 0.0

        # PaddleX Pipeline returns a list of OCRResult objects.
        # OCRResult is dict-like with keys: rec_texts, rec_scores, rec_boxes, etc.
        page_result = result[0]

        texts = page_result.get("rec_texts", [])
        scores = page_result.get("rec_scores", [])

        if not texts:
            return "", 0.0

        full_text = "".join(texts).upper().replace(" ", "").replace("-", "")
        avg_score = sum(scores) / len(scores) if scores else 0.0
        logger.info(f"OCR extracted text='{full_text}', score={avg_score:.2f}")
        return full_text, avg_score

    def read_with_fallback(self, crop_raw, crop_enhanced):
        """Try OCR on both raw and enhanced crops, return the better result.

        Args:
            crop_raw: Original cropped plate.
            crop_enhanced: Preprocessed / enhanced plate.

        Returns:
            Tuple (text, confidence) of the best attempt.
        """
        text1, score1 = self.read(crop_raw)
        text2, score2 = self.read(crop_enhanced)

        if score1 >= score2:
            return text1, score1
        return text2, score2
