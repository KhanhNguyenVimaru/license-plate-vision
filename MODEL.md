# Models

This document describes the machine learning models used in Plate Vision, their origins, and why they work out-of-the-box without custom training.

---

## 1. License Plate Detection — YOLOv8

### What is it?
A single-shot object detector that locates license plates inside an image and returns bounding boxes.

### Architecture
- **Base model:** YOLOv8n (nano) from [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Task:** Object detection (single class: "license plate")
- **Input:** RGB image, any resolution
- **Output:** Bounding boxes `[x1, y1, x2, y2, confidence]`

### Where did the weights come from?
The `models/best.pt` file was produced by fine-tuning YOLOv8n on a curated license-plate dataset.  It was published by **Muhammad Moin Faisal** in the open-source repository:

> [Automatic Number Plate Detection Recognition YOLOv8](https://github.com/MuhammadMoinFaisal/Automatic_Number_Plate_Detection_Recognition_YOLOv8)

You downloaded it during setup via Google Drive:
```bash
gdown "https://drive.google.com/uc?id=1dIyJooVaowaNUj0R1Q-HUnu-utiGsEj8" -O models/best.pt
```

### Why does it work without training?
YOLOv8 is **pretrained** on the COCO dataset (millions of general objects).  The author then **fine-tuned** it specifically for license plates, meaning the model already learned:
- Rectangular shape of plates
- Typical location on vehicles
- Scale and aspect-ratio patterns

Because Vietnamese plates share the same fundamental geometry (rectangle + text), the model generalizes to them **without additional training**.

---

## 2. Text Recognition — PaddleOCR (PP-OCRv5)

### What is it?
A two-stage OCR engine that reads characters inside the cropped plate region.

### Architecture
| Stage | Model | Purpose |
|-------|-------|---------|
| Text Detection | PP-OCRv5 Server Det | Finds text lines inside the crop |
| Text Recognition | en_PP-OCRv5 Mobile Rec | Reads Latin characters (English alphabet + digits) |
| Text Line Orientation | PP-LCNet_x1_0_textline_ori | Rotates vertical/slanted text to horizontal |

### Where did the weights come from?
PaddleOCR downloads the three models **automatically on first run** from the [PaddlePaddle official model hub](https://github.com/PaddlePaddle/PaddleOCR) to:
```
~/.paddlex/official_models/
```

### Why does it work without training?
PaddleOCR is pretrained on massive multilingual text datasets (books, street signs, scanned documents).  The English recognition model (`en_PP-OCRv5_mobile_rec`) already knows the Latin alphabet and digits used on Vietnamese plates.

**Limitation:** It does not know Vietnamese-specific rules (e.g., plate format).  That is handled later by `validator.py`.

---

## 3. The Complete Pipeline

```
Input Image
    │
    ▼
┌─────────────────────────┐
│  YOLOv8 Detector        │  ← models/best.pt (local file)
│  Finds plate regions    │
└───────────┬─────────────┘
            │ cropped plate image(s)
            ▼
┌─────────────────────────┐
│  PaddleOCR Engine       │  ← auto-downloaded PP-OCRv5 weights
│  Reads text in crop     │
└───────────┬─────────────┘
            │ raw text string
            ▼
┌─────────────────────────┐
│  Rule-based Validator   │  ← validator.py (no ML, just regex)
│  Fixes OCR errors       │
│  Checks VN format       │
└───────────┬─────────────┘
            │ validated plate text
            ▼
        Annotated Output
```

---

## 4. Why You Don't Need to Train

| Component | Already Learned | Your Data Needed? |
|-----------|----------------|-------------------|
| YOLOv8 Detector | Plate shape, location, scale | No (works out-of-the-box) |
| PaddleOCR | Latin letters & digits | No (works out-of-the-box) |
| Validator | Regex rules for VN plates | No (hard-coded) |

You only need to train or fine-tune if:
- Detection fails on **unusual angles** or **poor lighting** specific to your cameras.
- You need to detect **non-standard plates** (e.g., diplomatic, military, or custom sizes).
- You want to read plates in **non-Latin scripts** (e.g., Arabic, Thai, Chinese).

---

## 5. Model Files Summary

| File / Directory | Size | Source | Description |
|------------------|------|--------|-------------|
| `models/best.pt` | ~52 MB | [Google Drive](https://drive.google.com/uc?id=1dIyJooVaowaNUj0R1Q-HUnu-utiGsEj8) | YOLOv8 license-plate detector |
| `~/.paddlex/official_models/PP-OCRv5_server_det` | ~32 MB | Auto-download (PaddleOCR) | Text detection backbone |
| `~/.paddlex/official_models/en_PP-OCRv5_mobile_rec` | ~8 MB | Auto-download (PaddleOCR) | Text recognition head (English/Latin) |
| `~/.paddlex/official_models/PP-LCNet_x1_0_textline_ori` | ~7 MB | Auto-download (PaddleOCR) | Orientation classifier |

---

## 6. References

1. [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
2. [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
3. [MuhammadMoinFaisal YOLOv8 ANPR](https://github.com/MuhammadMoinFaisal/Automatic_Number_Plate_Detection_Recognition_YOLOv8)
4. [Kaggle License Plates Dataset](https://www.kaggle.com/datasets/duydieunguyen/licenseplates)
