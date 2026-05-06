# Plate Vision

> Vietnamese License Plate Recognition powered by **YOLOv8** + **PaddleOCR**

A license plate detection and recognition system for Vietnamese vehicles. Supports image upload and video file processing with frame-by-frame detection, returning annotated outputs and a deduplicated list of detected plates.

---

## Features

- **License Plate Detection** using YOLOv8 (`best.pt`)
- **Text Recognition** using PaddleOCR (PP-OCRv5)
- **Vietnamese Plate Validation** with regex patterns and OCR error correction
- **Web Interface** built with Flask + Tailwind CSS
- **Image Upload** – single-frame detection with annotated result
- **Video Upload** – frame-by-frame processing with downloadable annotated output video
- **Image Preprocessing** pipeline (CLAHE, denoising, adaptive threshold)
- **Multi-attempt OCR** – tries both raw and enhanced crops, picks the best result
- **Cross-platform** – works on Windows, Linux, macOS

---

## Architecture

```
Input (Image / Video)
    │
    ▼
┌─────────────────────────────┐
│  YOLOv8 Detection           │  ← models/best.pt
│  (License Plate Region)     │
└────────────┬────────────────┘
             │ crop plate region
             ▼
┌─────────────────────────────┐
│  Image Preprocessing        │  ← resize, grayscale, CLAHE,
│                             │    denoise, adaptive threshold
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  PaddleOCR                  │  ← PP-OCRv5 Server Det + EN Mobile Rec
│  (Text Recognition)         │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Normalization & Validation │  ← regex patterns + common OCR fixes
└────────────┬────────────────┘
             │
             ▼
    Annotated Image / Video + JSON
```

---

## Project Structure

```
plate-vision/
├── app.py                  # Flask web server (REST API + UI)
├── main.py                 # CLI entry point (image / video)
├── config.py               # Centralized configuration
├── detector.py             # YOLOv8 plate detector wrapper
├── ocr_engine.py           # PaddleOCR wrapper with fallback logic
├── preprocessor.py         # Image preprocessing pipeline
├── validator.py            # Regex validation & OCR error correction
├── utils.py                # Drawing utilities & logging
├── requirements.txt        # Python dependencies
├── setup.bat               # Windows setup script (CMD)
├── setup.ps1               # Windows setup script (PowerShell)
├── PLAN.md                 # Development plan (Vietnamese)
│
├── models/                 # Model weights
│   └── best.pt             # YOLOv8 license plate detector (52 MB)
│
├── templates/
│   └── index.html          # Tailwind CSS web interface
│
├── data/
│   ├── images/             # Test images
│   └── videos/             # Test videos
│
└── output/                 # Detection results (images & videos)
```

---

## Setup

### Prerequisites

- Python **3.10 - 3.12** (PaddlePaddle has limited support for Python 3.13)
- Windows / Linux / macOS

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/plate-vision.git
cd plate-vision
```

### 2. Create Virtual Environment

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, PaddleOCR will automatically download its inference models (`~100 MB`) to `~/.paddlex/official_models/`.

### 4. Download YOLOv8 Weights

If `models/best.pt` is missing, download it from Google Drive:

```bash
pip install gdown
gdown "https://drive.google.com/uc?id=1dIyJooVaowaNUj0R1Q-HUnu-utiGsEj8" -O models/best.pt
```

---

## Usage

### Web Interface (Recommended)

```bash
python app.py
```

Open your browser at [`http://127.0.0.1:5000`](http://127.0.0.1:5000).

#### Image Upload
- Switch to the **Upload Image** tab
- Drop or select an image (JPG, PNG, WEBP)
- Click **Detect License Plate**
- View annotated result and detected plates

#### Video Upload
- Switch to the **Upload Video** tab
- Drop or select a video (MP4, AVI, MOV, MKV)
- Preview your video, then click **Process Video**
- Wait for processing (depends on video length)
- Download the annotated output video
- View the deduplicated list of plates detected throughout the video

### Command Line

```bash
# Image
python main.py --source data/images/test.jpg --save

# Video
python main.py --source data/videos/traffic.mp4 --save
```

---

## How It Works

### 1. Detection
`detector.py` loads a YOLOv8 model (`models/best.pt`) to detect license plate bounding boxes in each input frame.

### 2. Preprocessing
`preprocessor.py` crops the detected region and applies:
- Resize (if width < 200 px)
- Grayscale conversion
- CLAHE contrast enhancement
- Non-local means denoising
- Adaptive Gaussian threshold

### 3. OCR
`ocr_engine.py` runs PaddleOCR on both the **raw crop** and the **enhanced crop**, then selects the result with the higher confidence score.

### 4. Validation
`validator.py` normalizes the text and validates it against Vietnamese license plate patterns:

| Pattern | Example |
|---------|---------|
| `\d{2}[A-Z]\d{5,6}` | `90B245230` (new format) |
| `\d{2}[A-Z]\d{4,5}` | `51A12345` |
| `\d{2}[A-Z]{1,2}\d{4,5}` | `51AB12345` |

Common OCR misreads are automatically corrected (`O` → `0`, `I` → `1`, `S` → `5`, etc.).

---

## Evaluation

The detection and OCR pipeline was evaluated on the [Kaggle License Plates dataset](https://www.kaggle.com/datasets/duydieunguyen/licenseplates) (YOLO polygon segmentation format) containing **1,145 validation images** with **1,294 ground-truth plates**.

### Detection Metrics (IoU ≥ 0.5, Confidence ≥ 0.5)

| Metric | Value |
|--------|-------|
| Ground-truth plates | 1,294 |
| Detections | 1,105 |
| True Positives | 1,042 |
| False Positives | 63 |
| False Negatives | 252 |
| **Precision** | **94.30%** |
| **Recall** | **80.53%** |
| **F1-Score** | **0.8687** |

### OCR Metrics (on matched plates)

| Metric | Value |
|--------|-------|
| Plates read | 1,042 |
| Avg OCR confidence | **95.31%** |

> **Note:** OCR accuracy cannot be measured directly because the dataset does not contain ground-truth text labels. The reported confidence reflects the PaddleOCR internal confidence.

### Speed

| Metric | Value |
|--------|-------|
| Images evaluated | 1,145 |
| Time elapsed | 5,325.7 s |
| Average speed | ~4.65 s / image |

---

## Configuration

Edit `config.py` to tune the pipeline:

```python
# Detection
YOLO_CONF = 0.5           # Confidence threshold
YOLO_IOU = 0.45           # NMS IoU threshold

# Preprocessing
MIN_PLATE_WIDTH = 200     # Upscale if plate is too small
PADDING = 5               # Crop padding in pixels

# OCR
OCR_LANG = "en"           # PaddleOCR language
OCR_USE_ANGLE_CLS = True  # Enable text-line orientation
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model download fails | Manually download `best.pt` with `gdown` (see Setup step 4). |
| Invalid plate detected | Check that the plate format matches Vietnamese patterns in `validator.py`. |
| Video processing is slow | Larger videos take more time. The pipeline processes every frame. |

---

## License

MIT

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [MuhammadMoinFaisal](https://github.com/MuhammadMoinFaisal/Automatic_Number_Plate_Detection_Recognition_YOLOv8) – YOLOv8 ANPR reference
