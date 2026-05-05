# Plate Vision

> Hệ thống nhận dạng biển số xe Việt Nam sử dụng **YOLOv8** + **PaddleOCR**

Hệ thống phát hiện và đọc biển số xe cho phương tiện Việt Nam. Hỗ trợ tải ảnh lên và xử lý file video từng khung hình, trả về file đã ghi chú và danh sách biển số không trùng lặp.

---

## Tính năng

- **Phát hiện biển số** bằng YOLOv8 (`best.pt`)
- **Nhận dạng ký tự** bằng PaddleOCR (PP-OCRv5)
- **Kiểm định biển số Việt Nam** với regex và sửa lỗi OCR
- **Giao diện Web** xây dựng bằng Flask + Tailwind CSS
- **Tải ảnh lên** – nhận diện một khung hình với kết quả đã ghi chú
- **Tải video lên** – xử lý từng khung hình, tải video đầu ra đã ghi chú
- **Tiền xử lý ảnh** (CLAHE, khử nhiễu, ngưỡng thích ứng)
- **OCR đa lần thử** – thử cả ảnh gốc và ảnh đã tăng cường, chọn kết quả tốt nhất
- **Đa nền tảng** – hoạt động trên Windows, Linux, macOS

---

## Kiến trúc

```
Đầu vào (Ảnh / Video)
    │
    ▼
┌─────────────────────────────┐
│  Phát hiện YOLOv8           │  ← models/best.pt
│  (Vùng biển số)             │
└────────────┬────────────────┘
             │ cắt vùng biển số
             ▼
┌─────────────────────────────┐
│  Tiền xử lý ảnh             │  ← resize, grayscale, CLAHE,
│                             │    khử nhiễu, ngưỡng thích ứng
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  PaddleOCR                  │  ← PP-OCRv5 Server Det + EN Mobile Rec
│  (Nhận dạng ký tự)          │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Chuẩn hóa & Kiểm định      │  ← regex + sửa lỗi OCR phổ biến
└────────────┬────────────────┘
             │
             ▼
    Ảnh / Video đã ghi chú + JSON
```

---

## Cấu trúc dự án

```
plate-vision/
├── app.py                  # Máy chủ Flask (REST API + UI)
├── main.py                 # Điểm vào CLI (ảnh / video)
├── config.py               # Cấu hình tập trung
├── detector.py             # Wrapper phát hiện YOLOv8
├── ocr_engine.py           # Wrapper PaddleOCR với logic fallback
├── preprocessor.py         # Pipeline tiền xử lý ảnh
├── validator.py            # Kiểm định regex & sửa lỗi OCR
├── utils.py                # Tiện ích vẽ & logging
├── requirements.txt        # Thư viện Python
├── setup.bat               # Script cài đặt Windows (CMD)
├── setup.ps1               # Script cài đặt Windows (PowerShell)
├── PLAN.md                 # Kế hoạch phát triển
│
├── models/                 # Trọng số mô hình
│   └── best.pt             # Bộ phát hiện biển số YOLOv8 (52 MB)
│
├── templates/
│   └── index.html          # Giao diện web Tailwind CSS
│
├── data/
│   ├── images/             # Ảnh thử nghiệm
│   └── videos/             # Video thử nghiệm
│
└── output/                 # Kết quả nhận diện (ảnh & video)
```

---

## Cài đặt

### Yêu cầu

- Python **3.10 - 3.12** (PaddlePaddle có hỗ trợ hạn chế cho Python 3.13)
- Windows / Linux / macOS

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/plate-vision.git
cd plate-vision
```

### 2. Tạo Môi trường ảo

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

### 3. Cài đặt Thư viện

```bash
pip install -r requirements.txt
```

> **Lưu ý:** Lần chạy đầu tiên, PaddleOCR sẽ tự động tải các mô hình suy luận (`~100 MB`) về `~/.paddlex/official_models/`.

### 4. Tải Trọng số YOLOv8

Nếu thiếu `models/best.pt`, tải từ Google Drive:

```bash
pip install gdown
gdown "https://drive.google.com/uc?id=1dIyJooVaowaNUj0R1Q-HUnu-utiGsEj8" -O models/best.pt
```

---

## Sử dụng

### Giao diện Web (Khuyên dùng)

```bash
python app.py
```

Mở trình duyệt tại [`http://127.0.0.1:5000`](http://127.0.0.1:5000).

#### Tải ảnh lên
- Chuyển sang tab **Upload Image**
- Kéo thả hoặc chọn ảnh (JPG, PNG, WEBP)
- Nhấn **Detect License Plate**
- Xem kết quả đã ghi chú và các biển số phát hiện được

#### Tải video lên
- Chuyển sang tab **Upload Video**
- Kéo thả hoặc chọn video (MP4, AVI, MOV, MKV)
- Xem trước video, sau đó nhấn **Process Video**
- Đợi xử lý (tùy thuộc vào độ dài video)
- Tải video đầu ra đã ghi chú
- Xem danh sách các biển số không trùng lặp trong toàn bộ video

### Dòng lệnh

```bash
# Ảnh
python main.py --source data/images/test.jpg --save

# Video
python main.py --source data/videos/traffic.mp4 --save
```

---

## Cách thức hoạt động

### 1. Phát hiện
`detector.py` tải mô hình YOLOv8 (`models/best.pt`) để phát hiện khung bao quanh biển số trong mỗi khung hình đầu vào.

### 2. Tiền xử lý
`preprocessor.py` cắt vùng phát hiện được và áp dụng:
- Thay đổi kích thước (nếu chiều rộng < 200 px)
- Chuyển sang grayscale
- Tăng cường tương phản CLAHE
- Khử nhiệu non-local means
- Ngưỡng Gaussian thích ứng

### 3. OCR
`ocr_engine.py` chạy PaddleOCR trên cả **ảnh cắt gốc** và **ảnh đã tăng cường**, sau đó chọn kết quả có độ tin cậy cao hơn.

### 4. Kiểm định
`validator.py` chuẩn hóa văn bản và kiểm định theo mẫu biển số Việt Nam:

| Mẫu | Ví dụ |
|-----|-------|
| `\d{2}[A-Z]\d{5,6}` | `90B245230` (định dạng mới) |
| `\d{2}[A-Z]\d{4,5}` | `51A12345` |
| `\d{2}[A-Z]{1,2}\d{4,5}` | `51AB12345` |

Các lỗi đọc OCR phổ biến được tự động sửa (`O` → `0`, `I` → `1`, `S` → `5`, v.v.).

---

## Cấu hình

Chỉnh sửa `config.py` để tinh chỉnh pipeline:

```python
# Phát hiện
YOLO_CONF = 0.5           # Ngưỡng độ tin cậy
YOLO_IOU = 0.45           # Ngưỡng NMS IoU

# Tiền xử lý
MIN_PLATE_WIDTH = 200     # Phóng to nếu biển quá nhỏ
PADDING = 5               # Đệm cắt (pixel)

# OCR
OCR_LANG = "en"           # Ngôn ngữ PaddleOCR
OCR_USE_ANGLE_CLS = True  # Bật định hướng dòng chữ
```

---

## Xử lý sự cố

| Vấn đề | Giải pháp |
|--------|-----------|
| Tải mô hình thất bại | Tải thủ công `best.pt` bằng `gdown` (xem Bước 4 Cài đặt). |
| Biển số bị đánh dấu INVALID | Kiểm tra định dạng biển có khớp mẫu Việt Nam trong `validator.py`. |
| Xử lý video chậm | Video dài hơn sẽ mất nhiều thờii gian hơn. Pipeline xử lý từng khung hình. |

---

## Giấy phép

MIT

---

## Tài liệu tham khảo

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [MuhammadMoinFaisal](https://github.com/MuhammadMoinFaisal/Automatic_Number_Plate_Detection_Recognition_YOLOv8) – Tài liệu tham khảo YOLOv8 ANPR
