import cv2
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    lang="en",
    enable_mkldnn=False,
)

img = cv2.imread("data/images/test.jpg")
result = ocr.ocr(img)

print("Result type:", type(result))
print("Result len:", len(result))
print("First item type:", type(result[0]))
print("First item attrs:", [a for a in dir(result[0]) if not a.startswith("_")])
print("First item keys:", list(result[0].keys()))
for k, v in result[0].items():
    print(f"  {k}: type={type(v).__name__}, val={str(v)[:200]}")
