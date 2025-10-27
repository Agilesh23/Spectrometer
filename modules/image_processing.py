from typing import Optional
import numpy as np
try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False


def load_image_from_upload(file) -> Optional[np.ndarray]:
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required. On Windows: pip install opencv-python. On Linux/Raspberry Pi: pip install opencv-python-headless")
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def load_image_from_camera(index: int = 0) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def apply_rotation(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for rotation. Install opencv-python/opencv-python-headless as appropriate for your OS.")
    if angle_deg == 0:
        return img
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def apply_crop_margins(img: np.ndarray, left: int, right: int, top: int, bottom: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, left)
    x2 = max(0, w - right)
    y1 = max(0, top)
    y2 = max(0, h - bottom)
    if x1 >= x2 or y1 >= y2:
        return img
    return img[y1:y2, x1:x2]


def adjust_brightness_contrast(img: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    # brightness: -100..100, contrast: -50..100 (rough)
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for brightness/contrast adjustment. Install opencv-python/opencv-python-headless.")
    beta = float(brightness)
    alpha = 1.0 + (contrast / 50.0)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for color conversion. Install opencv-python/opencv-python-headless.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def collapse_vertical(gray: np.ndarray, mode: str = "sum") -> np.ndarray:
    if mode == "mean":
        return gray.mean(axis=0)
    return gray.sum(axis=0)
