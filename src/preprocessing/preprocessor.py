import cv2
import numpy as np
import yaml
import logging
from typing import Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class FramePreprocessor:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["preprocessing"]

        self.img_size = self.config.get("img_size", 640)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.get("clahe_clip_limit", 2.0),
            tileGridSize=tuple(self.config.get("clahe_tile_grid_size", [8, 8]))
        )
        self.gamma_threshold = self.config.get("gamma_correction_threshold", 80)

    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, (r, r), (dw, dh)

    def process_frame(self, frame: np.ndarray, frame_id: int, source: str = "video") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Applies CLAHE, denoise, deblur, and gamma correction per requirements."""
        # 1. Resize + Letterbox
        resized, ratio, pad = self.letterbox(frame, new_shape=self.img_size)
        
        # 2. CLAHE Contrast enhancement
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = self.clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 3. Bilateral Filter for denoising
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # 4. Wiener Filter approximation (unsharp masking for real-time deblur proxy)
        gaussian = cv2.GaussianBlur(denoised, (9, 9), 10.0)
        deblurred = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

        # 5. Gamma Correction (Conditional)
        mean_brightness = np.mean(cv2.cvtColor(deblurred, cv2.COLOR_BGR2GRAY))
        if mean_brightness < self.gamma_threshold:
            gamma = 1.5
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            deblurred = cv2.LUT(deblurred, table)
            
        metadata = {
            "source": source,
            "frame_id": frame_id,
            "mean_brightness": float(mean_brightness)
        }
        
        return deblurred, metadata
