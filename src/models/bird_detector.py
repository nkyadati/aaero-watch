"""
Aero-Watch Bird Detection Model — PRODUCTION VERSION
=====================================================
YOLO26 pipeline.

YOLO26 (Ultralytics, January 2026):
  - End-to-end NMS-free inference (dual-head: one-to-one default)
  - ProgLoss + STAL for improved small object detection
  - Up to 43% faster CPU inference vs YOLO11
  - Drop-in replacement for YOLOv8 via same Ultralytics API
  - Model variants: yolo26n/s/m/l/x.pt
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class BirdDetection:
    bbox: list[float]
    confidence: float
    class_id: int = 0
    class_name: str = ""
    color_label: str = ""
    color_confidence: float = 0.0


@dataclass
class FrameResult:
    image_id: str
    camera_id: str
    timestamp: str
    detections: list[BirdDetection] = field(default_factory=list)

    @property
    def total_bird_count(self) -> int:
        return len(self.detections)

    @property
    def black_bird_count(self) -> int:
        return sum(1 for d in self.detections if d.color_label == "black")

    @property
    def has_black_birds(self) -> bool:
        return self.black_bird_count > 0

    def count_by_color(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.detections:
            label = d.color_label or "unknown"
            counts[label] = counts.get(label, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "total_birds": self.total_bird_count,
            "has_black_birds": self.has_black_birds,
            "black_bird_count": self.black_bird_count,
            "color_counts": self.count_by_color(),
            "detections": [
                {
                    "bbox": [round(b, 1) for b in d.bbox],
                    "confidence": round(d.confidence, 3),
                    "class_name": d.class_name,
                    "color": d.color_label,
                    "color_confidence": round(d.color_confidence, 3),
                }
                for d in self.detections
            ],
        }


# ---------------------------------------------------------------------------
# YOLO class IDs that map to birds in the COCO dataset
# ---------------------------------------------------------------------------
COCO_BIRD_CLASS_ID = 14  # "bird" in COCO 80-class


class BirdDetector:
    """
    YOLO26-based bird detector.
    Works on Mac (MPS), NVIDIA (CUDA), and CPU.

    YOLO26's native small-object improvements (ProgLoss/STAL) handle
    small and distant birds without the need for SAHI tiling.
    """

    def __init__(
        self,
        model_path: str = "yolo26x.pt",
        input_size: int = 1280,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_thresh = confidence_threshold
        self.iou_thresh = iou_threshold
        self.device = device
        self.model = None

    def load(self):
        """Load the YOLO26 model (auto-downloads weights if missing)."""
        from ultralytics import YOLO

        logger.info("Loading YOLO26 model: %s ...", self.model_path)
        self.model = YOLO(self.model_path)
        logger.info("YOLO26 loaded successfully (device=%s)", self.device)

    def detect(self, image: np.ndarray, bird_only: bool = True) -> list[BirdDetection]:
        """
        Run detection on a single image (HWC uint8 RGB).
        If bird_only=True, filters to COCO 'bird' class only.
        If bird_only=False, returns ALL detections (useful for general demo).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # classes=[14] filters to bird class in COCO
        kwargs = dict(
            source=image,
            imgsz=self.input_size,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            max_det=100,
            verbose=False,
            device=self.device,
        )
        if bird_only:
            kwargs["classes"] = [COCO_BIRD_CLASS_ID]

        results = self.model.predict(**kwargs)

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            names = r.names  # {0: 'person', ..., 14: 'bird', ...}
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                conf = float(box.conf[0].cpu())
                cls_id = int(box.cls[0].cpu())
                cls_name = names.get(cls_id, str(cls_id))
                detections.append(BirdDetection(
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                ))
        return detections


# ---------------------------------------------------------------------------
# Shadow Discriminator
# ---------------------------------------------------------------------------

class ShadowDiscriminator:
    """
    Distinguishes genuine black birds from shadows using 7-feature
    majority vote. No training needed.

    Features analysed from each dark crop:
      - Laplacian texture variance (feathers vs smooth shadow)
      - Sobel edge density (defined edges vs diffuse blobs)
      - HSV saturation (iridescent plumage vs desaturated shadow)
      - Aspect ratio (compact bird vs amorphous shadow)
      - Gradient orientation coherence (structured body vs random)
      - Center-surround contrast (distinct object vs blended shadow)
      - Frequency energy ratio (mid-freq feather detail vs low-freq only)

    Majority vote: ≥4 of 7 shadow signals → classify as shadow.
    """

    def __init__(
        self,
        texture_threshold: float = 15.0,
        edge_density_threshold: float = 0.08,
        min_saturation: float = 5.0,
        aspect_ratio_range: tuple[float, float] = (0.3, 3.0),
        coherence_threshold: float = 0.35,
        contrast_threshold: float = 10.0,
        freq_ratio_threshold: float = 0.25,
    ):
        self.texture_thresh = texture_threshold
        self.edge_thresh = edge_density_threshold
        self.min_sat = min_saturation
        self.ar_range = aspect_ratio_range
        self.coherence_thresh = coherence_threshold
        self.contrast_thresh = contrast_threshold
        self.freq_ratio_thresh = freq_ratio_threshold

    # ── Public API ────────────────────────────────────────────────────

    def is_shadow(self, crop: np.ndarray) -> tuple[bool, dict]:
        """Classify a dark crop as shadow or real bird.

        Returns (is_shadow: bool, features: dict).
        """
        features = self.extract_features(crop)
        return self._classify_rules(features), features

    def extract_features(self, crop: np.ndarray) -> dict:
        """Extract all 7 discriminative features from a crop."""
        gray = np.mean(crop, axis=2) if crop.ndim == 3 else crop.astype(np.float32)
        h, w = crop.shape[:2]

        features = {}

        # 1. Laplacian texture variance — feathers have high-freq detail
        features["texture_variance"] = float(self._laplacian_variance(gray))

        # 2. Sobel edge density — birds have well-defined edges
        features["edge_density"] = float(self._edge_density(gray))

        # 3. HSV saturation — shadows are desaturated
        features["avg_saturation"] = float(self._average_saturation(crop))

        # 4. Aspect ratio — birds are compact
        features["aspect_ratio"] = float(w / max(h, 1))

        # 5. Gradient orientation coherence — bird bodies have structured
        #    gradient patterns (contour, feather lines); shadows are random
        features["gradient_coherence"] = float(self._gradient_coherence(gray))

        # 6. Center-surround contrast — birds are distinct objects with
        #    different brightness from the background; shadows blend in
        features["center_surround_contrast"] = float(
            self._center_surround_contrast(gray)
        )

        # 7. Frequency energy ratio — birds have mid-frequency content
        #    (feather patterns, body structure); shadows are low-freq only
        features["freq_energy_ratio"] = float(self._frequency_energy_ratio(gray))

        return features

    # ── Rule-based classification ─────────────────────────────────────

    def _classify_rules(self, features: dict) -> bool:
        """Majority vote across 7 features. ≥4 shadow votes → shadow."""
        shadow_votes = 0

        if features["texture_variance"] < self.texture_thresh:
            shadow_votes += 1
        if features["edge_density"] < self.edge_thresh:
            shadow_votes += 1
        if features["avg_saturation"] < self.min_sat:
            shadow_votes += 1
        ar = features["aspect_ratio"]
        if ar < self.ar_range[0] or ar > self.ar_range[1]:
            shadow_votes += 1
        if features["gradient_coherence"] < self.coherence_thresh:
            shadow_votes += 1
        if features["center_surround_contrast"] < self.contrast_thresh:
            shadow_votes += 1
        if features["freq_energy_ratio"] < self.freq_ratio_thresh:
            shadow_votes += 1

        features["shadow_votes"] = shadow_votes
        features["vote_threshold"] = 4
        return shadow_votes >= 4

    # ── Feature extraction methods ────────────────────────────────────

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        """Texture measure — high for feathers, low for smooth shadows."""
        from scipy.signal import convolve2d
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        lap = convolve2d(gray.astype(np.float32), kernel, mode="same")
        return float(np.var(lap))

    @staticmethod
    def _edge_density(gray: np.ndarray) -> float:
        """Fraction of strong-edge pixels — high for defined bird outlines."""
        from scipy.ndimage import sobel
        sx = sobel(gray.astype(np.float32), axis=0)
        sy = sobel(gray.astype(np.float32), axis=1)
        mag = np.sqrt(sx**2 + sy**2)
        threshold = np.percentile(mag, 90)
        return float(np.mean(mag > threshold))

    @staticmethod
    def _average_saturation(crop: np.ndarray) -> float:
        """Mean saturation — near-zero for shadows, nonzero for iridescent plumage."""
        img = Image.fromarray(crop).convert("HSV")
        hsv_arr = np.array(img)
        return float(np.mean(hsv_arr[:, :, 1]))

    @staticmethod
    def _gradient_coherence(gray: np.ndarray) -> float:
        """Measures how structured/aligned gradient orientations are.

        Birds have coherent gradients along body contours and feather lines.
        Shadows have random or uniform gradients (no structure).

        Uses the structure tensor eigenvalue ratio — high coherence means
        a dominant gradient direction exists (like a body edge).
        """
        from scipy.ndimage import sobel, gaussian_filter

        gx = sobel(gray.astype(np.float32), axis=1)
        gy = sobel(gray.astype(np.float32), axis=0)

        # Structure tensor components, smoothed
        sigma = 2.0
        Jxx = gaussian_filter(gx * gx, sigma)
        Jyy = gaussian_filter(gy * gy, sigma)
        Jxy = gaussian_filter(gx * gy, sigma)

        # Eigenvalues of 2×2 structure tensor at each pixel
        trace = Jxx + Jyy
        det = Jxx * Jyy - Jxy * Jxy

        # Avoid division by zero
        trace_safe = np.maximum(trace, 1e-8)

        # Coherence = (λ1 - λ2)² / (λ1 + λ2)² = 1 - 4*det/trace²
        coherence = 1.0 - 4.0 * det / (trace_safe ** 2)
        coherence = np.clip(coherence, 0, 1)

        # Mean coherence across crop — higher = more structured
        return float(np.mean(coherence))

    @staticmethod
    def _center_surround_contrast(gray: np.ndarray) -> float:
        """Absolute brightness difference between center and border.

        Birds are distinct objects — their center brightness differs from
        the surrounding background. Shadows blend smoothly into the
        background with minimal center-surround contrast.
        """
        h, w = gray.shape
        cy, cx = h // 4, w // 4

        center = gray[cy:h - cy, cx:w - cx]
        if center.size == 0:
            return 0.0

        # Border = full crop minus center
        mask = np.ones_like(gray, dtype=bool)
        mask[cy:h - cy, cx:w - cx] = False
        border = gray[mask]
        if border.size == 0:
            return 0.0

        return float(abs(np.mean(center) - np.mean(border)))

    @staticmethod
    def _frequency_energy_ratio(gray: np.ndarray) -> float:
        """Ratio of mid-frequency to total energy in the frequency domain.

        Birds have mid-frequency content from feather patterns and body
        structure. Shadows are dominated by low-frequency (smooth gradients).
        A high ratio indicates texture/structure = likely a bird.
        """
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Create radial distance map
        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_r = np.sqrt(cx**2 + cy**2)

        # Low: 0–15% of radius, Mid: 15–50%, High: 50–100%
        low_mask = r < max_r * 0.15
        mid_mask = (r >= max_r * 0.15) & (r < max_r * 0.50)

        total_energy = np.sum(magnitude ** 2) + 1e-8
        mid_energy = np.sum(magnitude[mid_mask] ** 2)

        return float(mid_energy / total_energy)
