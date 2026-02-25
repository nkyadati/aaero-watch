"""
Aero-Watch Inference Engine — PRODUCTION VERSION
=================================================
Pipeline: detection → NMS → CLIP verification → shadow filter → colour classification → counting.
"""

import time
import logging
from typing import Optional

import numpy as np
from PIL import Image

from src.models.bird_detector import BirdDetector, BirdDetection, FrameResult, ShadowDiscriminator
from src.models.color_classifier import BirdColorClassifier
from src.data.data_pipeline import ProcessedImage, CameraMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLIP-based Bird Verifier
# ---------------------------------------------------------------------------

class CLIPBirdVerifier:
    """
    Uses OpenCLIP to verify whether a detected crop is actually a bird.

    Compares each crop against bird vs non-bird text descriptions.
    Far more robust than heuristic colour/texture checks because CLIP
    understands the semantic content of the image.
    """

    BIRD_PROMPTS = [
        "a photo of a bird",
        "a bird perched on a branch",
        "a bird sitting on the ground",
        "a bird in a tropical forest",
        "a black bird in the wild",
        "a small bird",
    ]

    NON_BIRD_PROMPTS = [
        "a tree branch with no animals",
        "leaves and green foliage",
        "a tree trunk and bark",
        "moss on a log",
        "a shadow on the ground",
        "a blurry out of focus background",
        "green leaves on a branch",
        "a twig",
    ]

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu"):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._bird_features = None
        self._non_bird_features = None

    def load(self):
        """Load CLIP model and pre-encode text prompts."""
        import open_clip
        import torch

        logger.info("Loading CLIP (%s) for crop verification...", self.model_name)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()

        # Pre-encode all text prompts once
        with torch.no_grad():
            bird_tokens = self.tokenizer(self.BIRD_PROMPTS).to(self.device)
            self._bird_features = self.model.encode_text(bird_tokens)
            self._bird_features /= self._bird_features.norm(dim=-1, keepdim=True)

            non_bird_tokens = self.tokenizer(self.NON_BIRD_PROMPTS).to(self.device)
            self._non_bird_features = self.model.encode_text(non_bird_tokens)
            self._non_bird_features /= self._non_bird_features.norm(dim=-1, keepdim=True)

        logger.info("CLIP verifier ready.")

    def is_bird(self, crop: np.ndarray, threshold: float = 0.0) -> tuple[bool, dict]:
        """
        Returns (is_bird, debug_info).
        is_bird = True when the best bird-prompt similarity exceeds best non-bird.
        """
        import torch

        pil_img = Image.fromarray(crop).convert("RGB")
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_features = self.model.encode_image(img_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            bird_sims = (img_features @ self._bird_features.T)[0].cpu().numpy()
            best_bird_idx = int(np.argmax(bird_sims))
            bird_score = float(bird_sims[best_bird_idx])

            non_bird_sims = (img_features @ self._non_bird_features.T)[0].cpu().numpy()
            best_nb_idx = int(np.argmax(non_bird_sims))
            non_bird_score = float(non_bird_sims[best_nb_idx])

        margin = bird_score - non_bird_score
        return margin > threshold, {
            "bird_score": round(bird_score, 3),
            "non_bird_score": round(non_bird_score, 3),
            "margin": round(margin, 3),
            "best_bird_prompt": self.BIRD_PROMPTS[best_bird_idx],
            "best_non_bird_prompt": self.NON_BIRD_PROMPTS[best_nb_idx],
        }


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """End-to-end inference pipeline."""

    def __init__(
        self,
        detector: BirdDetector,
        classifier: BirdColorClassifier,
        shadow_discriminator: Optional[ShadowDiscriminator] = None,
        clip_verifier: Optional[CLIPBirdVerifier] = None,
        min_crop_size: int = 16,
    ):
        self.detector = detector
        self.classifier = classifier
        self.shadow_disc = shadow_discriminator or ShadowDiscriminator()
        self.clip_verifier = clip_verifier  # None = skip verification
        self.min_crop_size = min_crop_size

    def load_models(self):
        self.detector.load()
        self.classifier.load()
        if self.clip_verifier:
            self.clip_verifier.load()
        logger.info("All models loaded.")

    def process_image(self, processed_image: ProcessedImage) -> FrameResult:
        t0 = time.time()
        image = processed_image.processed_array
        meta = processed_image.camera_metadata

        # Step 1: Detect
        raw_detections = self.detector.detect(image)
        logger.info("Raw detections: %d (%.2fs)", len(raw_detections), time.time() - t0)

        # Step 1.5: NMS + containment suppression
        raw_detections = self._nms(raw_detections, iou_threshold=0.40)
        logger.info("After NMS: %d detections", len(raw_detections))

        # Step 2: Verify + classify each crop
        valid_detections: list[BirdDetection] = []
        for det in raw_detections:
            crop = self._extract_crop(image, det.bbox)
            if crop is None:
                continue

            # 2a: CLIP verification (if available)
            if self.clip_verifier:
                is_bird, clip_info = self.clip_verifier.is_bird(crop)
                if not is_bird:
                    logger.info(
                        "  🌿 CLIP rejected [%.0f,%.0f,%.0f,%.0f] "
                        "bird=%.3f non_bird=%.3f ('%s')",
                        *det.bbox,
                        clip_info["bird_score"], clip_info["non_bird_score"],
                        clip_info["best_non_bird_prompt"],
                    )
                    continue

            # 2b: Shadow filter
            if self._is_dark_crop(crop):
                is_shadow, feat = self.shadow_disc.is_shadow(crop)
                if is_shadow:
                    logger.debug("Filtered shadow at %s", det.bbox)
                    continue

            # 2c: Colour classification
            color_pred = self.classifier.classify(crop)
            det.color_label = color_pred.primary_color
            det.color_confidence = color_pred.confidence
            valid_detections.append(det)

        elapsed = time.time() - t0
        logger.info("Result: %d birds (%.2fs)", len(valid_detections), elapsed)

        return FrameResult(
            image_id=processed_image.image_id,
            camera_id=meta.camera_id,
            timestamp=meta.timestamp,
            detections=valid_detections,
        )

    def process_numpy(self, image: np.ndarray, image_id="direct", camera_id="local") -> FrameResult:
        from datetime import datetime, timezone
        meta = CameraMetadata(
            camera_id=camera_id, sector="local", latitude=0, longitude=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            battery_level=100, temperature_celsius=25,
            humidity_percent=80, firmware_version="local",
        )
        return self.process_image(ProcessedImage(
            image_id=image_id, camera_metadata=meta,
            original_path="", processed_array=image,
        ))

    def _extract_crop(self, image: np.ndarray, bbox: list[float]) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
        if (x2 - x1) < self.min_crop_size or (y2 - y1) < self.min_crop_size:
            return None
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def _is_dark_crop(crop: np.ndarray, threshold: float = 60.0) -> bool:
        return float(np.mean(crop)) < threshold

    @staticmethod
    def _box_area(b: list[float]) -> float:
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    @staticmethod
    def _intersection_area(a: list[float], b: list[float]) -> float:
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * \
               max(0, min(a[3], b[3]) - max(a[1], b[1]))

    @classmethod
    def _compute_iou(cls, a: list[float], b: list[float]) -> float:
        inter = cls._intersection_area(a, b)
        union = cls._box_area(a) + cls._box_area(b) - inter
        return inter / max(union, 1e-6)

    @classmethod
    def _is_contained(cls, inner: list[float], outer: list[float], threshold: float = 0.60) -> bool:
        inter = cls._intersection_area(inner, outer)
        return (inter / max(cls._box_area(inner), 1e-6)) >= threshold

    @classmethod
    def _nms(cls, detections: list[BirdDetection], iou_threshold: float = 0.40) -> list[BirdDetection]:
        """Greedy NMS + containment suppression."""
        if len(detections) <= 1:
            return detections
        dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep: list[BirdDetection] = []
        while dets:
            best = dets.pop(0)
            keep.append(best)
            dets = [d for d in dets
                    if cls._compute_iou(best.bbox, d.bbox) < iou_threshold
                    and not cls._is_contained(d.bbox, best.bbox)]
        return keep
