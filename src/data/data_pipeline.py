"""
Aero-Watch Data Pipeline
========================
Handles image ingestion from remote cameras, preprocessing, and storage.
Designed for satellite-connected solar-powered cameras in equatorial forests.
"""

import io
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class CameraMetadata:
    """Metadata transmitted alongside each captured image."""
    camera_id: str
    sector: str
    latitude: float
    longitude: float
    timestamp: str  # ISO-8601
    battery_level: float
    temperature_celsius: float
    humidity_percent: float
    firmware_version: str


@dataclass
class ProcessedImage:
    """Container for a preprocessed image ready for inference."""
    image_id: str
    camera_metadata: CameraMetadata
    original_path: str
    processed_array: np.ndarray  # HWC uint8
    preprocessing_log: dict = field(default_factory=dict)
    ingestion_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Image Preprocessor
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """
    Preprocesses raw camera images for the detection pipeline.

    Equatorial forest images suffer from:
      - Heavy green colour cast from canopy filtering
      - High dynamic range (bright sky gaps vs dark understorey)
      - Fog / moisture artefacts on the lens
      - Motion blur from wind-blown branches

    The preprocessor applies targeted corrections while preserving
    bird-relevant features (shape, colour, contrast against foliage).
    """

    def __init__(self, target_size: int = 1280):
        self.target_size = target_size

    # -- public API --------------------------------------------------------

    def process(self, raw_bytes: bytes, metadata: CameraMetadata) -> ProcessedImage:
        """Full preprocessing pipeline: validate → correct → resize → return."""
        log: dict = {}

        # 1. Decode & validate
        image = self._decode_and_validate(raw_bytes)
        log["original_size"] = image.size  # (W, H)

        # 2. Correct white-balance (reduce green cast)
        image = self._auto_white_balance(image)
        log["white_balance"] = True

        # 3. Adaptive histogram equalisation (CLAHE-like via Pillow)
        image = self._enhance_local_contrast(image)
        log["contrast_enhanced"] = True

        # 4. Mild sharpening to recover detail lost to compression
        image = image.filter(ImageFilter.SHARPEN)
        log["sharpened"] = True

        # 5. Resize keeping aspect ratio (letterbox for YOLO)
        image, pad_info = self._letterbox_resize(image, self.target_size)
        log["letterbox_pad"] = pad_info

        arr = np.array(image)  # HWC uint8 RGB
        image_id = self._compute_id(raw_bytes, metadata)

        return ProcessedImage(
            image_id=image_id,
            camera_metadata=metadata,
            original_path="",  # filled by storage layer
            processed_array=arr,
            preprocessing_log=log,
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _decode_and_validate(raw_bytes: bytes) -> Image.Image:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        w, h = img.size
        if w < 320 or h < 240:
            raise ValueError(f"Image too small ({w}x{h}); minimum is 320x240.")
        return img

    @staticmethod
    def _auto_white_balance(img: Image.Image) -> Image.Image:
        """Grey-world white balance — equalises per-channel means."""
        arr = np.array(img, dtype=np.float32)
        means = arr.mean(axis=(0, 1))
        global_mean = means.mean()
        scale = global_mean / (means + 1e-6)
        balanced = np.clip(arr * scale, 0, 255).astype(np.uint8)
        return Image.fromarray(balanced)

    @staticmethod
    def _enhance_local_contrast(img: Image.Image, factor: float = 1.4) -> Image.Image:
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    @staticmethod
    def _letterbox_resize(
        img: Image.Image, target: int
    ) -> tuple[Image.Image, dict]:
        """Resize with aspect-ratio preservation and grey padding."""
        w, h = img.size
        scale = min(target / w, target / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGB", (target, target), (114, 114, 114))
        pad_x = (target - new_w) // 2
        pad_y = (target - new_h) // 2
        canvas.paste(img_resized, (pad_x, pad_y))

        return canvas, {"scale": scale, "pad_x": pad_x, "pad_y": pad_y}

    @staticmethod
    def _compute_id(raw_bytes: bytes, meta: CameraMetadata) -> str:
        h = hashlib.sha256(raw_bytes + meta.camera_id.encode()).hexdigest()[:16]
        return f"{meta.camera_id}_{meta.timestamp}_{h}"


# ---------------------------------------------------------------------------
# Ingestion Service (Camera → Cloud)
# ---------------------------------------------------------------------------

class ImageIngestionService:
    """
    Receives images from satellite-connected cameras and pushes them
    through the preprocessing pipeline into cloud storage.

    In production this runs behind an API gateway; here we expose the
    core logic that any transport layer (HTTP / MQTT / SQS) can call.
    """

    def __init__(self, config: dict):
        self.preprocessor = ImagePreprocessor(
            target_size=config["model"]["detector"]["input_size"]
        )
        self.max_size_mb = config["data"]["max_image_size_mb"]
        self.supported_formats = set(config["data"]["supported_formats"])

    def ingest(self, raw_bytes: bytes, metadata_json: str) -> ProcessedImage:
        """Validate, preprocess, and return a ProcessedImage."""

        # Parse metadata
        meta_dict = json.loads(metadata_json)
        metadata = CameraMetadata(**meta_dict)

        # Size guard
        size_mb = len(raw_bytes) / (1024 * 1024)
        if size_mb > self.max_size_mb:
            raise ValueError(
                f"Image exceeds max size ({size_mb:.1f} MB > {self.max_size_mb} MB)"
            )

        # Preprocess
        processed = self.preprocessor.process(raw_bytes, metadata)
        logger.info(
            "Ingested image %s from camera %s (sector %s)",
            processed.image_id, metadata.camera_id, metadata.sector,
        )
        return processed


# ---------------------------------------------------------------------------
# Cloud Storage Abstraction
# ---------------------------------------------------------------------------

class StorageBackend:
    """
    Thin abstraction over cloud object storage (S3 / GCS / Azure Blob).
    In production, swap with boto3 / google-cloud-storage client.
    """

    def __init__(self, bucket: str, local_fallback: Optional[Path] = None):
        self.bucket = bucket
        self.local = local_fallback or Path("/tmp/aero_watch_storage")
        self.local.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes) -> str:
        path = self.local / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.info("Stored %s (%d bytes)", key, len(data))
        return str(path)

    def get(self, key: str) -> bytes:
        return (self.local / key).read_bytes()

    def list_keys(self, prefix: str = "") -> list[str]:
        results = []
        for p in self.local.rglob("*"):
            if p.is_file() and str(p.relative_to(self.local)).startswith(prefix):
                results.append(str(p.relative_to(self.local)))
        return sorted(results)
