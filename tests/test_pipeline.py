"""
Aero-Watch Test Suite
======================
Run: pytest tests/test_pipeline.py -v
"""

import io
import json
import numpy as np
import pytest
from PIL import Image

from src.data.data_pipeline import (
    ImagePreprocessor, CameraMetadata, ImageIngestionService,
)
from src.models.bird_detector import BirdDetection, FrameResult, ShadowDiscriminator
from src.models.color_classifier import BirdColorClassifier, BirdColor, COLOR_NAMES


@pytest.fixture
def sample_image_bytes():
    img = Image.new("RGB", (1920, 1080), (34, 85, 34))
    for x in range(400, 440):
        for y in range(300, 330):
            img.putpixel((x, y), (20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def camera_metadata():
    return CameraMetadata(
        camera_id="CAM_001", sector="sector_A",
        latitude=0.3476, longitude=32.5825,
        timestamp="2025-06-15T08:30:00Z",
        battery_level=85.0, temperature_celsius=28.5,
        humidity_percent=92.0, firmware_version="2.1.0",
    )


@pytest.fixture
def sample_config():
    return {
        "model": {"detector": {"input_size": 640}},
        "data": {"max_image_size_mb": 10, "supported_formats": ["jpg", "jpeg", "png"]},
    }


class TestPreprocessor:
    def test_process_valid(self, sample_image_bytes, camera_metadata):
        pre = ImagePreprocessor(target_size=640)
        result = pre.process(sample_image_bytes, camera_metadata)
        assert result.processed_array.shape == (640, 640, 3)
        assert result.processed_array.dtype == np.uint8

    def test_reject_small(self, camera_metadata):
        tiny = Image.new("RGB", (100, 100))
        buf = io.BytesIO()
        tiny.save(buf, format="JPEG")
        with pytest.raises(ValueError, match="too small"):
            ImagePreprocessor().process(buf.getvalue(), camera_metadata)


class TestFrameResult:
    def test_counting(self):
        r = FrameResult("id", "cam", "ts", [
            BirdDetection([0, 0, 50, 50], 0.9, color_label="black", color_confidence=0.8),
            BirdDetection([100, 100, 150, 150], 0.8, color_label="green", color_confidence=0.7),
            BirdDetection([200, 200, 250, 250], 0.7, color_label="black", color_confidence=0.6),
        ])
        assert r.total_bird_count == 3
        assert r.black_bird_count == 2
        assert r.has_black_birds is True
        assert r.count_by_color() == {"black": 2, "green": 1}


class TestShadowDiscriminator:
    def test_uniform_dark_is_shadow(self):
        crop = np.full((50, 50, 3), 30, dtype=np.uint8)
        _, feat = ShadowDiscriminator().is_shadow(crop)
        assert feat["texture_variance"] < 100

    def test_textured_dark_has_texture(self):
        crop = np.random.randint(10, 60, (50, 50, 3), dtype=np.uint8)
        _, feat = ShadowDiscriminator().is_shadow(crop)
        assert feat["texture_variance"] > 0


class TestColorClassifier:
    def test_black(self):
        dark = np.full((50, 50, 3), 20, dtype=np.uint8)
        clf = BirdColorClassifier(confidence_threshold=0.3)
        clf.load()
        r = clf.classify(dark)
        assert r.primary_color == "black"
        assert r.is_black_bird is True

    def test_white(self):
        bright = np.full((50, 50, 3), 240, dtype=np.uint8)
        clf = BirdColorClassifier(confidence_threshold=0.3)
        clf.load()
        r = clf.classify(bright)
        assert r.primary_color == "white"

    def test_probs_sum_to_one(self):
        crop = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        clf = BirdColorClassifier()
        clf.load()
        r = clf.classify(crop)
        assert abs(sum(r.probabilities.values()) - 1.0) < 0.05
