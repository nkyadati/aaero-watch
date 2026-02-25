"""
Aero-Watch REST API — PRODUCTION VERSION
==========================================
SQLite-backed persistence: detection history, aggregate stats, and
drift baselines survive API restarts.
"""

import io
import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.device import get_device
from src.models.bird_detector import BirdDetector, ShadowDiscriminator
from src.models.color_classifier import BirdColorClassifier
from src.inference.inference_engine import InferenceEngine, CLIPBirdVerifier
from src.data.data_pipeline import ImagePreprocessor, CameraMetadata, ProcessedImage
from src.monitoring.drift_detector import DriftDetector, ImageDistributionStats
from src.db import DetectionStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App State
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.engine: Optional[InferenceEngine] = None
        self.preprocessor: Optional[ImagePreprocessor] = None
        self.drift_detector: Optional[DriftDetector] = None
        self.store: Optional[DetectionStore] = None
        self.start_time = time.time()

        # Transient buffers for drift computation (not persisted —
        # drift needs the actual pixel arrays which don't belong in SQLite)
        self._image_buffer: list[np.ndarray] = []
        self._pred_buffer: list[dict] = []
        self._drift_window = int(os.getenv("DRIFT_WINDOW", "100"))
        self.last_drift_report: Optional[dict] = None


state = AppState()


# ---------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and connect to database on startup."""
    device = get_device()
    clip_device = "cpu" if device == "mps" else device
    logger.info("Starting Aero-Watch API on device=%s", device)

    # --- Database ---
    db_path = os.getenv("AEROWATCH_DB", "aerowatch.db")
    state.store = DetectionStore(db_path)
    state.store.connect()

    # --- Config from env ---
    detector_path = os.getenv("MODEL_DETECTOR_PATH", "yolo26x.pt")
    color_path = os.getenv("MODEL_COLOR_PATH", "") or None
    conf_thresh = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
    input_size = int(os.getenv("INPUT_SIZE", "1280"))
    use_clip = os.getenv("USE_CLIP_VERIFIER", "true").lower() == "true"

    logger.info("Config: detector=%s color=%s conf=%.2f clip=%s db=%s",
                 detector_path, color_path or "hybrid", conf_thresh, use_clip, db_path)

    # --- Models ---
    detector = BirdDetector(
        model_path=detector_path,
        input_size=input_size,
        confidence_threshold=conf_thresh,
        device=device,
    )
    classifier = BirdColorClassifier(
        model_path=color_path,
        device=clip_device,
    )

    clip_verifier = None
    if use_clip:
        clip_verifier = CLIPBirdVerifier(device=clip_device)

    state.engine = InferenceEngine(
        detector, classifier, ShadowDiscriminator(), clip_verifier=clip_verifier,
    )
    state.engine.load_models()
    state.preprocessor = ImagePreprocessor(target_size=input_size)
    state.drift_detector = DriftDetector(psi_threshold=0.20)

    # --- Restore drift state from database ---
    saved_baseline = state.store.load_drift_baseline()
    if saved_baseline:
        state.drift_detector.set_reference(ImageDistributionStats(**saved_baseline))
        logger.info("Restored drift baseline from database (%d images)",
                     saved_baseline.get("num_images", 0))
    else:
        # Try file-based reference as fallback
        ref_path = Path("models/reference_stats.json")
        if ref_path.exists():
            import json as _json
            ref_data = _json.loads(ref_path.read_text())
            state.drift_detector.set_reference(ImageDistributionStats(**ref_data))
            logger.info("Loaded reference distribution from %s", ref_path)
        else:
            logger.info("No drift baseline found — will build from first %d images",
                         state._drift_window)

    saved_report = state.store.load_drift_report()
    if saved_report:
        state.last_drift_report = saved_report
        logger.info("Restored last drift report (severity=%s)",
                     saved_report.get("severity", "unknown"))

    existing = state.store.get_detection_count()
    if existing > 0:
        logger.info("Resuming with %d existing detections in database", existing)

    logger.info("Aero-Watch API ready.")
    yield

    # --- Shutdown ---
    state.store.close()
    logger.info("Shutting down.")


app = FastAPI(
    title="Aero-Watch API",
    description="Bird Detection & Colour Classification",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS — allow dashboard to talk to API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DetectionOut(BaseModel):
    bbox: list[float]
    confidence: float
    class_name: str
    color: str
    color_confidence: float


class FrameResultOut(BaseModel):
    image_id: str
    camera_id: str
    timestamp: str
    total_birds: int
    has_black_birds: bool
    black_bird_count: int
    color_counts: dict[str, int]
    detections: list[DetectionOut]
    processing_time_ms: float


class HealthOut(BaseModel):
    status: str
    device: str
    uptime_seconds: float
    detections_stored: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthOut)
async def health():
    return HealthOut(
        status="healthy" if state.engine else "starting",
        device=get_device(),
        uptime_seconds=round(time.time() - state.start_time, 1),
        detections_stored=state.store.get_detection_count() if state.store else 0,
    )


@app.post("/detect", response_model=FrameResultOut)
async def detect_birds(
    image: UploadFile = File(...),
    camera_id: str = Query("default"),
    sector: str = Query("default"),
):
    """Upload an image and get bird detection results."""
    t0 = time.time()

    raw_bytes = await image.read()
    if len(raw_bytes) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds 10 MB")

    meta = CameraMetadata(
        camera_id=camera_id, sector=sector,
        latitude=0, longitude=0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        battery_level=100, temperature_celsius=25,
        humidity_percent=80, firmware_version="api",
    )

    processed = state.preprocessor.process(raw_bytes, meta)
    result = state.engine.process_image(processed)
    result_dict = result.to_dict()

    elapsed_ms = (time.time() - t0) * 1000

    # --- Persist to SQLite ---
    record = {
        "image_id": result_dict["image_id"],
        "camera_id": result_dict["camera_id"],
        "sector": sector,
        "timestamp": result_dict["timestamp"],
        "total_birds": result_dict["total_birds"],
        "black_bird_count": result_dict["black_bird_count"],
        "color_counts": result_dict["color_counts"],
        "detections": result_dict["detections"],
        "processing_time_ms": round(elapsed_ms, 1),
    }
    state.store.insert_detection(record)

    # --- Drift monitoring: buffer image + prediction, check periodically ---
    state._image_buffer.append(processed.processed_array)
    state._pred_buffer.append(result_dict)
    if len(state._image_buffer) >= state._drift_window:
        _run_drift_check()

    return FrameResultOut(
        image_id=result_dict["image_id"],
        camera_id=result_dict["camera_id"],
        timestamp=result_dict["timestamp"],
        total_birds=result_dict["total_birds"],
        has_black_birds=result_dict["has_black_birds"],
        black_bird_count=result_dict["black_bird_count"],
        color_counts=result_dict["color_counts"],
        detections=[DetectionOut(**d) for d in result_dict["detections"]],
        processing_time_ms=round(elapsed_ms, 1),
    )


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    uptime = time.time() - state.start_time
    agg = state.store.get_aggregate_stats()

    lines = [
        f"# HELP aerowatch_images_processed Total images processed",
        f"# TYPE aerowatch_images_processed counter",
        f"aerowatch_images_processed {agg['images_processed']}",
        f"# HELP aerowatch_birds_detected Total birds detected",
        f"# TYPE aerowatch_birds_detected counter",
        f"aerowatch_birds_detected {agg['total_birds']}",
        f"# HELP aerowatch_black_birds_detected Total black birds detected",
        f"# TYPE aerowatch_black_birds_detected counter",
        f"aerowatch_black_birds_detected {agg['total_black_birds']}",
        f"# HELP aerowatch_avg_processing_ms Average processing time per image",
        f"# TYPE aerowatch_avg_processing_ms gauge",
        f"aerowatch_avg_processing_ms {agg['avg_latency_ms']:.1f}",
        f"# HELP aerowatch_uptime_seconds Service uptime",
        f"# TYPE aerowatch_uptime_seconds gauge",
        f"aerowatch_uptime_seconds {uptime:.1f}",
    ]

    # Add drift metrics if available
    if state.last_drift_report and state.last_drift_report.get("psi_scores"):
        for feature, psi in state.last_drift_report["psi_scores"].items():
            lines.append(f"# HELP aerowatch_drift_psi_{feature} PSI score for {feature}")
            lines.append(f"# TYPE aerowatch_drift_psi_{feature} gauge")
            lines.append(f"aerowatch_drift_psi_{feature} {psi}")
        severity_val = {"none": 0, "warning": 1, "critical": 2}.get(
            state.last_drift_report.get("severity", "none"), 0)
        lines.append(f"# HELP aerowatch_drift_severity Drift severity (0=none, 1=warning, 2=critical)")
        lines.append(f"# TYPE aerowatch_drift_severity gauge")
        lines.append(f"aerowatch_drift_severity {severity_val}")

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines) + "\n")


@app.get("/history")
async def detection_history(limit: int = Query(50, le=500)):
    """Return recent detection results for the dashboard."""
    return state.store.get_history(limit=limit)


@app.get("/stats")
async def dashboard_stats():
    """Aggregated stats for the dashboard."""
    uptime = time.time() - state.start_time
    agg = state.store.get_aggregate_stats()
    agg["uptime_seconds"] = round(uptime, 1)
    return agg


@app.get("/drift")
async def drift_status():
    """Return the latest drift detection report."""
    if state.last_drift_report is None:
        return {
            "status": "pending",
            "message": f"Collecting baseline — {len(state._image_buffer)}/{state._drift_window} images buffered",
        }
    return state.last_drift_report


@app.post("/reset")
async def reset_database():
    """Clear all stored detections and drift state. Useful for fresh runs."""
    state.store.reset()
    state.last_drift_report = None
    state._image_buffer.clear()
    state._pred_buffer.clear()
    if state.drift_detector:
        state.drift_detector.reference = None
    return {"status": "reset", "message": "All data cleared"}


def _run_drift_check():
    """Run drift detection on buffered images + predictions, then clear."""
    if not state.drift_detector or not state._image_buffer:
        return

    current_stats = state.drift_detector.compute_image_stats(
        state._image_buffer, predictions=state._pred_buffer,
    )

    if state.drift_detector.reference is None:
        # First window — use as reference baseline
        state.drift_detector.set_reference(current_stats)
        state.last_drift_report = {
            "status": "baseline_set",
            "timestamp": current_stats.timestamp,
            "num_images": current_stats.num_images,
            "green_dominance": round(current_stats.green_dominance_ratio, 4),
            "mean_brightness": round(current_stats.mean_brightness, 4),
            "color_distribution": current_stats.color_distribution,
            "mean_confidence": round(current_stats.mean_confidence, 4),
            "severity": "none",
        }
        # Persist baseline so it survives restarts
        state.store.save_drift_baseline({
            "num_images": current_stats.num_images,
            "timestamp": current_stats.timestamp,
            "green_dominance_ratio": current_stats.green_dominance_ratio,
            "mean_brightness": current_stats.mean_brightness,
            "edge_density": current_stats.edge_density,
            "channel_means": current_stats.channel_means,
            "channel_stds": current_stats.channel_stds,
            "color_distribution": current_stats.color_distribution,
            "mean_confidence": current_stats.mean_confidence,
        })
        logger.info("Drift detector: baseline set from %d images "
                     "(green=%.3f, brightness=%.3f, mean_conf=%.3f)",
                     current_stats.num_images,
                     current_stats.green_dominance_ratio,
                     current_stats.mean_brightness,
                     current_stats.mean_confidence)
    else:
        report = state.drift_detector.check_drift(current_stats)
        state.last_drift_report = report

        if report["drift_detected"]:
            logger.warning("DRIFT DETECTED [%s]: %s",
                            report["severity"], "; ".join(report["alerts"]))
        else:
            logger.info("Drift check OK (green=%.3f, brightness=%.3f, "
                         "color_js=%.4f, conf=%.3f)",
                         current_stats.green_dominance_ratio,
                         current_stats.mean_brightness,
                         report.get("prediction_drift", {}).get("color_js_divergence", 0),
                         current_stats.mean_confidence)

    # Persist drift report
    state.store.save_drift_report(state.last_drift_report)

    state._image_buffer.clear()
    state._pred_buffer.clear()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
