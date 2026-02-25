"""
Aero-Watch Model Monitoring & Drift Detection
===============================================

Tracks model behaviour over time WITHOUT ground truth labels.

Three complementary drift signals:

  1. DATA DRIFT — input image distribution changes
     Seasonal shifts change the background the model sees. A forest that
     turns from lush green to dry brown shifts the RGB distribution,
     brightness, and edge density. If the model was trained on green-season
     data, dry-season inputs are out-of-distribution.

     Metrics: green dominance ratio, mean RGB, brightness, edge density.
     Method:  Population Stability Index (PSI) per feature vs. reference.

  2. PREDICTION DRIFT — model output distribution changes
     Even without labels, if the colour distribution suddenly shifts
     (e.g. "brown" jumps from 15% to 40% of predictions) that's a signal.
     Same for detection counts, confidence scores, and black bird ratio.
     You can't tell if the model is WRONG, but you can tell it's behaving
     DIFFERENTLY — which warrants investigation.

     Metrics: colour distribution histogram, mean confidence, detection
              rate, black bird ratio.
     Method:  Jensen-Shannon divergence on colour histograms,
              PSI on scalar metrics.

  3. CONFIDENCE DRIFT — model uncertainty changes
     A well-calibrated model's confidence distribution is stable. If mean
     confidence drops or the low-confidence tail grows, the model is seeing
     inputs it's unsure about — classic OOD signal.

     Metrics: mean/std confidence, fraction below threshold, confidence
              histogram.
     Method:  PSI on confidence histogram bins.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

COLOR_NAMES = ["black", "white", "brown", "green", "red", "blue", "yellow", "grey", "mixed"]


# ---------------------------------------------------------------------------
# Distribution Statistics (computed from a window of images + predictions)
# ---------------------------------------------------------------------------

@dataclass
class WindowStats:
    """Statistics from a window of N images + their predictions."""
    timestamp: str
    num_images: int

    # ── Data features (from input images) ──
    mean_rgb: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    std_rgb: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    green_dominance_ratio: float = 0.0   # G / (R+G+B) — seasonal indicator
    mean_brightness: float = 0.0
    brightness_std: float = 0.0
    mean_edge_density: float = 0.0       # vegetation density proxy

    # ── Prediction features (from model outputs, no GT needed) ──
    color_distribution: dict[str, float] = field(default_factory=dict)
    mean_detections_per_image: float = 0.0
    mean_confidence: float = 0.0
    confidence_std: float = 0.0
    low_confidence_fraction: float = 0.0  # fraction with conf < 0.5
    black_bird_ratio: float = 0.0
    detection_rate: float = 0.0           # fraction of images with ≥1 bird


# Backward compatibility alias
ImageDistributionStats = WindowStats


# ---------------------------------------------------------------------------
# Drift Detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Detects data and prediction drift using statistical tests.
    Does NOT require ground truth labels.

    Usage:
        detector = DriftDetector()

        # Set reference from training-time data (or first production window)
        detector.set_reference(reference_stats)

        # Every N images, compute current stats and check for drift
        report = detector.check_drift(current_stats)
        if report["drift_detected"]:
            trigger_alert(report)
    """

    def __init__(
        self,
        psi_threshold: float = 0.20,        # PSI > 0.2 = significant drift
        js_threshold: float = 0.10,          # JS divergence > 0.1 = colour shift
        confidence_drop_threshold: float = 0.05,  # mean conf drop > 5%
    ):
        self.psi_threshold = psi_threshold
        self.js_threshold = js_threshold
        self.conf_drop_threshold = confidence_drop_threshold
        self.reference: Optional[WindowStats] = None
        self._history: list[WindowStats] = []

    # ── Compute stats from raw data ──────────────────────────────────

    def compute_image_stats(
        self,
        images: list[np.ndarray],
        predictions: Optional[list[dict]] = None,
    ) -> WindowStats:
        """
        Compute distribution stats from a batch of images + their predictions.

        Args:
            images: list of HWC uint8 RGB arrays
            predictions: list of result dicts from InferenceEngine, each with
                         keys: total_birds, detections[{confidence, color}], etc.
                         If None, only data features are computed.
        """
        if not images:
            return WindowStats(
                timestamp=datetime.now(timezone.utc).isoformat(),
                num_images=0,
            )

        # ── Data features ──
        all_means, all_stds = [], []
        all_brightness, all_green_ratios, all_edge_densities = [], [], []

        for img in images:
            img_f = img.astype(np.float32) / 255.0

            all_means.append(img_f.mean(axis=(0, 1)).tolist())
            all_stds.append(img_f.std(axis=(0, 1)).tolist())
            all_brightness.append(float(img_f.mean()))

            channel_sums = img_f.sum(axis=(0, 1))
            total = channel_sums.sum() + 1e-8
            all_green_ratios.append(float(channel_sums[1] / total))

            gray = np.mean(img_f, axis=2)
            from scipy.ndimage import sobel
            sx = sobel(gray, axis=0)
            sy = sobel(gray, axis=1)
            edge_mag = np.sqrt(sx**2 + sy**2)
            all_edge_densities.append(float(np.mean(edge_mag > 0.1)))

        stats = WindowStats(
            timestamp=datetime.now(timezone.utc).isoformat(),
            num_images=len(images),
            mean_rgb=np.mean(all_means, axis=0).tolist(),
            std_rgb=np.mean(all_stds, axis=0).tolist(),
            green_dominance_ratio=float(np.mean(all_green_ratios)),
            mean_brightness=float(np.mean(all_brightness)),
            brightness_std=float(np.std(all_brightness)),
            mean_edge_density=float(np.mean(all_edge_densities)),
        )

        # ── Prediction features (no GT needed) ──
        if predictions:
            all_confs = []
            color_counts = {c: 0 for c in COLOR_NAMES}
            total_birds = 0
            images_with_birds = 0

            for pred in predictions:
                n_birds = pred.get("total_birds", 0)
                total_birds += n_birds
                if n_birds > 0:
                    images_with_birds += 1

                for det in pred.get("detections", []):
                    conf = det.get("confidence", 0)
                    color = det.get("color", "mixed")
                    all_confs.append(conf)
                    if color in color_counts:
                        color_counts[color] += 1

            total_dets = max(sum(color_counts.values()), 1)
            stats.color_distribution = {
                c: round(n / total_dets, 4) for c, n in color_counts.items()
            }
            stats.mean_detections_per_image = total_birds / max(len(predictions), 1)
            stats.detection_rate = images_with_birds / max(len(predictions), 1)

            if all_confs:
                stats.mean_confidence = float(np.mean(all_confs))
                stats.confidence_std = float(np.std(all_confs))
                stats.low_confidence_fraction = float(np.mean(
                    [c < 0.5 for c in all_confs]
                ))

            stats.black_bird_ratio = stats.color_distribution.get("black", 0.0)

        return stats

    # ── Reference management ─────────────────────────────────────────

    def set_reference(self, stats: WindowStats):
        self.reference = stats
        logger.info("Reference distribution set (%d images)", stats.num_images)

    def save_reference(self, path: str):
        if self.reference:
            Path(path).write_text(json.dumps(asdict(self.reference), indent=2))

    def load_reference(self, path: str):
        data = json.loads(Path(path).read_text())
        self.reference = WindowStats(**data)
        logger.info("Loaded reference from %s", path)

    # ── Drift detection ──────────────────────────────────────────────

    def check_drift(self, current: WindowStats) -> dict:
        """
        Compare current window against reference. Returns:
            {
                "drift_detected": bool,
                "data_drift": {...},
                "prediction_drift": {...},
                "confidence_drift": {...},
                "alerts": list[str],
                "severity": "none" | "warning" | "critical",
            }
        """
        if self.reference is None:
            return {"drift_detected": False, "alerts": ["No reference set."]}

        alerts = []

        # ── 1. DATA DRIFT ──
        data_psi = {}
        data_features = [
            ("green_dominance", self.reference.green_dominance_ratio,
             current.green_dominance_ratio),
            ("brightness", self.reference.mean_brightness,
             current.mean_brightness),
            ("edge_density", self.reference.mean_edge_density,
             current.mean_edge_density),
        ]
        for i, ch in enumerate(["red", "green", "blue"]):
            data_features.append((
                f"mean_{ch}", self.reference.mean_rgb[i], current.mean_rgb[i],
            ))

        for name, ref_val, cur_val in data_features:
            psi = self._psi_scalar(ref_val, cur_val)
            data_psi[name] = round(psi, 4)
            if psi > self.psi_threshold:
                alerts.append(
                    f"Data drift in {name}: PSI={psi:.3f} "
                    f"(ref={ref_val:.3f} → cur={cur_val:.3f})"
                )

        # Specific seasonal check
        green_shift = abs(
            current.green_dominance_ratio - self.reference.green_dominance_ratio
        )
        if green_shift > 0.05:
            direction = ("greener" if current.green_dominance_ratio >
                         self.reference.green_dominance_ratio else "less green")
            alerts.append(
                f"Seasonal shift: forest is {direction} "
                f"(Δgreen={green_shift:.3f})"
            )

        # ── 2. PREDICTION DRIFT ──
        pred_drift = {}

        # Colour distribution: Jensen-Shannon divergence
        if current.color_distribution and self.reference.color_distribution:
            js = self._js_divergence(
                self.reference.color_distribution,
                current.color_distribution,
            )
            pred_drift["color_js_divergence"] = round(js, 4)
            if js > self.js_threshold:
                # Find which colours shifted most
                shifts = []
                for c in COLOR_NAMES:
                    ref_p = self.reference.color_distribution.get(c, 0)
                    cur_p = current.color_distribution.get(c, 0)
                    delta = cur_p - ref_p
                    if abs(delta) > 0.05:
                        shifts.append(f"{c} {ref_p:.0%}→{cur_p:.0%}")
                alerts.append(
                    f"Colour distribution shifted: JS={js:.3f} "
                    f"({', '.join(shifts) or 'subtle shift'})"
                )

        # Detection rate shift
        det_psi = self._psi_scalar(
            self.reference.detection_rate, current.detection_rate,
        )
        pred_drift["detection_rate_psi"] = round(det_psi, 4)
        if det_psi > self.psi_threshold:
            alerts.append(
                f"Detection rate shifted: "
                f"{self.reference.detection_rate:.0%} → {current.detection_rate:.0%}"
            )

        # Black bird ratio shift
        black_psi = self._psi_scalar(
            self.reference.black_bird_ratio, current.black_bird_ratio,
        )
        pred_drift["black_ratio_psi"] = round(black_psi, 4)
        if black_psi > self.psi_threshold:
            alerts.append(
                f"Black bird ratio shifted: "
                f"{self.reference.black_bird_ratio:.0%} → {current.black_bird_ratio:.0%}"
            )

        # ── 3. CONFIDENCE DRIFT ──
        conf_drift = {}
        conf_drop = self.reference.mean_confidence - current.mean_confidence
        conf_drift["mean_confidence_drop"] = round(conf_drop, 4)
        conf_drift["current_mean_confidence"] = round(current.mean_confidence, 4)
        conf_drift["current_low_conf_fraction"] = round(current.low_confidence_fraction, 4)

        if conf_drop > self.conf_drop_threshold:
            alerts.append(
                f"Confidence dropped: "
                f"{self.reference.mean_confidence:.3f} → {current.mean_confidence:.3f} "
                f"(Δ={conf_drop:.3f})"
            )

        low_conf_psi = self._psi_scalar(
            self.reference.low_confidence_fraction,
            current.low_confidence_fraction,
        )
        conf_drift["low_conf_psi"] = round(low_conf_psi, 4)
        if low_conf_psi > self.psi_threshold:
            alerts.append(
                f"More low-confidence detections: "
                f"{self.reference.low_confidence_fraction:.0%} → "
                f"{current.low_confidence_fraction:.0%}"
            )

        # ── Severity ──
        severity = "none"
        if len(alerts) >= 3:
            severity = "critical"
        elif len(alerts) >= 1:
            severity = "warning"

        self._history.append(current)

        return {
            "drift_detected": len(alerts) > 0,
            "data_drift": {"psi_scores": data_psi},
            "prediction_drift": pred_drift,
            "confidence_drift": conf_drift,
            "alerts": alerts,
            "severity": severity,
            "timestamp": current.timestamp,
            "psi_scores": {**data_psi, **pred_drift, **conf_drift},
        }

    # ── Statistical methods ──────────────────────────────────────────

    @staticmethod
    def _psi_scalar(ref_val: float, cur_val: float) -> float:
        """PSI for a single proportion/scalar."""
        eps = 1e-6
        r = max(abs(ref_val), eps)
        c = max(abs(cur_val), eps)
        return float((c - r) * np.log(c / r))

    @staticmethod
    def _js_divergence(p_dist: dict[str, float], q_dist: dict[str, float]) -> float:
        """
        Jensen-Shannon divergence between two colour distributions.
        Symmetric, bounded [0, ln(2)], no zero-handling issues.
        """
        keys = sorted(set(list(p_dist.keys()) + list(q_dist.keys())))
        eps = 1e-8
        p = np.array([p_dist.get(k, 0) + eps for k in keys])
        q = np.array([q_dist.get(k, 0) + eps for k in keys])
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        kl_pm = float(np.sum(p * np.log(p / m)))
        kl_qm = float(np.sum(q * np.log(q / m)))
        return 0.5 * (kl_pm + kl_qm)

    # ── Trend analysis ───────────────────────────────────────────────

    def get_trend(self, feature: str, last_n: int = 10) -> list[float]:
        """Get recent values for a feature across windows."""
        values = []
        for s in self._history[-last_n:]:
            val = getattr(s, feature, None)
            if val is not None:
                values.append(float(val) if not isinstance(val, dict) else 0.0)
        return values
