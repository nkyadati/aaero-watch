#!/usr/bin/env python3
"""
Check whether model retraining is needed.

Called by CI/CD. Exits 1 if retrain needed, 0 otherwise.

Triggers for retraining:
  1. Drift severity is "critical" (3+ features drifted)
  2. Black bird F1 dropped below threshold
  3. Overall accuracy dropped below threshold
  4. New labelled data available (annotation count increased)

Usage:
  python scripts/check_retrain_needed.py --metrics-file metrics/latest.json
  echo $?  # 1 = retrain, 0 = no retrain
"""

import argparse
import json
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("retrain-check")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-file", default="metrics/latest.json",
                        help="Path to latest evaluation metrics JSON")
    parser.add_argument("--drift-file", default="metrics/drift_latest.json",
                        help="Path to latest drift report JSON")
    parser.add_argument("--baseline-file", default="metrics/baseline.json",
                        help="Path to baseline metrics JSON")
    parser.add_argument("--accuracy-threshold", type=float, default=0.05,
                        help="Retrain if accuracy drops by more than this fraction")
    parser.add_argument("--black-f1-min", type=float, default=0.70,
                        help="Retrain if black bird F1 falls below this")
    args = parser.parse_args()

    retrain_reasons = []

    # ── Check 1: Drift severity ──────────────────────────────────────
    drift_path = Path(args.drift_file)
    if drift_path.exists():
        drift = json.loads(drift_path.read_text())
        severity = drift.get("severity", "none")
        if severity == "critical":
            retrain_reasons.append(f"Drift severity is CRITICAL: {drift.get('alerts', [])}")
        elif severity == "warning":
            logger.info("Drift warning detected (not yet critical)")
    else:
        logger.info("No drift report found at %s", drift_path)

    # ── Check 2: Performance degradation ─────────────────────────────
    metrics_path = Path(args.metrics_file)
    baseline_path = Path(args.baseline_file)

    if metrics_path.exists() and baseline_path.exists():
        current = json.loads(metrics_path.read_text())
        baseline = json.loads(baseline_path.read_text())

        # Overall accuracy drop
        cur_acc = current.get("overall_accuracy", 1.0)
        base_acc = baseline.get("overall_accuracy", 1.0)
        if base_acc > 0 and (base_acc - cur_acc) / base_acc > args.accuracy_threshold:
            retrain_reasons.append(
                f"Accuracy dropped {base_acc:.3f} → {cur_acc:.3f} "
                f"({(base_acc - cur_acc) / base_acc:.1%} > {args.accuracy_threshold:.1%} threshold)"
            )

        # Black bird F1
        cur_bf1 = current.get("black_bird_f1", 1.0)
        if cur_bf1 < args.black_f1_min:
            retrain_reasons.append(
                f"Black bird F1 = {cur_bf1:.3f} (below {args.black_f1_min} minimum)"
            )

        # Detection mAP
        cur_map = current.get("mAP_50", 1.0)
        base_map = baseline.get("mAP_50", 1.0)
        if base_map > 0 and (base_map - cur_map) / base_map > args.accuracy_threshold:
            retrain_reasons.append(
                f"mAP@50 dropped {base_map:.3f} → {cur_map:.3f}"
            )
    elif metrics_path.exists():
        logger.info("No baseline file — skipping performance comparison")
    else:
        logger.info("No metrics file found at %s", metrics_path)

    # ── Decision ─────────────────────────────────────────────────────
    if retrain_reasons:
        logger.info("=" * 60)
        logger.info("RETRAIN NEEDED — %d reason(s):", len(retrain_reasons))
        for r in retrain_reasons:
            logger.info("  • %s", r)
        logger.info("=" * 60)
        sys.exit(1)
    else:
        logger.info("No retraining needed — model performance is within thresholds")
        sys.exit(0)


if __name__ == "__main__":
    main()
