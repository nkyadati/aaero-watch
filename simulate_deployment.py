#!/usr/bin/env python3
"""
Aero-Watch — End-to-End Deployment Simulation
===============================================

Simulates the full production lifecycle on your Mac. Works with ANY
folder of images — no CUB-200 dataset required.

What it does (5 stages):

  Stage 1 — DEPLOY:      Start the FastAPI server locally
  Stage 2 — SIMULATE:    Feed images to the API as if they were camera frames
  Stage 3 — OBSERVE:     Query /health, /drift, /metrics, /stats endpoints
  Stage 4 — DASHBOARD:   Save results JSON and open the HTML dashboard
  Stage 5 — RETRAIN:     Check drift report and decide if retraining is needed

Usage:
  # Any folder of images
  python3 simulate_deployment.py --images ./my_bird_photos/

  # With trained colour classifier
  python3 simulate_deployment.py --images ./field_photos/ \
    --color-model color_classifier_best.pt

  # Control how many images and which detector
  python3 simulate_deployment.py --images ./photos/ \
    --color-model color_classifier_best.pt \
    --detector yolo26x.pt \
    --num-images 50

  # CUB-200 still works too
  python3 simulate_deployment.py --images ./CUB_200_2011/images/

Prerequisites:
  pip install -r requirements.txt
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("simulate")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def wait_for_api(url: str, timeout: int = 120) -> bool:
    """Wait for the API to become healthy."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(f"{url}/health", timeout=3)
            data = json.loads(resp.read())
            if data.get("status") == "healthy":
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def collect_images(image_dir: str, max_images: int) -> list[Path]:
    """Recursively find images in any directory structure."""
    root = Path(image_dir)
    if not root.exists():
        logger.error("Directory not found: %s", root)
        sys.exit(1)

    all_images = sorted([
        p for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    ])

    if not all_images:
        logger.error("No images found in %s", root)
        sys.exit(1)

    # Spread evenly across the folder (take every Nth)
    if len(all_images) > max_images:
        step = max(1, len(all_images) // max_images)
        selected = all_images[::step][:max_images]
    else:
        selected = all_images

    return selected


def main():
    parser = argparse.ArgumentParser(description="Aero-Watch End-to-End Simulation")
    parser.add_argument("--images", required=True,
                        help="Path to any folder of bird images")
    parser.add_argument("--color-model", default=None,
                        help="Path to trained colour classifier weights (.pt)")
    parser.add_argument("--detector", default="yolo26n.pt",
                        help="YOLO model (default: yolo26n.pt for fast simulation)")
    parser.add_argument("--num-images", type=int, default=100,
                        help="Max number of images to feed through API")
    parser.add_argument("--use-clip", action="store_true",
                        help="Enable CLIP verifier (slower but fewer false positives)")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--dashboard", default="dashboard.html",
                        help="Path to dashboard HTML file")
    args = parser.parse_args()

    api_url = f"http://127.0.0.1:{args.api_port}"

    # Collect images
    test_images = collect_images(args.images, args.num_images)

    print()
    print("=" * 70)
    print("  🦅  AERO-WATCH — End-to-End Deployment Simulation")
    print("=" * 70)
    print(f"  Images:     {args.images} ({len(test_images)} selected)")
    print(f"  Detector:   {args.detector}")
    print(f"  Classifier: {args.color_model if args.color_model else '(heuristic/hybrid)'}")
    print(f"  CLIP:       {'on' if args.use_clip else 'off'}")
    print("=" * 70)
    print()

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 1: DEPLOY — Start FastAPI server
    # ═══════════════════════════════════════════════════════════════════
    print("━" * 70)
    print("  Stage 1/5: DEPLOY — Starting API server")
    print("━" * 70)
    print()

    env = os.environ.copy()
    env["MODEL_DETECTOR_PATH"] = args.detector
    if args.color_model:
        env["MODEL_COLOR_PATH"] = args.color_model
    env["USE_CLIP_VERIFIER"] = "true" if args.use_clip else "false"
    env["CONFIDENCE_THRESHOLD"] = "0.35"

    server_log = open("api_server.log", "w")
    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "src.api.app:app",
            "--host", "127.0.0.1",
            "--port", str(args.api_port),
            "--log-level", "info",
        ],
        env=env,
        stdout=server_log,
        stderr=server_log,
    )

    logger.info("Waiting for API to start (log: api_server.log)...")
    if not wait_for_api(api_url, timeout=180):
        server_proc.kill()
        print()
        print("  ❌ API failed to start. Last 30 lines of log:")
        print("  " + "─" * 60)
        try:
            server_log.close()
            for line in open("api_server.log").readlines()[-30:]:
                print(f"  {line.rstrip()}")
        except Exception:
            pass
        print("  " + "─" * 60)
        print()
        print("  Try running the API directly to debug:")
        print(f"    MODEL_COLOR_PATH={args.color_model or ''} \\")
        print(f"    python3 -m uvicorn src.api.app:app --host 127.0.0.1 --port {args.api_port}")
        sys.exit(1)

    logger.info("✅ API is healthy at %s", api_url)

    # Reset database for a fresh simulation run
    import urllib.request
    try:
        req = urllib.request.Request(f"{api_url}/reset", method="POST")
        urllib.request.urlopen(req, timeout=5)
        logger.info("Database reset for fresh run")
    except Exception:
        pass  # OK if reset endpoint doesn't exist

    print()

    try:
        # ═══════════════════════════════════════════════════════════════
        # STAGE 2: SIMULATE — Feed images as camera frames
        # ═══════════════════════════════════════════════════════════════
        print("━" * 70)
        print(f"  Stage 2/5: SIMULATE — Feeding {len(test_images)} images to API")
        print("━" * 70)
        print()

        import urllib.request
        import mimetypes

        total_birds = 0
        total_black = 0
        color_totals = {}
        results_all = []
        errors = 0
        t_start = time.time()

        camera_ids = ["CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05"]
        sectors = ["A", "A", "B", "B", "C"]

        for i, img_path in enumerate(test_images):
            try:
                boundary = "----AeroWatchBoundary"
                mime_type = mimetypes.guess_type(str(img_path))[0] or "image/jpeg"
                img_data = img_path.read_bytes()

                body = (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="image"; filename="{img_path.name}"\r\n'
                    f"Content-Type: {mime_type}\r\n\r\n"
                ).encode() + img_data + f"\r\n--{boundary}--\r\n".encode()

                cam_idx = i % len(camera_ids)
                camera_id = camera_ids[cam_idx]
                sector = sectors[cam_idx]
                url = f"{api_url}/detect?camera_id={camera_id}&sector={sector}"

                req = urllib.request.Request(
                    url, data=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                    method="POST",
                )
                resp = urllib.request.urlopen(req, timeout=60)
                result = json.loads(resp.read())

                total_birds += result.get("total_birds", 0)
                total_black += result.get("black_bird_count", 0)
                for color, count in result.get("color_counts", {}).items():
                    color_totals[color] = color_totals.get(color, 0) + count

                results_all.append({
                    "image_id": str(img_path.name),
                    "camera_id": camera_id,
                    "sector": sector,
                    "total_birds": result.get("total_birds", 0),
                    "black_bird_count": result.get("black_bird_count", 0),
                    "color_counts": result.get("color_counts", {}),
                    "detections": result.get("detections", []),
                    "processing_time_ms": result.get("processing_time_ms", 0),
                })

                if (i + 1) % 10 == 0 or (i + 1) == len(test_images):
                    elapsed = time.time() - t_start
                    logger.info(
                        "  [%d/%d] %.1f img/s — birds=%d black=%d",
                        i + 1, len(test_images),
                        (i + 1) / elapsed,
                        total_birds, total_black,
                    )

            except Exception as e:
                errors += 1
                if errors <= 3:
                    logger.warning("  Image %d error: %s", i, e)

        elapsed = time.time() - t_start
        print()
        print(f"  ✅ Simulation complete:")
        print(f"    Images sent:   {len(test_images)}")
        print(f"    Total birds:   {total_birds}")
        print(f"    Black birds:   {total_black}")
        print(f"    Errors:        {errors}")
        print(f"    Throughput:    {len(test_images) / max(elapsed, 0.01):.1f} img/s")
        print(f"    Total time:    {elapsed:.1f}s")
        if color_totals:
            print()
            print(f"  🎨 Birds by colour:")
            for color, count in sorted(color_totals.items(), key=lambda x: -x[1]):
                bar = "█" * min(count, 40)
                print(f"     {color:>8s}: {count:>4d}  {bar}")
        print()

        # ═══════════════════════════════════════════════════════════════
        # STAGE 3: OBSERVE — Check drift, metrics, health
        # ═══════════════════════════════════════════════════════════════
        print("━" * 70)
        print("  Stage 3/5: OBSERVE — Checking system state")
        print("━" * 70)
        print()

        # Health
        try:
            resp = urllib.request.urlopen(f"{api_url}/health", timeout=5)
            health = json.loads(resp.read())
            print(f"  /health:")
            print(f"    Status:  {health['status']}")
            print(f"    Device:  {health['device']}")
            print(f"    Uptime:  {health['uptime_seconds']:.0f}s")
            print(f"    Stored:  {health.get('detections_stored', '?')} detections")
        except Exception as e:
            print(f"  /health: error — {e}")

        print()

        # Drift
        try:
            resp = urllib.request.urlopen(f"{api_url}/drift", timeout=5)
            drift = json.loads(resp.read())
            print(f"  /drift:")
            severity = drift.get("severity", drift.get("status", "unknown"))
            print(f"    Severity:  {severity}")

            alerts = drift.get("alerts", [])
            if alerts:
                for alert in alerts:
                    print(f"    ⚠️  {alert}")
            elif severity == "none":
                print(f"    ✅  No drift detected")
            elif "baseline_set" in str(drift.get("status", "")):
                print(f"    Baseline set from first {drift.get('num_images', '?')} images")
                print(f"    Green dominance: {drift.get('green_dominance', '?')}")
                print(f"    Mean confidence: {drift.get('mean_confidence', '?')}")
        except Exception as e:
            print(f"  /drift: error — {e}")

        print()

        # Stats
        try:
            resp = urllib.request.urlopen(f"{api_url}/stats", timeout=5)
            stats = json.loads(resp.read())
            print(f"  /stats:")
            for k, v in stats.items():
                print(f"    {k}: {v}")
        except Exception as e:
            print(f"  /stats: error — {e}")

        print()

        # ═══════════════════════════════════════════════════════════════
        # STAGE 4: DASHBOARD — Save results and open
        # ═══════════════════════════════════════════════════════════════
        print("━" * 70)
        print("  Stage 4/5: DASHBOARD — Saving results")
        print("━" * 70)
        print()

        results_json = {
            "images_processed": len(test_images),
            "total_birds": total_birds,
            "total_black_birds": total_black,
            "elapsed_seconds": round(elapsed, 1),
            "color_totals": color_totals,
            "per_image": results_all,
        }

        output_path = "pipeline_results.json"
        Path(output_path).write_text(json.dumps(results_json, indent=2))
        logger.info("Results saved to %s", output_path)

        # Try to open dashboard
        dashboard_path = Path(args.dashboard)
        if dashboard_path.exists():
            print(f"\n  Opening dashboard...")
            try:
                if sys.platform == "darwin":
                    subprocess.Popen(["open", str(dashboard_path)])
                elif sys.platform == "linux":
                    subprocess.Popen(["xdg-open", str(dashboard_path)])
                elif sys.platform == "win32":
                    os.startfile(str(dashboard_path))
                print(f"  → Dashboard opened. Drop {output_path} onto it.")
            except Exception:
                print(f"  → Open {dashboard_path} in your browser and drop {output_path} onto it.")
        else:
            print(f"  Dashboard not found at {dashboard_path}")
            print(f"  → Open dashboard.html in your browser and drop {output_path} onto it.")

        print()

        # ═══════════════════════════════════════════════════════════════
        # STAGE 5: RETRAIN CHECK — Would production trigger retraining?
        # ═══════════════════════════════════════════════════════════════
        print("━" * 70)
        print("  Stage 5/5: RETRAIN CHECK — Drift-based retrain decision")
        print("━" * 70)
        print()

        # Save drift report for check script
        os.makedirs("metrics", exist_ok=True)
        try:
            resp = urllib.request.urlopen(f"{api_url}/drift", timeout=5)
            drift_report = json.loads(resp.read())
            Path("metrics/drift_latest.json").write_text(json.dumps(drift_report, indent=2))
            logger.info("Drift report saved to metrics/drift_latest.json")
        except Exception as e:
            logger.warning("Could not fetch drift report: %s", e)
            drift_report = None

        # Save current stats as baseline if none exists
        baseline_path = Path("metrics/baseline.json")
        if not baseline_path.exists():
            baseline = {
                "overall_accuracy": 0.85,  # placeholder — use evaluate_cub200.py for real
                "black_bird_f1": 0.75,
            }
            baseline_path.write_text(json.dumps(baseline, indent=2))
            logger.info("Placeholder baseline saved (run evaluate_cub200.py for real metrics)")

        # Run retrain check
        check_script = Path("scripts/check_retrain_needed.py")
        if check_script.exists() and Path("metrics/drift_latest.json").exists():
            cmd = [
                sys.executable, str(check_script),
                "--metrics-file", "metrics/baseline.json",
                "--drift-file", "metrics/drift_latest.json",
                "--baseline-file", "metrics/baseline.json",
            ]
            result = subprocess.run(cmd)
            print()
            if result.returncode == 1:
                print("  🔄 Retraining WOULD be triggered in production")
                print("     (drift severity or metric degradation exceeded thresholds)")
            else:
                print("  ✅ No retraining needed — model is performing well")
        else:
            if not check_script.exists():
                print(f"  ⚠ Retrain check script not found at {check_script}")
            if drift_report and drift_report.get("severity", "none") != "none":
                print(f"  🔄 Drift severity: {drift_report['severity']} — would trigger retrain")
            else:
                print("  ✅ No drift detected — no retrain needed")

        print()

    finally:
        # Clean shutdown
        print("━" * 70)
        print("  Shutting down API server...")
        print("━" * 70)
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        server_log.close()
        print("  ✅ Done.")
        print()


if __name__ == "__main__":
    main()
