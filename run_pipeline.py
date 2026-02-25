#!/usr/bin/env python3
"""
Aero-Watch — Run Pipeline
===========================

Single image:
    python run_pipeline.py -i photo.jpg
    python run_pipeline.py -i photo.jpg --color-model color_classifier_best.pt

Batch (folder):
    python run_pipeline.py -i ./field_photos/
    python run_pipeline.py -i img1.jpg img2.jpg img3.jpg
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aero-watch")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

COLOR_PALETTE = {
    "black": (40, 40, 40), "white": (255, 255, 255), "brown": (139, 90, 43),
    "green": (34, 139, 34), "red": (220, 20, 60), "blue": (30, 100, 220),
    "yellow": (255, 215, 0), "grey": (150, 150, 150), "mixed": (200, 100, 200),
    "unknown": (128, 128, 128),
}


def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        logger.info("Device: CUDA — %s", torch.cuda.get_device_name(0))
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Device: Apple MPS (Metal)")
        return "mps"
    else:
        logger.info("Device: CPU")
        return "cpu"


def draw_results(image: np.ndarray, result: dict, output_path: str) -> None:
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=16)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

    for det in result["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        color_name = det["color"] or "unknown"
        box_color = COLOR_PALETTE.get(color_name, (0, 255, 0))

        for offset in range(3):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=box_color)

        class_label = det.get("class_name", "bird")
        label = f"{class_label} | {color_name} ({det['confidence']:.0%})"
        text_bbox = draw.textbbox((x1, y1 - 22), label, font=font)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=box_color)
        brightness = sum(box_color) / 3
        draw.text((x1, y1 - 22), label, fill=(255, 255, 255) if brightness < 128 else (0, 0, 0), font=font)

    img.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Aero-Watch Bird Detection Pipeline")
    parser.add_argument("--image", "--input", "-i", type=str, required=True,
                        nargs="+", help="Input image path(s) or folder")
    parser.add_argument("--model", "-m", type=str, default="yolo26x.pt",
                        help="YOLO variant (default: yolo26x.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--nms-iou", type=float, default=0.40, help="NMS IoU threshold")
    parser.add_argument("--no-clip", action="store_true", help="Disable CLIP crop verification")
    parser.add_argument("--all-classes", action="store_true", help="Detect all COCO classes")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image path (single image only)")
    parser.add_argument("--input-size", type=int, default=1280, help="Model input resolution")
    parser.add_argument("--color-model", type=str, default=None,
                        help="Path to trained colour classifier weights (.pt)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results as JSON (for dashboard)")
    args = parser.parse_args()

    # ── Resolve input images ─────────────────────────────────────────
    image_paths = []
    for entry in args.image:
        p = Path(entry)
        if p.is_dir():
            image_paths.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS))
        elif p.is_file():
            image_paths.append(p)
        else:
            logger.warning("Skipping %s (not found)", entry)

    if not image_paths:
        logger.error("No images found")
        sys.exit(1)

    print("=" * 70)
    print("  🦅  AERO-WATCH — Equatorial Bird Detection Pipeline")
    print("=" * 70)
    print(f"  Images: {len(image_paths)}")
    print()

    # ── Load models ──────────────────────────────────────────────────
    device = get_device()
    clip_device = "cpu" if device == "mps" else device

    from src.models.bird_detector import BirdDetector, ShadowDiscriminator, FrameResult
    from src.models.color_classifier import BirdColorClassifier
    from src.inference.inference_engine import InferenceEngine, CLIPBirdVerifier
    from src.data.data_pipeline import ImagePreprocessor, CameraMetadata
    from datetime import datetime, timezone

    logger.info("Loading models...")
    detector = BirdDetector(
        model_path=args.model, input_size=args.input_size,
        confidence_threshold=args.conf, device=device,
    )
    classifier = BirdColorClassifier(model_path=args.color_model, confidence_threshold=0.4, device=device)
    shadow_disc = ShadowDiscriminator()

    clip_verifier = None
    if not args.no_clip:
        clip_verifier = CLIPBirdVerifier(device=clip_device)

    engine = InferenceEngine(detector, classifier, shadow_disc, clip_verifier=clip_verifier)
    engine.load_models()

    preprocessor = ImagePreprocessor(target_size=args.input_size)

    # ── Process images ───────────────────────────────────────────────
    t_start = time.time()
    total_birds = 0
    total_black = 0
    results_all = []

    for img_idx, image_path in enumerate(image_paths):
        logger.info("[%d/%d] %s", img_idx + 1, len(image_paths), image_path.name)

        output_path = args.output if (args.output and len(image_paths) == 1) else str(
            image_path.parent / f"{image_path.stem}_detected{image_path.suffix}"
        )

        try:
            with open(image_path, "rb") as f:
                raw_bytes = f.read()

            metadata = CameraMetadata(
                camera_id="LOCAL", sector="test", latitude=0.0, longitude=0.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
                battery_level=100.0, temperature_celsius=25.0,
                humidity_percent=80.0, firmware_version="local",
            )
            processed = preprocessor.process(raw_bytes, metadata)

            # Detect
            raw_dets = detector.detect(processed.processed_array, bird_only=not args.all_classes)
            raw_dets = InferenceEngine._nms(raw_dets, iou_threshold=args.nms_iou)

            # Verify + classify
            valid = []
            for det in raw_dets:
                h_img, w_img = processed.processed_array.shape[:2]
                x1, y1 = max(0, int(det.bbox[0])), max(0, int(det.bbox[1]))
                x2, y2 = min(w_img, int(det.bbox[2])), min(h_img, int(det.bbox[3]))
                if (x2 - x1) < 16 or (y2 - y1) < 16:
                    continue
                crop = processed.processed_array[y1:y2, x1:x2].copy()

                if clip_verifier:
                    is_bird, _ = clip_verifier.is_bird(crop)
                    if not is_bird:
                        continue

                if float(np.mean(crop)) < 60:
                    is_shadow, _ = shadow_disc.is_shadow(crop)
                    if is_shadow:
                        continue

                color_pred = classifier.classify(crop)
                det.color_label = color_pred.primary_color
                det.color_confidence = color_pred.confidence
                valid.append(det)

            frame_result = FrameResult(
                image_id=processed.image_id, camera_id="LOCAL",
                timestamp=metadata.timestamp, detections=valid,
            )
            result_dict = frame_result.to_dict()
            total_birds += result_dict["total_birds"]
            total_black += result_dict["black_bird_count"]
            results_all.append(result_dict)

            # Per-image log
            colors = result_dict["color_counts"]
            color_str = ", ".join(f"{c}={n}" for c, n in colors.items()) if colors else "none"
            logger.info("  → %d birds (%s)", result_dict["total_birds"], color_str)

            draw_results(processed.processed_array, result_dict, output_path)

        except Exception as e:
            logger.error("  Error: %s", e)

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print()
    print("━" * 70)
    print("  📊  RESULTS")
    print("━" * 70)
    print(f"  Images processed:  {len(image_paths)}")
    print(f"  Total birds:       {total_birds}")
    print(f"  Black birds:       {total_black}")
    print(f"  Time:              {elapsed:.1f}s ({len(image_paths) / max(elapsed, 0.01):.1f} img/s)")

    if len(image_paths) == 1 and results_all:
        r = results_all[0]
        if r["color_counts"]:
            print()
            print("  🎨 Birds by colour:")
            for color, count in sorted(r["color_counts"].items()):
                print(f"     {color:>8s}: {count:>2d}  {'█' * count}")
    elif len(image_paths) > 1 and results_all:
        color_totals = {}
        for r in results_all:
            for color, count in r.get("color_counts", {}).items():
                color_totals[color] = color_totals.get(color, 0) + count
        if color_totals:
            print()
            print("  🎨 Birds by colour (all images):")
            for color, count in sorted(color_totals.items(), key=lambda x: -x[1]):
                bar = "█" * min(count, 40)
                print(f"     {color:>8s}: {count:>3d}  {bar}")

    print("━" * 70)

    # ── Save JSON results ─────────────────────────────────────────────
    if args.output_json:
        # Aggregate colour totals
        color_totals = {}
        for r in results_all:
            for color, count in r.get("color_counts", {}).items():
                color_totals[color] = color_totals.get(color, 0) + count

        summary = {
            "images_processed": len(image_paths),
            "total_birds": total_birds,
            "total_black_birds": total_black,
            "elapsed_seconds": round(elapsed, 1),
            "color_totals": color_totals,
            "per_image": results_all,
        }
        Path(args.output_json).write_text(json.dumps(summary, indent=2))
        logger.info("Results saved to %s", args.output_json)


if __name__ == "__main__":
    main()
