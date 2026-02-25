#!/usr/bin/env python3
"""
Aero-Watch — CUB-200-2011 Evaluation Suite
============================================

Usage:
  python evaluate_cub200.py --dataset-dir ./CUB_200_2011 --max-images 50    # smoke test
  python evaluate_cub200.py --dataset-dir ./CUB_200_2011 --max-images 500             # medium run
  python evaluate_cub200.py --dataset-dir ./CUB_200_2011                               # full (5794 imgs)
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cub200-eval")


# ═══════════════════════════════════════════════════════════════════════════
# GROUND TRUTH COLOUR MAPPING
# ═══════════════════════════════════════════════════════════════════════════

# CUB attribute colours → our 9-class scheme
CUB_COLOR_MAP = {
    "black": "black",
    "blue": "blue",
    "brown": "brown",
    "buff": "brown",
    "grey": "grey",
    "green": "green",
    "olive": "green",
    "orange": "brown",     # orange birds look brown in most conditions
    "pink": "red",
    "purple": "mixed",
    "rufous": "brown",
    "red": "red",
    "white": "white",
    "yellow": "yellow",
    "iridescent": "mixed",
}

# Authoritative species → colour mapping for CUB-200 classes.
# This overrides noisy CUB attributes for species where we KNOW the colour.
# CUB class format: "001.Black_footed_Albatross"
SPECIES_COLOR_OVERRIDE = {
    # BLACK birds — critical for assessment, must be correct
    "004.Groove_billed_Ani": "black",
    "009.Brewer_Blackbird": "black",
    "011.Rusty_Blackbird": "black",       # appears black in most photos
    "026.Bronzed_Cowbird": "black",
    "027.Shiny_Cowbird": "black",
    "029.American_Crow": "black",
    "030.Fish_Crow": "black",
    "079.Belted_Kingfisher": "grey",      # not black despite dark parts
    "100.Brown_Pelican": "brown",
    "133.White_throated_Sparrow": "brown",
    "134.Grasshopper_Sparrow": "brown",

    # WHITE birds
    "001.Black_footed_Albatross": "brown",  # actually brownish, not white
    "002.Laysan_Albatross": "white",
    "023.Brandt_Cormorant": "black",
    "024.Red_faced_Cormorant": "black",
    "025.Pelagic_Cormorant": "black",

    # RED birds
    "017.Cardinal": "red",
    "036.Northern_Flicker": "brown",
    "010.Red_winged_Blackbird": "black",   # primarily black with red patch

    # BLUE birds
    "014.Indigo_Bunting": "blue",
    "047.American_Goldfinch": "yellow",

    # YELLOW birds
    "012.Yellow_headed_Blackbird": "black",  # primarily black with yellow head

    # GREY birds
    "019.Gray_Catbird": "grey",
    "034.Gray_crowned_Rosy_Finch": "grey",

    # GT CORRECTIONS — CUB attributes mark these as black incorrectly
    "173.Orange_crowned_Warbler": "green",    # olive-green, NOT black
    "160.Black_throated_Blue_Warbler": "blue", # blue+black but primarily blue
    "188.Pileated_Woodpecker": "mixed",       # black+red+white = mixed
    "033.Yellow_billed_Cuckoo": "brown",      # brown/grey, NOT black
    "038.Great_Crested_Flycatcher": "brown",  # olive-brown, NOT black
    "107.Common_Raven": "black",              # definitely black
    "065.Slaty_backed_Gull": "white",         # white+grey, NOT black
}


# Keyword-based fallback for species not in the override list
def _color_from_species_name(species_name: str) -> str:
    """Derive colour from CUB species name (e.g. '029.American_Crow')."""
    name = species_name.lower()

    # Ordered by specificity — more specific matches first
    rules = [
        # Black species
        (["crow", "raven", "grackle", "ani", "cowbird", "starling", "cormorant"], "black"),
        (["blackbird"], "black"),
        # White species
        (["egret", "swan", "tern", "gull"], "white"),
        # Red species
        (["cardinal", "tanager"], "red"),
        # Blue species
        (["bluebird", "indigo"], "blue"),
        # Yellow species
        (["goldfinch", "yellowthroat"], "yellow"),
        # Green species
        (["vireo", "hummingbird"], "green"),
        # Grey species
        (["mockingbird", "gray_catbird", "grey"], "grey"),
        # Brown species (common default)
        (["sparrow", "wren", "creeper", "thrasher", "finch", "flycatcher", "warbler"], "brown"),
    ]

    for keywords, color in rules:
        for kw in keywords:
            if kw in name:
                return color

    return "mixed"


# ═══════════════════════════════════════════════════════════════════════════
# CUB-200-2011 DATASET LOADER
# ═══════════════════════════════════════════════════════════════════════════

class CUB200Dataset:
    """Loader for CUB-200-2011 dataset."""

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self._validate()
        self.images = self._load_images()
        self.bboxes = self._load_bboxes()
        self.split = self._load_split()
        self.classes = self._load_classes()
        self.image_labels = self._load_image_labels()
        self.attributes, self.primary_color_attrs = self._load_attributes()
        self.image_colors = self._load_image_colors()

    def _validate(self):
        required = ["images.txt", "bounding_boxes.txt", "train_test_split.txt"]
        for f in required:
            if not (self.root / f).exists():
                raise FileNotFoundError(f"Missing {f} in {self.root}")

    def _load_images(self) -> dict[int, str]:
        images = {}
        for line in (self.root / "images.txt").read_text().strip().split("\n"):
            parts = line.strip().split()
            images[int(parts[0])] = parts[1]
        logger.info("Loaded %d images", len(images))
        return images

    def _load_bboxes(self) -> dict[int, list[float]]:
        bboxes = {}
        for line in (self.root / "bounding_boxes.txt").read_text().strip().split("\n"):
            parts = line.strip().split()
            img_id = int(parts[0])
            bboxes[img_id] = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
        return bboxes

    def _load_split(self) -> dict[int, bool]:
        split = {}
        for line in (self.root / "train_test_split.txt").read_text().strip().split("\n"):
            parts = line.strip().split()
            split[int(parts[0])] = int(parts[1]) == 1
        return split

    def _load_classes(self) -> dict[int, str]:
        classes = {}
        for line in (self.root / "classes.txt").read_text().strip().split("\n"):
            parts = line.strip().split(maxsplit=1)
            classes[int(parts[0])] = parts[1]
        return classes

    def _load_image_labels(self) -> dict[int, int]:
        labels = {}
        for line in (self.root / "image_class_labels.txt").read_text().strip().split("\n"):
            parts = line.strip().split()
            labels[int(parts[0])] = int(parts[1])
        return labels

    def _load_attributes(self) -> tuple[dict[int, str], dict[int, str]]:
        attr_file = self.root / "attributes" / "attributes.txt"
        if not attr_file.exists():
            logger.warning("attributes.txt not found")
            return {}, {}

        all_attrs = {}
        primary_color_attrs = {}
        for line in attr_file.read_text().strip().split("\n"):
            parts = line.strip().split(maxsplit=1)
            attr_id = int(parts[0])
            attr_name = parts[1]
            all_attrs[attr_id] = attr_name

            if "has_primary_color" in attr_name.lower():
                color_part = attr_name.split("::")[-1].strip().lower()
                primary_color_attrs[attr_id] = color_part

        logger.info("Found %d primary_color attributes", len(primary_color_attrs))
        return all_attrs, primary_color_attrs

    def _load_image_colors(self) -> dict[int, str]:
        """
        Determine GT colour for each image using this priority:
          1. Species override (authoritative for known species)
          2. CUB attributes with certainty ≥ 3 (probably/definitely)
          3. Species name keyword fallback
        """
        image_colors = {}

        # Load attribute votes
        attr_votes: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        attr_label_file = self.root / "attributes" / "image_attribute_labels.txt"

        if self.primary_color_attrs and attr_label_file.exists():
            for line in attr_label_file.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                img_id, attr_id = int(parts[0]), int(parts[1])
                is_present, certainty = int(parts[2]), int(parts[3])

                if attr_id in self.primary_color_attrs and is_present == 1 and certainty >= 3:
                    weight = 1.0 if certainty == 3 else 2.0
                    cub_color = self.primary_color_attrs[attr_id]
                    our_color = CUB_COLOR_MAP.get(cub_color, "mixed")
                    attr_votes[img_id][our_color] += weight

        # Assign colours
        src_counts = {"override": 0, "attribute": 0, "keyword": 0}

        for img_id in self.images:
            class_id = self.image_labels.get(img_id, 0)
            species_name = self.classes.get(class_id, "")

            # Priority 1: Species override
            if species_name in SPECIES_COLOR_OVERRIDE:
                image_colors[img_id] = SPECIES_COLOR_OVERRIDE[species_name]
                src_counts["override"] += 1
                continue

            # Priority 2: CUB attributes (high certainty)
            if img_id in attr_votes and attr_votes[img_id]:
                best = max(attr_votes[img_id], key=attr_votes[img_id].get)
                image_colors[img_id] = best
                src_counts["attribute"] += 1
                continue

            # Priority 3: Species name keywords
            image_colors[img_id] = _color_from_species_name(species_name)
            src_counts["keyword"] += 1

        logger.info("GT colour sources: %s", src_counts)

        # Distribution
        dist = defaultdict(int)
        for c in image_colors.values():
            dist[c] += 1
        logger.info("GT colour distribution: %s", dict(sorted(dist.items())))

        return image_colors

    def get_test_images(self) -> list[int]:
        return [i for i, is_train in self.split.items() if not is_train]

    def get_image_path(self, img_id: int) -> Path:
        return self.root / "images" / self.images[img_id]

    def get_gt_bbox_xyxy(self, img_id: int) -> list[float]:
        x, y, w, h = self.bboxes[img_id]
        return [x, y, x + w, y + h]

    def get_gt_color(self, img_id: int) -> str:
        return self.image_colors.get(img_id, "mixed")


# ═══════════════════════════════════════════════════════════════════════════
# DETECTION METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


def compute_ap(precisions: list[float], recalls: list[float]) -> float:
    if not recalls:
        return 0.0
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        prec_at_rec = [p for p, r in zip(precisions, recalls) if r >= t]
        if prec_at_rec:
            ap += max(prec_at_rec) / 11.0
    return ap


class DetectionEvaluator:
    def __init__(self):
        self.all_detections = []
        self.total_gt = 0
        self.total_small_gt = 0
        self.missed_gt = 0
        self.counting_errors = []

    def add_image(self, pred_boxes, pred_confs, gt_box):
        self.total_gt += 1
        gt_w, gt_h = gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]
        is_small = (gt_w * gt_h) < 1024
        if is_small:
            self.total_small_gt += 1

        self.counting_errors.append(abs(len(pred_boxes) - 1))

        if not pred_boxes:
            self.missed_gt += 1
            self.all_detections.append((0.0, 0.0, is_small))
            return

        best_iou, best_conf = 0.0, 0.0
        for box, conf in zip(pred_boxes, pred_confs):
            iou = compute_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_conf = conf

        self.all_detections.append((best_conf, best_iou, is_small))

        # Extra detections are false positives
        if len(pred_boxes) > 1:
            ious = [compute_iou(b, gt_box) for b in pred_boxes]
            best_idx = int(np.argmax(ious))
            for i, (box, conf) in enumerate(zip(pred_boxes, pred_confs)):
                if i != best_idx:
                    self.all_detections.append((conf, 0.0, False))

    def compute_metrics(self) -> dict:
        results = {}
        results["mAP@50"] = round(self._ap_at_iou(0.50), 4)
        aps = [self._ap_at_iou(t) for t in np.arange(0.50, 1.00, 0.05)]
        results["mAP@50:95"] = round(float(np.mean(aps)), 4)

        pr = self._pr_at_iou(0.50)
        results["precision@50"] = round(pr["precision"], 4)
        results["recall@50"] = round(pr["recall"], 4)
        f1 = 2 * pr["precision"] * pr["recall"] / max(pr["precision"] + pr["recall"], 1e-8)
        results["f1@50"] = round(f1, 4)

        if self.total_small_gt > 0:
            small_hits = sum(1 for c, iou, s in self.all_detections if s and iou >= 0.5 and c > 0)
            results["small_object_recall@50"] = round(small_hits / self.total_small_gt, 4)
        else:
            results["small_object_recall@50"] = "N/A"
        results["small_object_count"] = self.total_small_gt

        results["counting_MAE"] = round(float(np.mean(self.counting_errors)), 4)
        results["total_gt"] = self.total_gt
        results["missed"] = self.missed_gt
        results["detection_rate"] = round(1 - self.missed_gt / max(self.total_gt, 1), 4)
        return results

    def _ap_at_iou(self, thresh):
        sorted_d = sorted(self.all_detections, key=lambda x: x[0], reverse=True)
        tp, fp = [], []
        for conf, iou, _ in sorted_d:
            if conf == 0.0 and iou == 0.0:
                continue
            tp.append(1 if iou >= thresh else 0)
            fp.append(0 if iou >= thresh else 1)
        if not tp:
            return 0.0
        tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
        prec = tp_c / (tp_c + fp_c)
        rec = tp_c / max(self.total_gt, 1)
        return compute_ap(prec.tolist(), rec.tolist())

    def _pr_at_iou(self, thresh):
        tp = sum(1 for c, iou, _ in self.all_detections if iou >= thresh and c > 0)
        fp = sum(1 for c, iou, _ in self.all_detections if iou < thresh and c > 0)
        fn = self.total_gt - tp
        return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION METRICS
# ═══════════════════════════════════════════════════════════════════════════

COLOR_NAMES = ["black", "white", "brown", "green", "red", "blue", "yellow", "grey", "mixed"]


class ClassificationEvaluator:
    def __init__(self):
        self.predictions = []

    def add(self, gt_color: str, pred_color: str):
        self.predictions.append((gt_color, pred_color))

    def compute_metrics(self) -> dict:
        if not self.predictions:
            return {"error": "No predictions"}

        results = {}
        correct = sum(1 for g, p in self.predictions if g == p)
        results["overall_accuracy"] = round(correct / len(self.predictions), 4)
        results["total_classified"] = len(self.predictions)

        per_class = {}
        for color in COLOR_NAMES:
            tp = sum(1 for g, p in self.predictions if g == color and p == color)
            fp = sum(1 for g, p in self.predictions if g != color and p == color)
            fn = sum(1 for g, p in self.predictions if g == color and p != color)
            support = tp + fn
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            per_class[color] = {
                "precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(f1, 4), "support": support,
            }

        results["per_class"] = per_class
        f1s = [v["f1"] for v in per_class.values() if v["support"] > 0]
        results["macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else 0.0

        b = per_class.get("black", {})
        results["black_bird_precision"] = b.get("precision", 0)
        results["black_bird_recall"] = b.get("recall", 0)
        results["black_bird_f1"] = b.get("f1", 0)
        results["black_bird_support"] = b.get("support", 0)

        cm = np.zeros((len(COLOR_NAMES), len(COLOR_NAMES)), dtype=int)
        for g, p in self.predictions:
            if g in COLOR_NAMES and p in COLOR_NAMES:
                cm[COLOR_NAMES.index(g)][COLOR_NAMES.index(p)] += 1
        results["confusion_matrix"] = cm.tolist()
        results["confusion_labels"] = COLOR_NAMES
        return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(args):
    logger.info("Loading CUB-200-2011 from %s", args.dataset_dir)
    dataset = CUB200Dataset(args.dataset_dir)

    test_ids = dataset.get_test_images()
    logger.info("Test set: %d images", len(test_ids))
    if args.max_images and args.max_images < len(test_ids):
        # Stratified sampling: pick evenly across species so all colours
        # are represented even with small --max-images values.
        # (First-N would only include albatrosses/auklets due to ID ordering.)
        import random
        rng = random.Random(args.seed)  # deterministic, change with --seed

        # Group test images by species
        species_groups: dict[int, list[int]] = defaultdict(list)
        for img_id in test_ids:
            cls_id = dataset.image_labels.get(img_id, 0)
            species_groups[cls_id].append(img_id)

        # Round-robin: take 1 from each species, repeat until we have enough
        sampled = []
        species_ids = sorted(species_groups.keys())
        rng.shuffle(species_ids)
        per_species = {s: list(imgs) for s, imgs in species_groups.items()}
        for s in per_species:
            rng.shuffle(per_species[s])

        idx = 0
        while len(sampled) < args.max_images:
            added_any = False
            for s in species_ids:
                if idx < len(per_species[s]) and len(sampled) < args.max_images:
                    sampled.append(per_species[s][idx])
                    added_any = True
            idx += 1
            if not added_any:
                break

        test_ids = sampled
        logger.info("Stratified sample: %d images across %d species",
                     len(test_ids), len(species_groups))

    # Load pipeline
    import torch
    if torch.cuda.is_available():
        device = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    clip_device = "cpu" if device == "mps" else device
    logger.info("Device: %s (CLIP: %s)", device, clip_device)

    from src.models.bird_detector import BirdDetector, ShadowDiscriminator
    from src.models.color_classifier import BirdColorClassifier
    from src.inference.inference_engine import InferenceEngine, CLIPBirdVerifier

    detector = BirdDetector(
        model_path=args.model, input_size=args.input_size,
        confidence_threshold=args.conf, device=device,
    )
    color_model = getattr(args, 'color_model', None)
    classifier = BirdColorClassifier(model_path=color_model, confidence_threshold=0.4, device=clip_device)

    clip_verifier = None
    if not args.no_clip:
        clip_verifier = CLIPBirdVerifier(device=clip_device)

    engine = InferenceEngine(detector, classifier, ShadowDiscriminator(), clip_verifier=clip_verifier)
    engine.load_models()

    det_eval = DetectionEvaluator()
    cls_eval = ClassificationEvaluator()

    # Black bird tracking
    black_gt_count = 0
    black_detected = 0
    black_clip_passed = 0
    black_classified_as = defaultdict(int)

    logger.info("Classifier mode: %s", classifier._mode)
    logger.info("=" * 70)
    logger.info("Evaluating %d images...", len(test_ids))
    t_start = time.time()
    errors = 0

    # Log GT distribution
    gt_dist = defaultdict(int)
    for iid in test_ids:
        gt_dist[dataset.get_gt_color(iid)] += 1
    logger.info("GT colour distribution: %s", dict(sorted(gt_dist.items())))

    for idx, img_id in enumerate(test_ids):
        try:
            img_path = dataset.get_image_path(img_id)
            gt_bbox = dataset.get_gt_bbox_xyxy(img_id)
            gt_color = dataset.get_gt_color(img_id)

            pil_img = Image.open(img_path).convert("RGB")
            img_array = np.array(pil_img)

            raw_dets = detector.detect(img_array, bird_only=True)
            raw_dets = InferenceEngine._nms(raw_dets, iou_threshold=0.40)

            pred_boxes = [d.bbox for d in raw_dets]
            pred_confs = [d.confidence for d in raw_dets]
            det_eval.add_image(pred_boxes, pred_confs, gt_bbox)

            # Colour classification on best-matching detection
            is_gt_black = (gt_color == "black")
            if is_gt_black:
                black_gt_count += 1

            if raw_dets:
                if is_gt_black:
                    black_detected += 1

                ious = [compute_iou(d.bbox, gt_bbox) for d in raw_dets]
                best = raw_dets[int(np.argmax(ious))]

                x1, y1 = max(0, int(best.bbox[0])), max(0, int(best.bbox[1]))
                x2, y2 = min(img_array.shape[1], int(best.bbox[2])), min(img_array.shape[0], int(best.bbox[3]))

                if (x2 - x1) >= 16 and (y2 - y1) >= 16:
                    crop = img_array[y1:y2, x1:x2].copy()

                    skip = False
                    if clip_verifier:
                        is_bird, _ = clip_verifier.is_bird(crop)
                        if not is_bird:
                            skip = True

                    if not skip:
                        if is_gt_black:
                            black_clip_passed += 1
                        pred = classifier.classify(crop)
                        cls_eval.add(gt_color, pred.primary_color)
                        if is_gt_black:
                            black_classified_as[pred.primary_color] += 1

                        # Log features for every black-related prediction
                        if pred.primary_color == "black" or is_gt_black:
                            from src.models.color_classifier import BlackBirdDetector
                            bbd = BlackBirdDetector()
                            _, _, feat = bbd.detect(crop)
                            label = "TP" if (is_gt_black and pred.primary_color == "black") \
                                else "FN" if is_gt_black \
                                else "FP"
                            species = dataset.classes.get(
                                dataset.image_labels.get(img_id, 0), "?")
                            logger.info(
                                "  [%s] gt=%s pred=%s %s "
                                "dpf=%.2f v25=%.0f v50=%.0f db=%.0f s=%.0f rgbr=%.2f sc=%.2f",
                                label, gt_color, pred.primary_color, species,
                                feat.get("dark_pixel_frac", 0),
                                feat.get("v_p25", 0), feat.get("v_p50", 0),
                                feat.get("dark_brightness", 0),
                                feat.get("s_of_dark", 0),
                                feat.get("dark_rgb_ratio", 0),
                                feat.get("black_score", 0),
                            )

            if (idx + 1) % 50 == 0 or idx == 0:
                elapsed = time.time() - t_start
                speed = (idx + 1) / elapsed
                eta = (len(test_ids) - idx - 1) / max(speed, 0.01)
                n_cls = len(cls_eval.predictions)
                n_correct = sum(1 for g, p in cls_eval.predictions if g == p)
                logger.info(
                    "  [%d/%d] %.1f img/s ETA %.0fs | det=%.1f%% cls=%d/%d (%.0f%%)",
                    idx + 1, len(test_ids), speed, eta,
                    det_eval.compute_metrics()["detection_rate"] * 100,
                    n_correct, n_cls,
                    100 * n_correct / max(n_cls, 1),
                )
        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning("Error on image %d: %s", img_id, e)

    total_time = time.time() - t_start
    logger.info("Done: %.1fs (%.1f img/s), %d errors", total_time, len(test_ids) / max(total_time, 0.1), errors)

    # Black bird tracking summary
    logger.info("─── BLACK BIRD TRACKING ───")
    logger.info("  GT black in test set:   %d", black_gt_count)
    logger.info("  Detected by YOLO:       %d", black_detected)
    logger.info("  Passed CLIP verifier:   %d", black_clip_passed)
    logger.info("  Classified as:          %s", dict(black_classified_as))
    # Count FPs: non-black birds classified as black
    black_fps = sum(1 for g, p in cls_eval.predictions if g != "black" and p == "black")
    black_fp_sources = defaultdict(int)
    for g, p in cls_eval.predictions:
        if g != "black" and p == "black":
            black_fp_sources[g] += 1
    logger.info("  Non-black pred as black (FP): %d  from: %s", black_fps, dict(black_fp_sources))

    det_m = det_eval.compute_metrics()
    cls_m = cls_eval.compute_metrics()

    # ── Print report ─────────────────────────────────────────────────────
    print()
    print("═" * 70)
    print("  📊  AERO-WATCH EVALUATION REPORT — CUB-200-2011")
    print("═" * 70)
    print(f"\n  Config: model={args.model} "
          f"clip={'on' if not args.no_clip else 'off'} conf={args.conf} "
          f"images={len(test_ids)} device={device} time={total_time:.0f}s")

    print(f"\n{'─'*70}\n  🎯 DETECTION\n{'─'*70}")
    for k in ["mAP@50", "mAP@50:95", "precision@50", "recall@50", "f1@50",
              "detection_rate", "counting_MAE", "small_object_recall@50"]:
        print(f"    {k:<25} {det_m[k]}")

    print(f"\n{'─'*70}\n  🎨 CLASSIFICATION\n{'─'*70}")
    print(f"    Overall accuracy:     {cls_m.get('overall_accuracy', 0):.4f}")
    print(f"    Macro F1:             {cls_m.get('macro_f1', 0):.4f}")

    print(f"\n  ⬛ BLACK BIRD:")
    print(f"    Precision:  {cls_m.get('black_bird_precision', 0):.4f}")
    print(f"    Recall:     {cls_m.get('black_bird_recall', 0):.4f}")
    print(f"    F1:         {cls_m.get('black_bird_f1', 0):.4f}")
    print(f"    Support:    {cls_m.get('black_bird_support', 0)}")

    print(f"\n  Per-class:")
    print(f"    {'Colour':<10} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
    print(f"    {'─'*10} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for c in COLOR_NAMES:
        if c in cls_m.get("per_class", {}):
            v = cls_m["per_class"][c]
            print(f"    {c:<10} {v['precision']:>6.3f} {v['recall']:>6.3f} "
                  f"{v['f1']:>6.3f} {v['support']:>6d}")

    cm = cls_m.get("confusion_matrix", [])
    if cm:
        print(f"\n  Confusion Matrix (rows=GT, cols=Pred):")
        print(f"    {'':>10}" + "".join(f"{c[:5]:>6}" for c in COLOR_NAMES))
        for i, row in enumerate(cm):
            print(f"    {COLOR_NAMES[i]:<10}" + "".join(f"{v:>6}" for v in row))

    print(f"\n{'─'*70}\n  📋 TARGETS\n{'─'*70}")
    targets = {
        "mAP@50 ≥ 0.65":           det_m["mAP@50"] >= 0.65,
        "mAP@50:95 ≥ 0.45":        det_m["mAP@50:95"] >= 0.45,
        "Precision ≥ 0.75":        det_m["precision@50"] >= 0.75,
        "Recall ≥ 0.70":          det_m["recall@50"] >= 0.70,
        "Counting MAE ≤ 1.5":      det_m["counting_MAE"] <= 1.5,
        "Black prec ≥ 0.85":       cls_m.get("black_bird_precision", 0) >= 0.85,
        "Black rec ≥ 0.75":        cls_m.get("black_bird_recall", 0) >= 0.75,
        "Black F1 ≥ 0.80":         cls_m.get("black_bird_f1", 0) >= 0.80,
        "Accuracy ≥ 0.70":         cls_m.get("overall_accuracy", 0) >= 0.70,
        "Macro F1 ≥ 0.60":         cls_m.get("macro_f1", 0) >= 0.60,
    }
    for t, ok in targets.items():
        print(f"    {'✅' if ok else '❌'} {t}")
    print(f"\n    Score: {sum(targets.values())}/{len(targets)}")
    print("═" * 70)

    # Save JSON
    out_path = args.output or "eval_results.json"
    results_json = {"config": vars(args), "detection": det_m, "classification": cls_m,
                    "targets": {k: v for k, v in targets.items()}}
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    logger.info("Results → %s", out_path)

    # CI/CD output: flat metrics file for check_retrain_needed.py
    if args.output_json:
        flat_metrics = {
            "overall_accuracy": cls_m.get("overall_accuracy", 0),
            "macro_f1": cls_m.get("macro_f1", 0),
            "black_bird_precision": cls_m.get("black_bird_precision", 0),
            "black_bird_recall": cls_m.get("black_bird_recall", 0),
            "black_bird_f1": cls_m.get("black_bird_f1", 0),
            "mAP_50": det_m.get("mAP@50", 0),
        }
        with open(args.output_json, "w") as f:
            json.dump(flat_metrics, f, indent=2)
        logger.info("CI metrics → %s", args.output_json)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Aero-Watch on CUB-200-2011")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--input-size", type=int, default=1280)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--color-model", default=None,
                        help="Path to trained color classifier weights (.pt)")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified sampling")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-json", default=None,
                        help="Write evaluation metrics to JSON file (for CI/CD)")
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
