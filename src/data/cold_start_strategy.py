"""
Aero-Watch Cold-Start Data Strategy
====================================
Solves the "no labelled data" problem through a multi-phase approach:

Phase 0 — Foundation Model Bootstrap (immediate, no training required)
  Use a pre-trained open-vocabulary detector (Grounding DINO / OWL-ViT)
  with text prompts like "bird" to generate pseudo-labels on incoming images.

Phase 1 — Transfer Learning from Public Datasets
  Fine-tune on large public bird datasets (NABirds, CUB-200, iNaturalist)
  which provide hundreds of thousands of labelled examples.

Phase 2 — Active Learning Loop
  Route low-confidence detections to human annotators via Label Studio.
  Ornithologists correct / verify labels → model retrains.

Phase 3 — Synthetic Data Augmentation
  Generate synthetic training images by compositing bird cutouts onto
  forest backgrounds with realistic lighting, shadows, and occlusions.
"""

import json
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)


# ============================================================================
# Phase 0 — Foundation Model Pseudo-Labelling
# ============================================================================

class FoundationModelLabeler:
    """
    Uses a pre-trained open-vocabulary detector to generate initial
    pseudo-labels without ANY task-specific training data.

    Supported backends:
      - Grounding DINO (best accuracy, GPU required)
      - OWL-ViT v2     (good accuracy, lighter weight)
      - YOLO-World      (fastest, still decent)
    """

    TEXT_PROMPTS = [
        "bird",
        "black bird",
        "bird perched on branch",
        "bird in flight",
        "bird on ground",
    ]

    def __init__(self, backend: str = "grounding_dino", device: str = "cuda:0"):
        self.backend = backend
        self.device = device
        self.model = None  # Loaded lazily

    def load_model(self):
        """Load the foundation model."""
        if self.backend == "grounding_dino":
            # In production:
            # from groundingdino.util.inference import load_model
            # self.model = load_model(
            #     "groundingdino/config/GroundingDINO_SwinB_cfg.py",
            #     "weights/groundingdino_swinb_cogcoor.pth",
            #     device=self.device,
            # )
            logger.info("Loaded Grounding DINO model on %s", self.device)

        elif self.backend == "owl_vit":
            # from transformers import OwlViTProcessor, OwlViTForObjectDetection
            # self.processor = OwlViTProcessor.from_pretrained(
            #     "google/owlvit-large-patch14"
            # )
            # self.model = OwlViTForObjectDetection.from_pretrained(
            #     "google/owlvit-large-patch14"
            # ).to(self.device)
            logger.info("Loaded OWL-ViT model on %s", self.device)

        elif self.backend == "yolo_world":
            # from ultralytics import YOLOWorld
            # self.model = YOLOWorld("yolov8x-worldv2.pt")
            # self.model.set_classes(self.TEXT_PROMPTS)
            logger.info("Loaded YOLO-World model on %s", self.device)

    def generate_pseudo_labels(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.30,
    ) -> list[dict]:
        """
        Run zero-shot detection and return pseudo-labels.

        Returns:
            List of dicts with keys:
              bbox: [x1, y1, x2, y2] normalised to [0, 1]
              class_name: str
              confidence: float
              is_pseudo: True  (flag for downstream filtering)
        """
        # --- Placeholder: in production, run actual inference ---
        # For Grounding DINO:
        #   from groundingdino.util.inference import predict
        #   boxes, logits, phrases = predict(
        #       self.model, image, "bird . black bird .",
        #       box_threshold=confidence_threshold,
        #       text_threshold=0.25,
        #   )
        #   return [
        #       {"bbox": box.tolist(), "class_name": phrase,
        #        "confidence": float(logit), "is_pseudo": True}
        #       for box, logit, phrase in zip(boxes, logits, phrases)
        #   ]
        logger.info("Generated pseudo-labels for image (placeholder)")
        return []


# ============================================================================
# Phase 1 — Public Dataset Integration (Transfer Learning)
# ============================================================================

@dataclass
class PublicDatasetConfig:
    """Configuration for integrating a public bird dataset."""
    name: str
    source_url: str
    num_classes: int
    num_images: int
    format: str  # "coco", "voc", "csv"
    notes: str = ""


# Registry of useful public datasets for transfer learning
PUBLIC_DATASETS = [
    PublicDatasetConfig(
        name="NABirds",
        source_url="https://dl.allaboutbirds.org/nabirds",
        num_classes=555,
        num_images=48_562,
        format="csv",
        notes="North American birds; excellent species-level labels.",
    ),
    PublicDatasetConfig(
        name="CUB-200-2011",
        source_url="https://www.vision.caltech.edu/datasets/cub_200_2011/",
        num_classes=200,
        num_images=11_788,
        format="csv",
        notes="Fine-grained bird classification benchmark. Bounding boxes included.",
    ),
    PublicDatasetConfig(
        name="iNaturalist-Aves",
        source_url="https://github.com/visipedia/inat_comp",
        num_classes=1486,
        num_images=214_295,
        format="coco",
        notes="Subset of iNaturalist competition; global bird species.",
    ),
    PublicDatasetConfig(
        name="BirdSnap",
        source_url="https://thomasberg.org/",
        num_classes=500,
        num_images=49_829,
        format="csv",
        notes="Good colour diversity; useful for colour classifier pre-training.",
    ),
]


class DatasetConverter:
    """Converts public datasets into a unified COCO-format for training."""

    UNIFIED_CATEGORIES = [
        {"id": 0, "name": "bird", "supercategory": "animal"},
    ]

    def convert_to_coco(
        self,
        source_dir: Path,
        output_path: Path,
        dataset_config: PublicDatasetConfig,
    ) -> dict:
        """
        Convert any supported public dataset into COCO JSON format.

        Regardless of the source dataset's species-level labels, we map
        all bird detections to a single 'bird' category for the detector
        (species classification is handled downstream by the classifier).
        """
        coco = {
            "info": {
                "description": f"Converted from {dataset_config.name}",
                "version": "1.0",
                "year": 2025,
            },
            "categories": self.UNIFIED_CATEGORIES,
            "images": [],
            "annotations": [],
        }

        if dataset_config.format == "coco":
            coco = self._convert_coco_subset(source_dir, coco)
        elif dataset_config.format == "voc":
            coco = self._convert_voc(source_dir, coco)
        elif dataset_config.format == "csv":
            coco = self._convert_csv(source_dir, coco)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(coco, indent=2))
        logger.info(
            "Converted %s: %d images, %d annotations",
            dataset_config.name, len(coco["images"]), len(coco["annotations"]),
        )
        return coco

    def _convert_coco_subset(self, source_dir: Path, coco: dict) -> dict:
        """Filter an existing COCO JSON to keep only bird (Aves) entries."""
        ann_file = source_dir / "annotations.json"
        if ann_file.exists():
            src = json.loads(ann_file.read_text())
            bird_cat_ids = {
                c["id"] for c in src.get("categories", [])
                if "bird" in c.get("supercategory", "").lower()
                or "aves" in c.get("supercategory", "").lower()
            }
            img_ids = set()
            ann_id = 0
            for ann in src.get("annotations", []):
                if ann["category_id"] in bird_cat_ids:
                    img_ids.add(ann["image_id"])
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": ann["image_id"],
                        "category_id": 0,  # unified 'bird'
                        "bbox": ann["bbox"],
                        "area": ann.get("area", 0),
                        "iscrowd": 0,
                    })
                    ann_id += 1
            coco["images"] = [
                img for img in src.get("images", []) if img["id"] in img_ids
            ]
        return coco

    def _convert_voc(self, source_dir: Path, coco: dict) -> dict:
        """Placeholder for Pascal VOC XML → COCO conversion."""
        logger.info("VOC conversion (placeholder)")
        return coco

    def _convert_csv(self, source_dir: Path, coco: dict) -> dict:
        """Placeholder for CSV-based dataset → COCO conversion."""
        logger.info("CSV conversion (placeholder)")
        return coco


# ============================================================================
# Phase 2 — Active Learning Manager
# ============================================================================

class ActiveLearningManager:
    """
    Routes uncertain predictions to human annotators and prioritises
    the most informative samples for labelling.

    Uncertainty strategies:
      - Least-confidence: samples where max(P) is lowest
      - Margin: samples where top-2 class probabilities are close
      - Entropy: samples with highest prediction entropy
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.50,
        budget_per_cycle: int = 200,
        strategy: str = "least_confidence",
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.budget = budget_per_cycle
        self.strategy = strategy
        self._queue: list[dict] = []

    def score_uncertainty(self, predictions: list[dict]) -> list[dict]:
        """Score each prediction's uncertainty and add to the queue."""
        scored = []
        for pred in predictions:
            conf = pred.get("confidence", 1.0)

            if self.strategy == "least_confidence":
                uncertainty = 1.0 - conf
            elif self.strategy == "margin":
                # Would use top-2 class probs if available
                uncertainty = 1.0 - conf
            elif self.strategy == "entropy":
                p = max(conf, 1e-8)
                uncertainty = -(p * np.log(p) + (1 - p) * np.log(1 - p))
            else:
                uncertainty = 1.0 - conf

            pred["uncertainty"] = float(uncertainty)
            scored.append(pred)
        return scored

    def select_for_annotation(self, scored_predictions: list[dict]) -> list[dict]:
        """Select the top-N most uncertain samples for human review."""
        uncertain = [
            p for p in scored_predictions
            if p["uncertainty"] > (1 - self.uncertainty_threshold)
        ]
        uncertain.sort(key=lambda x: x["uncertainty"], reverse=True)
        selected = uncertain[: self.budget]
        self._queue.extend(selected)
        logger.info(
            "Selected %d / %d uncertain samples for annotation",
            len(selected), len(scored_predictions),
        )
        return selected

    def export_to_label_studio(self, samples: list[dict], output_path: Path):
        """Export selected samples as Label Studio JSON tasks."""
        tasks = []
        for s in samples:
            tasks.append({
                "data": {
                    "image": s.get("image_path", ""),
                },
                "predictions": [{
                    "model_version": "v0-pseudo",
                    "result": [{
                        "type": "rectanglelabels",
                        "from_name": "label",
                        "to_name": "image",
                        "value": {
                            "x": s["bbox"][0] * 100,
                            "y": s["bbox"][1] * 100,
                            "width": (s["bbox"][2] - s["bbox"][0]) * 100,
                            "height": (s["bbox"][3] - s["bbox"][1]) * 100,
                            "rectanglelabels": [s.get("class_name", "bird")],
                        },
                    }],
                }],
            })
        output_path.write_text(json.dumps(tasks, indent=2))
        logger.info("Exported %d tasks to Label Studio format", len(tasks))


# ============================================================================
# Phase 3 — Synthetic Data Generator
# ============================================================================

class SyntheticDataGenerator:
    """
    Generates synthetic training images by compositing bird cutouts
    onto forest background images with realistic augmentations.

    Approach:
      1. Collect bird cutout images (from public datasets or manual segmentation).
      2. Collect equatorial forest backgrounds (from camera feed).
      3. Composite birds onto backgrounds with:
         - Random scale, rotation, position
         - Colour-matched lighting adjustment
         - Soft-edge blending to avoid hard cutout borders
         - Shadow generation
         - Occlusion by overlaying semi-transparent leaf patches
    """

    def __init__(self, backgrounds_dir: Path, cutouts_dir: Path):
        self.backgrounds_dir = backgrounds_dir
        self.cutouts_dir = cutouts_dir

    def generate_image(
        self,
        num_birds: int = 3,
        include_black_bird: bool = True,
        output_size: tuple[int, int] = (1920, 1080),
    ) -> tuple[Image.Image, list[dict]]:
        """
        Generate a single synthetic image with annotations.

        Returns:
            (composite_image, annotations)
            where each annotation is {"bbox": [x1,y1,x2,y2], "class": str}
        """
        # Load a random background
        bg = self._load_random_background(output_size)
        annotations = []

        for i in range(num_birds):
            is_black = include_black_bird and i == 0
            bird_cutout = self._load_random_cutout(color="black" if is_black else None)

            # Random transform
            scale = random.uniform(0.03, 0.15)
            bird_w = int(output_size[0] * scale)
            bird_h = int(bird_w * bird_cutout.size[1] / bird_cutout.size[0])
            bird_cutout = bird_cutout.resize((bird_w, bird_h), Image.LANCZOS)

            # Random rotation
            angle = random.uniform(-30, 30)
            bird_cutout = bird_cutout.rotate(angle, expand=True, resample=Image.BICUBIC)

            # Random position (ensure bird is within frame)
            max_x = output_size[0] - bird_cutout.size[0]
            max_y = output_size[1] - bird_cutout.size[1]
            if max_x < 0 or max_y < 0:
                continue
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            # Colour-match bird to background lighting
            bird_cutout = self._match_lighting(bird_cutout, bg, x, y)

            # Add soft shadow
            bg = self._add_shadow(bg, bird_cutout, x, y)

            # Composite
            if bird_cutout.mode == "RGBA":
                bg.paste(bird_cutout, (x, y), bird_cutout)
            else:
                bg.paste(bird_cutout, (x, y))

            # Store annotation
            x2 = x + bird_cutout.size[0]
            y2 = y + bird_cutout.size[1]
            annotations.append({
                "bbox": [x, y, x2, y2],
                "class": "black_bird" if is_black else "bird",
                "is_synthetic": True,
            })

        return bg, annotations

    def generate_batch(
        self, count: int, output_dir: Path
    ) -> list[dict]:
        """Generate a batch of synthetic images with COCO annotations."""
        output_dir.mkdir(parents=True, exist_ok=True)
        all_annotations = []

        for i in range(count):
            num_birds = random.randint(1, 6)
            has_black = random.random() < 0.3  # 30% chance of black bird
            img, anns = self.generate_image(
                num_birds=num_birds, include_black_bird=has_black
            )
            fname = f"synthetic_{i:05d}.jpg"
            img.save(output_dir / fname, quality=90)
            for ann in anns:
                ann["image_file"] = fname
            all_annotations.extend(anns)

        logger.info("Generated %d synthetic images in %s", count, output_dir)
        return all_annotations

    # -- helpers -----------------------------------------------------------

    def _load_random_background(self, size: tuple[int, int]) -> Image.Image:
        """Load and resize a random forest background."""
        bg_files = list(self.backgrounds_dir.glob("*.jpg")) + \
                   list(self.backgrounds_dir.glob("*.png"))
        if not bg_files:
            # Generate a green placeholder if no backgrounds available
            return Image.new("RGB", size, (34, 85, 34))
        bg = Image.open(random.choice(bg_files)).convert("RGB")
        return bg.resize(size, Image.LANCZOS)

    def _load_random_cutout(self, color: Optional[str] = None) -> Image.Image:
        """Load a random bird cutout (RGBA with transparency)."""
        subdir = self.cutouts_dir / color if color else self.cutouts_dir
        cutout_files = list(subdir.glob("*.png")) if subdir.exists() else []
        if not cutout_files:
            # Placeholder: create a simple bird silhouette
            return self._generate_placeholder_bird(color or "brown")
        return Image.open(random.choice(cutout_files)).convert("RGBA")

    @staticmethod
    def _generate_placeholder_bird(color: str) -> Image.Image:
        """Create a simple bird-shaped placeholder for testing."""
        colors = {
            "black": (20, 20, 20),
            "brown": (120, 80, 40),
            "green": (50, 120, 50),
            "white": (240, 240, 240),
        }
        c = colors.get(color, (100, 100, 100))
        img = Image.new("RGBA", (80, 50), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Simple bird shape: ellipse body + triangle beak
        draw.ellipse([10, 10, 60, 40], fill=(*c, 255))
        draw.polygon([(60, 22), (78, 25), (60, 28)], fill=(*c, 255))
        return img

    @staticmethod
    def _match_lighting(
        bird: Image.Image, bg: Image.Image, x: int, y: int
    ) -> Image.Image:
        """Adjust bird brightness to match the local background region."""
        region = bg.crop((x, y, x + bird.size[0], y + bird.size[1]))
        bg_brightness = np.array(region).mean() / 255.0
        bird_arr = np.array(bird).astype(np.float32)
        # Subtle adjustment: blend toward background brightness
        factor = 0.7 + 0.6 * bg_brightness  # range ~[0.7, 1.3]
        bird_arr[..., :3] = np.clip(bird_arr[..., :3] * factor, 0, 255)
        return Image.fromarray(bird_arr.astype(np.uint8))

    @staticmethod
    def _add_shadow(
        bg: Image.Image, bird: Image.Image, x: int, y: int
    ) -> Image.Image:
        """Add a soft drop shadow beneath the bird."""
        if bird.mode != "RGBA":
            return bg
        shadow = Image.new("RGBA", bird.size, (0, 0, 0, 0))
        alpha = bird.split()[-1]
        shadow.putalpha(alpha)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=5))
        # Offset shadow slightly down-right
        sx, sy = x + 4, y + 6
        sx = min(sx, bg.size[0] - shadow.size[0])
        sy = min(sy, bg.size[1] - shadow.size[1])
        if sx >= 0 and sy >= 0:
            bg.paste(shadow, (sx, sy), shadow)
        return bg
