"""
Aero-Watch Bird Colour Classifier — PRODUCTION VERSION
=======================================================

Three classification modes (in priority order):

  1. NEURAL — DINOv2 ViT-S/14 backbone (frozen) + trained linear head
     Best accuracy and generalization. DINOv2 was pretrained on 142M
     images without labels, producing features that transfer across
     domains without fine-tuning. Only the linear head is trained.
     Falls back to EfficientNetV2-S for legacy weight compatibility.

  2. HYBRID — Pixel-based black detector + CLIP for non-black colours

  3. HEURISTIC — HSV rules (no model needed)
"""

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class BirdColor(IntEnum):
    BLACK = 0
    WHITE = 1
    BROWN = 2
    GREEN = 3
    RED = 4
    BLUE = 5
    YELLOW = 6
    GREY = 7
    MIXED = 8

    @classmethod
    def from_name(cls, name: str) -> "BirdColor":
        return cls[name.upper()]


COLOR_NAMES = [c.name.lower() for c in BirdColor]


@dataclass
class ColorPrediction:
    primary_color: str
    confidence: float
    probabilities: dict[str, float]
    is_black_bird: bool


# ---------------------------------------------------------------------------
# Stage 1: Pixel-based Black Bird Detector
# ---------------------------------------------------------------------------

class BlackBirdDetector:
    """
    Determines if a bird crop is BLACK using pixel statistics.

    Why not CLIP? CLIP cosine similarities for "black bird" vs "brown bird"
    vs "grey bird" converge at low brightness — black literally never wins.
    Pixel analysis is definitive for the black/not-black question.

    Key design decisions (from CUB-200 evaluation):
      - Analyze CENTER 50% of crop (excludes background at edges)
      - Use PERCENTILE brightness, not mean (robust to bright background)
      - DARK PIXEL DOMINANCE: what fraction is actually dark? This is the
        key feature that separates "black bird" from "mixed bird with some
        dark patches" (fixes the 61 FP problem)
      - Skip saturation/RGB checks at very low brightness to handle
        iridescent black birds like Ani and Brewer's Blackbird (fixes 8 FN)
    """

    def detect(self, crop: np.ndarray) -> tuple[bool, float, dict]:
        h, w = crop.shape[:2]

        # --- Center crop (bird body) ---
        my, mx = h // 4, w // 4
        center = crop[my:h - my, mx:w - mx]
        if center.size == 0:
            center = crop

        # --- HSV on center ---
        center_hsv = np.array(
            Image.fromarray(center).convert("HSV"), dtype=np.float32
        )
        v_flat = center_hsv[:, :, 2].ravel()
        s_flat = center_hsv[:, :, 1].ravel()

        v_p25 = float(np.percentile(v_flat, 25))
        v_p50 = float(np.percentile(v_flat, 50))

        # --- DARK PIXEL DOMINANCE ---
        # What fraction of center pixels have V < 80?
        # (V<100 was too generous — brown birds in shade pass that easily)
        dark_pixel_frac = float(np.mean(v_flat < 80))

        # --- RGB on darkest 40% of pixels ---
        center_f = center.astype(np.float32)
        brightness_pp = center_f.mean(axis=2).ravel()
        dark_thresh = np.percentile(brightness_pp, 40)
        dark_mask = brightness_pp <= max(dark_thresh, 1.0)

        flat_r = center_f[:, :, 0].ravel()
        flat_g = center_f[:, :, 1].ravel()
        flat_b = center_f[:, :, 2].ravel()

        if dark_mask.sum() > 10:
            dr = float(flat_r[dark_mask].mean())
            dg = float(flat_g[dark_mask].mean())
            db = float(flat_b[dark_mask].mean())
            dark_brightness = float(brightness_pp[dark_mask].mean())
            s_of_dark = float(s_flat[dark_mask].mean())
        else:
            dr, dg, db = float(flat_r.mean()), float(flat_g.mean()), float(flat_b.mean())
            dark_brightness = float(brightness_pp.mean())
            s_of_dark = float(s_flat.mean())

        dark_rgb = np.array([dr, dg, db])
        dark_rgb_ratio = float(dark_rgb.max() / max(dark_rgb.min(), 1.0))

        features = {
            "v_p25": round(v_p25, 1),
            "v_p50": round(v_p50, 1),
            "dark_pixel_frac": round(dark_pixel_frac, 3),
            "dark_brightness": round(dark_brightness, 1),
            "s_of_dark": round(s_of_dark, 1),
            "dark_rgb_ratio": round(dark_rgb_ratio, 3),
            "dark_rgb": [round(dr, 1), round(dg, 1), round(db, 1)],
        }

        # ── Scoring ──────────────────────────────────────────────────────
        score = 0.0

        # TRIPLE HARD GATE:
        # 1. Majority of center must be dark (V < 80)
        # 2. Bird body must be dark (25th percentile < 60)
        # 3. Bird must be UNIFORMLY dark (median < 30)
        #    This is the key FP killer: mixed birds have dark patches
        #    (low v_p25) but bright patches too (high v_p50).
        #    True black birds are dark EVERYWHERE (v_p50 < 10 typically).
        if dark_pixel_frac < 0.50 or v_p25 > 60 or v_p50 > 30:
            features["black_score"] = 0.0
            features["is_black"] = False
            return False, 0.0, features

        # FEATURE 1: Dark pixel dominance
        if dark_pixel_frac >= 0.65:
            score += 0.30
        elif dark_pixel_frac >= 0.50:
            score += 0.20
        elif dark_pixel_frac >= 0.45:
            score += 0.10

        # FEATURE 2: 25th percentile brightness (bird body darkness)
        if v_p25 < 35:
            score += 0.25
        elif v_p25 < 50:
            score += 0.20
        elif v_p25 < 70:
            score += 0.12
        elif v_p25 < 90:
            score += 0.05

        # FEATURE 3: Brightness of darkest 40% of pixels
        if dark_brightness < 40:
            score += 0.20
        elif dark_brightness < 55:
            score += 0.15
        elif dark_brightness < 70:
            score += 0.08

        # FEATURE 4: Saturation + RGB (only for MODERATE darkness)
        if v_p25 >= 50:
            if s_of_dark < 40:
                score += 0.15
            elif s_of_dark < 60:
                score += 0.08

            if dark_rgb_ratio < 1.15:
                score += 0.10
            elif dark_rgb_ratio < 1.30:
                score += 0.05

        features["black_score"] = round(score, 3)

        is_black = score >= 0.55
        confidence = min(score / 0.80, 0.95) if is_black else 0.0

        features["is_black"] = is_black
        return is_black, round(confidence, 3), features


# ---------------------------------------------------------------------------
# Stage 2: CLIP Non-Black Colour Classifier
# ---------------------------------------------------------------------------

class CLIPColorClassifier:
    """CLIP-based classifier for non-black bird colours."""

    NON_BLACK_PROMPTS = {
        "white":  [
            "a white bird with bright white feathers",
            "a pure white colored bird like an egret or swan",
            "a snowy white bird",
        ],
        "brown":  [
            "a brown bird with warm brown feathers",
            "a light brown or tan bird like a sparrow",
            "a chestnut brown bird with earthy brown tones",
        ],
        "green":  [
            "a green bird with green feathers",
            "a bright green colored bird like a parrot",
            "a bird with vivid emerald green plumage",
        ],
        "red":    [
            "a red bird with red feathers",
            "a bright red colored bird like a cardinal",
            "a crimson scarlet red bird",
        ],
        "blue":   [
            "a blue bird with blue feathers",
            "a bright blue colored bird like a bluebird or jay",
            "a vivid blue bird with azure plumage",
        ],
        "yellow": [
            "a yellow bird with yellow feathers",
            "a bright yellow colored bird like a goldfinch",
            "a golden yellow bird with sunny plumage",
        ],
        "grey":   [
            "a grey bird with silvery grey feathers",
            "a pale grey or ash colored bird",
            "a light grey bird like a mockingbird",
        ],
        "mixed":  [
            "a multicolored bird with several distinct colors",
            "a bird with patches of many different colors",
            "a colorful patterned bird with mixed plumage",
        ],
    }

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu"):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.model = None
        self.preprocess = None
        self._color_features: dict[str, object] = {}

    def load(self):
        import open_clip
        import torch

        logger.info("Loading CLIP colour classifier (%s)...", self.model_name)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device,
        )
        tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()

        with torch.no_grad():
            for color, prompts in self.NON_BLACK_PROMPTS.items():
                tokens = tokenizer(prompts).to(self.device)
                features = self.model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
                avg = features.mean(dim=0, keepdim=True)
                avg /= avg.norm(dim=-1, keepdim=True)
                self._color_features[color] = avg

        logger.info("CLIP colour classifier ready (%d non-black colours).", len(self._color_features))

    def classify_non_black(self, crop: np.ndarray) -> ColorPrediction:
        import torch

        pil_img = Image.fromarray(crop).convert("RGB")
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_features = self.model.encode_image(img_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            scores = {}
            for color, text_feat in self._color_features.items():
                sim = (img_features @ text_feat.T)[0, 0].cpu().item()
                scores[color] = sim

        non_black_names = list(self.NON_BLACK_PROMPTS.keys())
        score_arr = np.array([scores[c] for c in non_black_names])
        temperature = 0.5
        exp_scores = np.exp((score_arr - score_arr.max()) / temperature)
        probs = exp_scores / exp_scores.sum()

        prob_dict = {"black": 0.0}
        for i, c in enumerate(non_black_names):
            prob_dict[c] = round(float(probs[i]), 4)

        best_idx = int(np.argmax(probs))
        primary = non_black_names[best_idx]
        confidence = float(probs[best_idx])

        return ColorPrediction(
            primary_color=primary,
            confidence=float(confidence),
            probabilities=prob_dict,
            is_black_bird=False,
        )


# ---------------------------------------------------------------------------
# Main Classifier
# ---------------------------------------------------------------------------

class BirdColorClassifier:
    """
    Two-stage classifier:
      Stage 1: BlackBirdDetector (pixel-based) — is it black?
      Stage 2: CLIPColorClassifier — if not black, what colour?
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = len(BirdColor),
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.num_classes = num_classes
        self.conf_thresh = confidence_threshold
        self.device = device
        self._mode = "heuristic"

        self.black_detector = BlackBirdDetector()
        self.clip_classifier: Optional[CLIPColorClassifier] = None
        self.neural_model = None
        self._backbone = None  # DINOv2 backbone (frozen)
        self._head = None      # Trained linear head

    def load(self):
        if self.model_path:
            try:
                self._load_neural()
                self._mode = "neural"
                logger.info("Colour classifier: NEURAL mode")
                return
            except Exception as e:
                logger.warning("Neural load failed (%s)", e)

        try:
            clip_device = "cpu" if self.device == "mps" else self.device
            self.clip_classifier = CLIPColorClassifier(device=clip_device)
            self.clip_classifier.load()
            self._mode = "hybrid"
            logger.info("Colour classifier: HYBRID mode (pixel black + CLIP non-black)")
            return
        except ImportError:
            logger.warning("open_clip not installed")
        except Exception as e:
            logger.warning("CLIP failed (%s)", e)

        self._mode = "heuristic"
        logger.info("Colour classifier: HEURISTIC mode")

    def classify(self, crop: np.ndarray) -> ColorPrediction:
        if self._mode == "neural":
            return self._classify_neural(crop)
        elif self._mode == "hybrid":
            return self._classify_hybrid(crop)
        else:
            return self._classify_heuristic(crop)

    def classify_batch(self, crops: list[np.ndarray]) -> list[ColorPrediction]:
        return [self.classify(c) for c in crops]

    def _classify_hybrid(self, crop: np.ndarray) -> ColorPrediction:
        is_black, black_conf, _ = self.black_detector.detect(crop)

        if is_black:
            remaining = 1.0 - black_conf
            per_other = remaining / (len(COLOR_NAMES) - 1)
            prob_dict = {c: round(black_conf if c == "black" else per_other, 4) for c in COLOR_NAMES}
            return ColorPrediction(
                primary_color="black",
                confidence=float(black_conf),
                probabilities=prob_dict,
                is_black_bird=True,
            )

        return self.clip_classifier.classify_non_black(crop)

    def _load_neural(self):
        import torch
        import torch.nn as nn

        # DINOv2 ViT-S/14 backbone (frozen) + trained linear head
        # DINOv2 features are robust across domains without fine-tuning.
        # Only the linear head is trained, making training fast and
        # the model highly generalizable to new environments.
        backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            trust_repo=True,
        )
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        embed_dim = backbone.embed_dim  # 384 for ViT-S
        head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=0.2),
            nn.Linear(embed_dim, self.num_classes),
        )

        # Load trained head weights
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)

        # Handle both formats: full model state_dict or head-only
        if any(k.startswith("backbone.") for k in state_dict):
            # Full model save (backbone + head)
            head_state = {
                k.replace("head.", ""): v
                for k, v in state_dict.items()
                if k.startswith("head.")
            }
            if head_state:
                head.load_state_dict(head_state)
            else:
                # Try loading as-is (legacy EfficientNet format fallback)
                logger.warning("DINOv2 head weights not found, attempting EfficientNet fallback")
                self._load_neural_efficientnet(state_dict)
                return
        else:
            # Head-only save
            head.load_state_dict(state_dict)

        self._backbone = backbone.to(self.device)
        self._head = head.to(self.device)
        self._head.eval()
        self.neural_model = True  # flag that neural mode is active

    def _load_neural_efficientnet(self, state_dict):
        """Fallback for loading legacy EfficientNet weights."""
        import torch
        import torch.nn as nn
        import torchvision.models as models

        self._backbone = None
        model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3), nn.Linear(in_features, 512),
            nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(512, self.num_classes),
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self._efficientnet = model

    def _classify_neural(self, crop: np.ndarray) -> ColorPrediction:
        import torch
        import torchvision.transforms as T

        img = Image.fromarray(crop).convert("RGB")

        if self._backbone is not None:
            # DINOv2 path
            transform = T.Compose([
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensor = transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self._backbone(tensor)  # [1, embed_dim]
                logits = self._head(features)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        elif hasattr(self, "_efficientnet"):
            # EfficientNet fallback
            transform = T.Compose([
                T.Resize((224, 224)), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensor = transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self._efficientnet(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        else:
            raise RuntimeError("No neural model loaded")

        idx = int(np.argmax(probs))
        color = COLOR_NAMES[idx]
        conf = float(probs[idx])
        prob_dict = {COLOR_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)}
        return ColorPrediction(
            primary_color=color, confidence=float(conf),
            probabilities=prob_dict,
            is_black_bird=bool(color == "black" and conf >= self.conf_thresh),
        )

    def _classify_heuristic(self, crop: np.ndarray) -> ColorPrediction:
        hsv = np.array(Image.fromarray(crop).convert("HSV"), dtype=np.float32)
        h, s, v = float(hsv[:, :, 0].mean()), float(hsv[:, :, 1].mean()), float(hsv[:, :, 2].mean())
        hue_360 = h * 360.0 / 255.0

        is_black, black_conf, _ = self.black_detector.detect(crop)
        if is_black:
            color, confidence = "black", black_conf
        elif v > 210 and s < 40:
            color, confidence = "white", 0.8
        elif s < 35:
            color, confidence = "grey", 0.6
        elif hue_360 < 20 or hue_360 > 340:
            color, confidence = "red", 0.6
        elif 20 <= hue_360 < 45:
            color, confidence = "brown", 0.55
        elif 45 <= hue_360 < 70:
            color, confidence = "yellow", 0.6
        elif 70 <= hue_360 < 160:
            color, confidence = "green", 0.6
        elif 160 <= hue_360 < 260:
            color, confidence = "blue", 0.6
        else:
            color, confidence = "brown", 0.5

        remaining = 1.0 - confidence
        per_other = remaining / (len(COLOR_NAMES) - 1)
        prob_dict = {c: round(confidence if c == color else per_other, 4) for c in COLOR_NAMES}
        return ColorPrediction(
            primary_color=color, confidence=float(confidence),
            probabilities=prob_dict,
            is_black_bird=bool(color == "black" and confidence >= self.conf_thresh),
        )
