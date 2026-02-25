#!/usr/bin/env python3
"""
Train Bird Colour Classifier — DINOv2 ViT-S/14 on CUB-200-2011
================================================================

Trains a linear head on frozen DINOv2 features. DINOv2 (Meta, 2023)
was pretrained on 142M images without labels — its features generalize
across domains without fine-tuning. Only the lightweight head trains,
making this fast (~10 min on Mac MPS, ~3 min on CUDA).

Saved weights are head-only (~15KB), loaded by BirdColorClassifier.

Usage:
  # Quick test (~2 min on Mac)
  python3 train_color_classifier.py --dataset-dir ./CUB_200_2011 --epochs 3 --batch-size 16

  # Full training (~10 min on Mac MPS, ~3 min on CUDA)
  python3 train_color_classifier.py --dataset-dir ./CUB_200_2011 --epochs 25 --batch-size 32

Output:
  color_classifier_best.pt  — best validation head weights
  color_classifier_last.pt  — final epoch head weights

To use trained weights:
  python3 run_pipeline.py --color-model color_classifier_best.pt ...
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

# ── Colour scheme (must match color_classifier.py) ────────────────────────

COLOR_NAMES = ["black", "white", "brown", "green", "red", "blue", "yellow", "grey", "mixed"]
NUM_CLASSES = len(COLOR_NAMES)

# CUB attribute → our colour
CUB_COLOR_MAP = {
    "black": "black", "blue": "blue", "brown": "brown", "buff": "brown",
    "grey": "grey", "green": "green", "olive": "green", "orange": "brown",
    "pink": "red", "purple": "mixed", "rufous": "brown", "red": "red",
    "white": "white", "yellow": "yellow", "iridescent": "mixed",
}

# Authoritative overrides
SPECIES_COLOR_OVERRIDE = {
    "004.Groove_billed_Ani": "black",
    "009.Brewer_Blackbird": "black",
    "011.Rusty_Blackbird": "black",
    "026.Bronzed_Cowbird": "black",
    "027.Shiny_Cowbird": "black",
    "029.American_Crow": "black",
    "030.Fish_Crow": "black",
    "001.Black_footed_Albatross": "brown",
    "002.Laysan_Albatross": "white",
    "023.Brandt_Cormorant": "black",
    "024.Red_faced_Cormorant": "black",
    "025.Pelagic_Cormorant": "black",
    "017.Cardinal": "red",
    "036.Northern_Flicker": "brown",
    "010.Red_winged_Blackbird": "black",
    "014.Indigo_Bunting": "blue",
    "047.American_Goldfinch": "yellow",
    "012.Yellow_headed_Blackbird": "black",
    "019.Gray_Catbird": "grey",
    "034.Gray_crowned_Rosy_Finch": "grey",
    "173.Orange_crowned_Warbler": "green",
    "160.Black_throated_Blue_Warbler": "blue",
    "188.Pileated_Woodpecker": "mixed",
    "033.Yellow_billed_Cuckoo": "brown",
    "038.Great_Crested_Flycatcher": "brown",
    "107.Common_Raven": "black",
    "065.Slaty_backed_Gull": "white",
    "079.Belted_Kingfisher": "grey",
    "100.Brown_Pelican": "brown",
    "133.White_throated_Sparrow": "brown",
    "134.Grasshopper_Sparrow": "brown",
}


def _color_from_species_name(name: str) -> str:
    name = name.lower()
    rules = [
        (["crow", "raven", "grackle", "ani", "cowbird", "starling", "cormorant"], "black"),
        (["blackbird"], "black"),
        (["egret", "swan", "tern", "gull"], "white"),
        (["cardinal", "tanager"], "red"),
        (["bluebird", "indigo"], "blue"),
        (["goldfinch", "yellowthroat"], "yellow"),
        (["vireo", "hummingbird"], "green"),
        (["mockingbird", "gray_catbird", "grey"], "grey"),
        (["sparrow", "wren", "creeper", "thrasher", "finch", "flycatcher", "warbler"], "brown"),
    ]
    for keywords, color in rules:
        for kw in keywords:
            if kw in name:
                return color
    return "mixed"


# ── CUB-200 Dataset ──────────────────────────────────────────────────────

class CUBColorDataset(Dataset):
    """
    CUB-200-2011 dataset that yields (crop_tensor, color_label) pairs.
    Crops birds using GT bounding boxes.
    """

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples = []  # list of (image_path, bbox_xyxy, color_idx)

        # Load dataset files
        images = {}
        for line in (self.root / "images.txt").read_text().strip().split("\n"):
            parts = line.strip().split()
            images[int(parts[0])] = parts[1]

        bboxes = {}
        for line in (self.root / "bounding_boxes.txt").read_text().strip().split("\n"):
            parts = line.strip().split()
            img_id = int(parts[0])
            x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            bboxes[img_id] = [x, y, x + w, y + h]

        split_map = {}
        for line in (self.root / "train_test_split.txt").read_text().strip().split("\n"):
            parts = line.strip().split()
            split_map[int(parts[0])] = int(parts[1]) == 1  # True = train

        classes = {}
        for line in (self.root / "classes.txt").read_text().strip().split("\n"):
            parts = line.strip().split(maxsplit=1)
            classes[int(parts[0])] = parts[1]

        image_labels = {}
        for line in (self.root / "image_class_labels.txt").read_text().strip().split("\n"):
            parts = line.strip().split()
            image_labels[int(parts[0])] = int(parts[1])

        # Load CUB attributes for color
        primary_color_attrs = {}
        attr_file = self.root / "attributes" / "attributes.txt"
        if attr_file.exists():
            for line in attr_file.read_text().strip().split("\n"):
                parts = line.strip().split(maxsplit=1)
                attr_id, attr_name = int(parts[0]), parts[1]
                if "has_primary_color" in attr_name.lower():
                    primary_color_attrs[attr_id] = attr_name.split("::")[-1].strip().lower()

        attr_votes: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        attr_label_file = self.root / "attributes" / "image_attribute_labels.txt"
        if primary_color_attrs and attr_label_file.exists():
            for line in attr_label_file.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                img_id, attr_id = int(parts[0]), int(parts[1])
                is_present, certainty = int(parts[2]), int(parts[3])
                if attr_id in primary_color_attrs and is_present == 1 and certainty >= 3:
                    weight = 1.0 if certainty == 3 else 2.0
                    cub_color = primary_color_attrs[attr_id]
                    our_color = CUB_COLOR_MAP.get(cub_color, "mixed")
                    attr_votes[img_id][our_color] += weight

        # Build samples
        is_train = (split == "train")
        color_dist = defaultdict(int)

        for img_id, rel_path in images.items():
            if split_map.get(img_id) != is_train:
                continue

            class_id = image_labels.get(img_id, 0)
            species = classes.get(class_id, "")

            # Get color label (same priority as evaluator)
            if species in SPECIES_COLOR_OVERRIDE:
                color = SPECIES_COLOR_OVERRIDE[species]
            elif img_id in attr_votes and attr_votes[img_id]:
                color = max(attr_votes[img_id], key=attr_votes[img_id].get)
            else:
                color = _color_from_species_name(species)

            color_idx = COLOR_NAMES.index(color)
            img_path = str(self.root / "images" / rel_path)
            bbox = bboxes.get(img_id, [0, 0, 100, 100])

            self.samples.append((img_path, bbox, color_idx))
            color_dist[color] += 1

        logger.info("[%s] %d samples, distribution: %s",
                     split, len(self.samples), dict(sorted(color_dist.items())))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, color_idx = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Crop to bounding box with 10% padding
        x1, y1, x2, y2 = bbox
        pad_x = (x2 - x1) * 0.10
        pad_y = (y2 - y1) * 0.10
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))

        crop = img.crop((x1, y1, x2, y2))

        if self.transform:
            crop = self.transform(crop)

        return crop, color_idx


# ── Model ─────────────────────────────────────────────────────────────────

class DINOv2Classifier(nn.Module):
    """
    DINOv2 ViT-S/14 backbone (frozen) + trainable linear head.

    DINOv2 was pretrained on 142M images without labels, producing
    features that transfer across domains without fine-tuning.
    Only the lightweight head is trained — fast and generalizable.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            trust_repo=True,
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        embed_dim = self.backbone.embed_dim  # 384 for ViT-S
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=0.2),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)  # [B, embed_dim]
        return self.head(features)

    def save_head(self, path: str):
        """Save only the head weights (small file, ~15KB)."""
        torch.save(self.head.state_dict(), path)

    def load_head(self, path: str, device: str = "cpu"):
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.head.load_state_dict(state_dict)


def build_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = False):
    """
    DINOv2 ViT-S/14 with linear head.
    Architecture matches BirdColorClassifier._load_neural() exactly.
    freeze_backbone is ignored (backbone is always frozen in DINOv2).
    """
    return DINOv2Classifier(num_classes=num_classes)


# ── Training ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    per_class_tp = defaultdict(int)
    per_class_total = defaultdict(int)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            per_class_total[COLOR_NAMES[l]] += 1
            if p == l:
                per_class_tp[COLOR_NAMES[l]] += 1

    acc = correct / max(total, 1)
    per_class_acc = {}
    for c in COLOR_NAMES:
        if per_class_total[c] > 0:
            per_class_acc[c] = round(per_class_tp[c] / per_class_total[c], 3)

    return total_loss / max(total, 1), acc, per_class_acc


def main():
    parser = argparse.ArgumentParser(description="Train bird colour classifier")
    parser.add_argument("--dataset-dir", required=True, help="Path to CUB_200_2011")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for head (backbone is frozen)")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers (0=main thread, best for Mac)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--output", default="color_classifier_best.pt")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # ── Transforms ────────────────────────────────────────────────────
    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.RandomRotation(15),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Dataset ───────────────────────────────────────────────────────
    train_ds = CUBColorDataset(args.dataset_dir, split="train", transform=train_transform)
    val_ds = CUBColorDataset(args.dataset_dir, split="test", transform=val_transform)

    # Class-weighted sampler to handle imbalanced classes
    class_counts = defaultdict(int)
    for _, _, c in train_ds.samples:
        class_counts[c] += 1
    weights_per_class = {c: 1.0 / max(n, 1) for c, n in class_counts.items()}
    sample_weights = [weights_per_class[c] for _, _, c in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # Class-weighted loss
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor(
        [total_samples / max(class_counts.get(i, 1) * NUM_CLASSES, 1)
         for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model()

    if args.resume:
        model.load_head(args.resume, device="cpu")
        logger.info("Resumed head weights from %s", args.resume)

    model.to(device)

    # ── Optimizer (only head parameters — backbone is frozen) ────────
    head_params = list(model.head.parameters())

    optimizer = optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────
    best_val_acc = 0.0
    logger.info("=" * 70)
    logger.info("Training: DINOv2 ViT-S/14 backbone (frozen) + linear head")
    logger.info("  %d epochs, batch=%d, lr=%.1e", args.epochs, args.batch_size, args.lr)
    logger.info("  Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))
    logger.info("  Head params: %d", sum(p.numel() for p in head_params))
    logger.info("=" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, per_class = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        black_acc = per_class.get("black", 0)
        is_best = val_acc > best_val_acc

        if is_best:
            best_val_acc = val_acc
            model.save_head(args.output)

        logger.info(
            "Epoch %2d/%d │ %.0fs │ train %.3f/%.1f%% │ val %.3f/%.1f%% │ "
            "black=%.0f%% │ %s",
            epoch, args.epochs, elapsed,
            train_loss, train_acc * 100,
            val_loss, val_acc * 100,
            black_acc * 100,
            "★ BEST" if is_best else "",
        )

        # Per-class detail every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            logger.info("  Per-class: %s", per_class)

    # Save final
    last_path = args.output.replace("best", "last")
    model.save_head(last_path)

    logger.info("=" * 70)
    logger.info("Done. Best val acc: %.1f%%", best_val_acc * 100)
    logger.info("  Best weights: %s", args.output)
    logger.info("  Last weights: %s", last_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
