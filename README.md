# Project Aero-Watch — Technical Assessment Solution

## Equatorial Bird Preservation: End-to-End Computer Vision System

**Author:** Karthik Yadati  
**Date:** February 2026  
**Stack:** Python 3.14 · PyTorch · YOLO26 · DINOv2 · CLIP · FastAPI

**Development Approach:** I used Claude as a pair-programming tool throughout this project — for code generation, debugging, architecture discussion, and documentation. All design decisions, system architecture, model selection rationale, and evaluation strategy are my own. I reviewed, tested, and ran every piece of code on my local machine.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Model & Framework Justification](#3-model--framework-justification)
4. [Detection, Counting & Classification Logic](#4-detection-counting--classification-logic)
5. [Data Strategy](#5-data-strategy)
6. [Containerisation & Deployment](#6-containerisation--deployment)
7. [Scaling to 1000s of Cameras](#7-scaling-to-1000s-of-cameras)
8. [Model Drift Tracking](#8-model-drift-tracking)
9. [Retraining, Evaluation & Deployment Pipeline](#9-retraining-evaluation--deployment-pipeline)
10. [Performance Metrics](#10-performance-metrics)
11. [Project Structure](#11-project-structure)
12. [Quick Start](#12-quick-start)

---

## 1. Executive Summary

Aero-Watch is an end-to-end computer vision system that processes images from solar-powered, satellite-connected cameras deployed across equatorial African forests. The system detects and counts birds in each captured frame, and specifically identifies endangered "Black Birds" (Melampitta and similar species) by combining object detection with colour classification.

The pipeline uses three models in sequence:

1. **YOLO26x** — Detects bird-class objects with NMS-free inference and native small-object improvements
2. **CLIP ViT-B/32** — Verifies each crop is actually a bird (rejects branches, leaves, shadows)
3. **DINOv2 ViT-S/14** — Classifies plumage colour into 9 categories via a frozen foundation backbone with a trainable linear head

A dedicated **shadow discriminator** prevents dark canopy shadows from being misclassified as black birds — the single largest source of false positives in equatorial forest imagery.

The system runs offline on a MacBook (Apple MPS) for development and evaluation.

---

## 2. End-to-End System Architecture

The system is organised into seven layers:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        🌿 FIELD LAYER                                │
│  Solar cameras (Sectors A, B, C) → JPEG frames → Satellite uplink   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ HTTPS
┌──────────────────────────▼───────────────────────────────────────────┐
│                      ☁️  INGESTION LAYER                              │
│  API Gateway (rate limit + auth) → Redis queue → S3 (raw archive)   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                    ⚙️  ML PROCESSING PIPELINE                         │
│                                                                      │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────┐               │
│  │ Preprocessor │──▶│  YOLO26x     │──▶│ CLIP       │               │
│  │ white bal.   │   │  NMS-free    │   │ ViT-B/32   │               │
│  │ CLAHE        │   │  detection   │   │ "is bird?" │               │
│  │ letterbox    │   │  detection   │   │ verifier   │               │
│  └─────────────┘   └──────────────┘   └─────┬──────┘               │
│                                              │                       │
│                    ┌─────────────────────────▼──────┐               │
│                    │  Shadow Discriminator           │               │
│                    │  Laplacian · Sobel · HSV · shape│               │
│                    └─────────────┬──────────────────┘               │
│                                  │                                   │
│                    ┌─────────────▼──────────────────┐               │
│                    │  DINOv2 ViT-S/14 Classifier    │               │
│                    │  frozen backbone + linear head  │               │
│                    │  9 colour classes               │               │
│                    └─────────────┬──────────────────┘               │
└──────────────────────────────────┼───────────────────────────────────┘
                                   │ results + crops
┌──────────────────────────────────▼───────────────────────────────────┐
│                    📦 DATA & ANNOTATION LAYER                        │
│                                                                      │
│  SQLite (PoC) / PostgreSQL (prod) ◄── detections, drift state       │
│  Label Studio ◄── pre-filled bboxes, active learning queue          │
│  S3 ◄── model weights, training data, raw images                    │
│  Synthetic Generator ◄── bird-on-forest composites                  │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────┐
│                    🔄 RETRAIN PIPELINE (GitHub Actions)               │
│                                                                      │
│  check_retrain_needed.py                                             │
│    │ drift critical? accuracy dropped? black F1 < 0.70?             │
│    ▼                                                                 │
│  train_color_classifier.py (DINOv2 head only, ~10 min)              │
│    │                                                                 │
│    ▼                                                                 │
│  evaluate_cub200.py (500 test images, 4 seeds)                      │
│    │ candidate better?                                               │
│    ▼                                                                 │
│  Promote to S3 → archive old → rolling restart                      │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────┐
│                    📈 MONITORING & OUTPUT                             │
│                                                                      │
│  Drift Detector: data PSI, prediction JS divergence, confidence Δ   │
│  Prometheus + Grafana: latency p99, bird counts/hr, PSI trends      │
│  Alerts: Slack/PagerDuty (drift critical, API down, black spike)    │
│  Dashboard: HTML (offline) or React (live API polling)              │
│  REST API: GET /health, POST /detect, GET /drift, GET /stats, POST /reset │
└──────────────────────────────────────────────────────────────────────┘
```

**Data flow summary:**
- Camera frames enter the ingestion layer, get queued, and flow through the ML pipeline
- Each detection goes through YOLO26 → CLIP verification → shadow filtering → DINOv2 colour classification
- Results are persisted to SQLite (local PoC) or PostgreSQL (production) — detection history, aggregate stats, and drift baselines survive API restarts
- Low-confidence detections (< 0.6) are routed to Label Studio for ornithologist review
- The drift detector continuously monitors image and prediction distributions; its baseline is persisted so it doesn't need to recollect after a restart
- When drift is critical or metrics degrade, the retrain pipeline fires automatically


### PoC vs Production — What Runs vs What's Described

The architecture above shows the full production target. This PoC implements the core ML pipeline and monitoring end-to-end; the remaining layers are described as strategy.

**What runs locally in this PoC:**
- Full ML pipeline: YOLO26 → CLIP → Shadow Discriminator → DINOv2 colour classification
- SQLite persistence — detections, aggregate stats, and drift baselines survive API restarts
- Drift monitoring — PSI, JS divergence, confidence tracking with persisted baseline
- REST API — `/detect`, `/health`, `/drift`, `/stats`, `/history`, `/metrics`, `/reset`
- Offline HTML dashboard and live React dashboard option
- Retrain pipeline — `check_retrain_needed.py`, `train_color_classifier.py`, `evaluate_cub200.py` as local scripts

**What's described as production strategy (not implemented in PoC):**
- Redis queue + async GPU workers for horizontal scaling (currently synchronous single-server)
- PostgreSQL instead of SQLite
- Grafana dashboards and Slack/PagerDuty alerting (currently log warnings only)
- Live Label Studio integration (currently JSON export via `cold_start_strategy.py`)
- S3 storage for raw images and model weights
- GitHub Actions automated weekly retraining

---

## 3. Model & Framework Justification

### 3.1 Detection: YOLO26x

| Criterion | YOLOv8x | YOLO26x (chosen) | RT-DETR | Faster R-CNN |
|-----------|---------|-------------------|---------|--------------|
| Small object detection | Good (needs SAHI) | Native (ProgLoss + STAL) | Good | Excellent (FPN) |
| NMS requirement | Yes (adds latency) | NMS-free (dual-head) | NMS-free | Yes |
| Inference speed (CPU) | Baseline | 43% faster | Slower | ~3× slower |
| Ultralytics ecosystem | Yes | Yes (drop-in) | Partial | No |
| ONNX/TensorRT export | Excellent | Excellent | Good | Good |

**Why YOLO26x:** Released January 2026, YOLO26 is a drop-in upgrade from YOLOv8 via the same Ultralytics API. Its end-to-end NMS-free inference removes variable-latency post-processing. The ProgLoss and STAL loss functions specifically improve small-object detection — critical for birds at distance — eliminating the need for SAHI tiling that previous YOLO versions required (a ~4× latency multiplier). First run auto-downloads weights from Ultralytics.

### 3.2 Classification: Three Modes (with and without labelled data)

The colour classifier operates in three modes, automatically selecting the best available based on whether trained weights exist:

**Mode 1 — NEURAL (requires labelled data for training)**

DINOv2 ViT-S/14 frozen backbone + trained linear head. Best accuracy. Requires a trained `.pt` weights file.

| Criterion | EfficientNetV2-S | DINOv2 ViT-S/14 (chosen) | ResNet-50 |
|-----------|------------------|---------------------------|-----------|
| Pre-training data | ImageNet-1K (1.2M) | LVD-142M (142M images) | ImageNet-1K |
| Training approach | Full fine-tune | Frozen backbone + linear head | Full fine-tune |
| Training time (Mac MPS) | 1–2 hours | ~10 minutes | ~1 hour |
| Weight file size | ~80 MB | ~15 KB (head only) | ~90 MB |
| Generalisation to new domains | Moderate | Excellent (self-supervised) | Moderate |
| Per-crop inference | ~5ms | ~15–20ms | ~8ms |

**Why DINOv2:** The core challenge is generalisation. We train on CUB-200 (North American birds) but deploy on equatorial African species. DINOv2 was pre-trained on 142 million images without labels using self-supervised learning — it learns universal visual features that transfer across domains without fine-tuning the backbone. We freeze the entire ViT-S/14 backbone and only train a linear classification head (LayerNorm → Dropout → Linear). This means:

- **Training is fast** (~10 min vs hours) because we only optimise ~3,500 parameters
- **Weight files are tiny** (~15 KB) — easy to version, deploy, and roll back
- **Retraining is cheap** — can retrain weekly on new labelled data without GPU-hours
- **Domain shift is handled** — backbone features generalise to unseen bird species

The tradeoff is ~10ms slower per crop. For camera trap imagery processed every few seconds, this is negligible.

**Mode 2 — HYBRID (no labelled data needed)**

When no trained classifier weights are available, the system falls back to a hybrid approach that requires zero labelled data:

- **Black birds:** Detected via pixel-level analysis using the `BlackBirdDetector`. This analyses the center 50% of each crop (to exclude background) and measures percentile brightness, dark pixel dominance (fraction of pixels with V < 80 in HSV), and RGB channel ratios of the darkest pixels. This is more reliable than CLIP for the black/not-black question because CLIP's cosine similarities for "black bird" vs "brown bird" converge at low brightness.

- **Non-black colours:** Classified using CLIP zero-shot. Each crop is compared against text prompts like "a photo of a white bird", "a photo of a brown bird", etc. CLIP returns the colour prompt with the highest cosine similarity. No training data needed — CLIP's language-vision alignment handles any bird species.

This mode activates automatically when CLIP is available but no trained `.pt` weights file is provided.

**Mode 3 — HEURISTIC (no models needed at all)**

The simplest fallback — pure HSV colour space rules with no neural network:

- Black detection via the same `BlackBirdDetector` pixel analysis
- White: high brightness (V > 210) + low saturation (S < 40)
- Grey: low saturation (S < 35)
- Red/brown/yellow/green/blue: hue angle ranges in HSV space

This mode works offline, on CPU, with zero downloads. It activates when neither trained weights nor CLIP are available.

**Automatic fallback chain:** The classifier tries Neural → Hybrid → Heuristic, using the best mode available. This means the system always works, even with no labelled data and no internet connection for model downloads.

*Implementation: `src/models/color_classifier.py` → `BirdColorClassifier`*

### 3.3 Verification: CLIP ViT-B/32

CLIP acts as a false-positive filter between detection and classification. YOLO26 is trained on COCO's 80 classes and sometimes misclassifies branches, leaves, or shadows as birds — especially in dense forest canopy. CLIP compares each crop against the text prompts "a photo of a bird" vs "a photo of something that is not a bird" and rejects non-bird crops before they reach the colour classifier. Can be disabled with `--no-clip` for faster inference.

### 3.4 Shadow Discriminator

Equatorial forests produce deep canopy shadows that look almost identical to dark-coloured birds in low-resolution crops. The `ShadowDiscriminator` operates in two modes:

**Rule-based mode (default, no training needed)** — analyses each dark crop using 7 features:

1. **Laplacian texture variance** — birds have feather texture producing high-frequency detail; shadows are smooth
2. **Sobel edge density** — birds have well-defined body/beak/feet edges; shadows are diffuse
3. **HSV saturation** — shadows have near-zero saturation; black birds retain subtle iridescence
4. **Aspect ratio** — shadows are amorphous; birds have compact shapes (0.3–3.0 W:H)
5. **Gradient orientation coherence** — bird bodies create structured gradient patterns along contours; shadows have random/uniform gradients (uses structure tensor eigenvalue ratio)
6. **Center-surround contrast** — birds are distinct objects with different brightness from their background; shadows blend smoothly into surroundings
7. **Frequency energy ratio** — birds have mid-frequency content from feather patterns and body structure; shadows are dominated by low-frequency smooth gradients (uses FFT)

A majority vote (≥4 of 7 features indicating shadow) rejects the detection.

*Implementation: `src/models/bird_detector.py` → `ShadowDiscriminator`*

### 3.5 Framework: PyTorch

PyTorch was chosen because:
- **Ultralytics** — YOLO26 is natively PyTorch with mature training and export pipelines
- **torch.hub** — DINOv2 loads directly via `torch.hub.load('facebookresearch/dinov2')`
- **open_clip** — CLIP loads via OpenCLIP, which is PyTorch-native
- **MPS support** — Runs natively on Apple Silicon for local development
- **TensorRT export** — PyTorch → ONNX → TensorRT for production GPU optimisation

---

## 4. Detection, Counting & Classification Logic

### 4.1 Pipeline Flow

```
Input Image
    │
    ▼
┌─────────────────┐
│ PREPROCESS       │  Auto white balance (grey-world algorithm)
│                  │  CLAHE contrast enhancement
│                  │  Letterbox resize to 1280×1280
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DETECT           │  YOLO26x: single-pass NMS-free inference
│                  │  Output: bounding boxes + confidence scores
└────────┬────────┘
         │
         ▼  (for each detection)
┌─────────────────┐
│ VERIFY           │  CLIP ViT-B/32: "is this a bird?"
│                  │  Reject crops scoring < 0.5 for "bird" prompt
└────────┬────────┘
         │
         ▼  (dark crops only)
┌─────────────────┐
│ SHADOW FILTER    │  Laplacian + Sobel + HSV + shape analysis
│                  │  Majority vote: ≥3/4 shadow signals → reject
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CLASSIFY         │  DINOv2 frozen features → linear head
│                  │  Output: colour class + confidence
│                  │  Classes: black, white, brown, green, red,
│                  │           blue, yellow, grey, mixed
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ AGGREGATE        │  Count total birds
│                  │  Count by colour
│                  │  Flag black birds
│                  │  Return FrameResult
└─────────────────┘
```

### 4.2 Pseudocode

```python
def process_frame(image: np.ndarray) -> FrameResult:
    # Step 1: Preprocess
    preprocessed = letterbox_resize(clahe(white_balance(image)), 1280)

    # Step 2: Detect (YOLO26 — NMS-free, single pass)
    detections = yolo26.detect(preprocessed, conf=0.35)

    # Step 3: Verify each crop is actually a bird
    verified = []
    for det in detections:
        crop = extract_crop(preprocessed, det.bbox)
        if clip_verifier.is_bird(crop):  # cosine similarity > 0.5
            verified.append(det)

    # Step 4: Shadow filtering for dark crops
    filtered = []
    for det in verified:
        crop = extract_crop(preprocessed, det.bbox)
        if is_dark(crop) and shadow_discriminator.is_shadow(crop):
            continue  # skip shadows
        filtered.append(det)

    # Step 5: Colour classification (DINOv2)
    for det in filtered:
        crop = extract_crop(preprocessed, det.bbox)
        features = dinov2_backbone(crop)       # frozen, no grad
        det.color = linear_head(features)      # trainable head
        det.color_confidence = softmax_max

    # Step 6: Aggregate
    return FrameResult(
        total_birds=len(filtered),
        black_bird_count=sum(1 for d in filtered if d.color == "black"),
        color_counts=count_by_color(filtered),
        detections=filtered,
    )
```

*Full implementation: `src/inference/inference_engine.py` → `InferenceEngine`*

---

## 5. Data Strategy

### The Problem

No labelled dataset exists for equatorial African bird species. CUB-200 covers North American species. We cannot wait months for expert annotations before deploying. The system must work from Day 1 and improve continuously.

### Four-Phase Solution

**Phase 0 — Zero-Shot Bootstrap**

Deploy immediately using foundation models that generalise without training:
- **YOLO26:** COCO-pretrained — bird silhouettes are universal across geographies
- **DINOv2:** 142M-image self-supervised features — domain-agnostic visual understanding
- **CLIP:** Language-based verification ("a photo of a bird") — works on any bird species

This gives a working system from the first camera frame with reasonable accuracy.

**Phase 1 — Transfer Learning from Public Datasets**

Train on progressively larger public datasets:

| Dataset | Images | Species | Value |
|---------|--------|---------|-------|
| CUB-200-2011 | 11,788 | 200 | Fine-grained benchmark, quick to train |
| NABirds | 48,562 | 555 | Excellent bounding boxes |
| iNaturalist-Aves | 214,295 | 1,486 | Global diversity, closest to target domain |
| BirdSnap | 49,829 | 500 | Good colour diversity |

All datasets are normalised to COCO format via `DatasetConverter`. With DINOv2's frozen backbone, adding more training data only retrains the linear head — minutes, not hours.

**Phase 2 — Active Learning Loop (To be implemented)**

**Phase 3 — Synthetic Data Augmentation (To be implemented)**


### Ideal Infrastructure for Data Management

| Component | Tool | Purpose |
|-----------|------|---------|
| Raw images | S3 | Archive with lifecycle policies (hot → warm → cold) |
| Metadata & results | SQLite (PoC) / PostgreSQL (prod) | Structured queries, dashboard backing |
| Annotations | Label Studio (Docker) | Pre-filled suggestions, ornithologist review |
| Dataset versioning | DVC | Track data alongside code |
| Experiment tracking | MLflow | Model versions, metrics, artefacts |
| Model weights | S3 | Versioned storage with promotion stages |

---

## 6. Containerisation & Deployment

### Strategy

The deployment follows a standard containerised microservices pattern. The approach would be:

**Docker Build (multi-stage):** A two-stage Dockerfile — the first stage installs all Python dependencies on an NVIDIA CUDA 12.2 base image, the second stage copies only the installed packages and application code into a minimal production image. This reduces the final image size and attack surface. The application runs as a non-root user.

**Docker Compose for the full stack:** A single `docker-compose.yaml` orchestrates all services:
- `api` — FastAPI inference service with GPU access via NVIDIA Container Toolkit
- `redis` — Message queue for camera events
- `postgres` — Detection results and metadata store
- `prometheus` + `grafana` — Metrics collection and alerting dashboards
- `label-studio` — Annotation interface for the active learning loop

All model weights are stored in S3 and downloaded at container startup. Environment variables control which models load (`MODEL_DETECTOR_PATH`, `MODEL_COLOR_PATH`, `USE_CLIP_VERIFIER`, `DRIFT_WINDOW`), making it easy to swap models without rebuilding the image.

**Production (Kubernetes):** For production at scale, the deployment moves to Kubernetes with GPU node pools for inference workers (auto-scaling based on queue depth), CPU node pools for API gateway, database, and monitoring, Horizontal Pod Autoscaler targeting queue backlog < 100 images, and rolling deployments with health check gates.

---

## 7. Scaling to 1000s of Cameras

One API server processes images sequentially at ~6 images/second (~400 cameras at 1 frame/minute). To scale to thousands, decouple ingestion from inference:

```
                                    ┌──────────────────┐
                                    │  GPU Worker 1     │
                                    │  YOLO26 + DINOv2  │
                               ┌───▶│  ~400 cameras     │───┐
                               │    └──────────────────┘   │
┌─────────┐   ┌────────────┐   │    ┌──────────────────┐   │   ┌──────────┐
│ Cameras │──▶│ API Gateway│──▶├───▶│  GPU Worker 2     │───├──▶│PostgreSQL│
│ (1000s) │   │ (stateless)│   │    │  ~400 cameras     │   │   │ + S3     │
└─────────┘   └─────┬──────┘   │    └──────────────────┘   │   └────┬─────┘
                    │          │    ┌──────────────────┐   │        │
                    ▼          └───▶│  GPU Worker N     │───┘        ▼
              ┌──────────┐          │  ~400 cameras     │      ┌──────────┐
              │  Redis    │          └──────────────────┘      │Dashboard │
              │  Queue    │                                     └──────────┘
              └──────────┘
```

**What changes in the code:**
- `/detect` becomes async: enqueue image → return job ID
- New `/results/{job_id}` endpoint to fetch results
- Workers are copies of `InferenceEngine` pulling from Redis
- Drift detection, dashboard, retrain pipeline read from PostgreSQL

**Scaling math:**
- 1 GPU worker ≈ 400 cameras (1 frame/min)
- 3 workers = 1,200 cameras
- 10 workers = 4,000 cameras

Scale horizontally by adding GPU machines. Everything else stays the same.

---

## 8. Model Drift Tracking

### Why Drift Matters Here

Equatorial forests change seasonally:
- **Wet season:** Dense green canopy, high humidity, fog on lens, deeper shadows
- **Dry season:** Sparser vegetation, brighter images, different bird species present
- **Camera degradation:** Lens fogging, solar panel dust, sensor aging

These changes shift the image distribution away from what the model was trained on, causing silent accuracy degradation.

### How We Track It (No Ground Truth Required)

The `DriftDetector` monitors three signals from raw inputs and model outputs alone:

**1. Data Drift — Population Stability Index (PSI)**

Compares current image statistics against a reference baseline:
- Green dominance ratio (G / (R+G+B)) — drops during dry season
- Mean brightness — increases when canopy thins
- Edge density — proxy for vegetation density
- Per-channel RGB means and standard deviations

A PSI > 0.20 on any feature triggers a warning.

**2. Prediction Drift — Jensen-Shannon Divergence**

Compares the distribution of predicted colour classes between the current window and the reference. If the model suddenly predicts 40% brown (was 15%), something changed — either the bird population shifted or the model is confused.

**3. Confidence Drift — Mean Confidence Drop**

Tracks the average detection confidence over time. A sustained drop indicates the model is seeing inputs it wasn't trained for.

### Severity Levels

| Level | Condition | Action |
|-------|-----------|--------|
| None | All metrics within thresholds | Continue |
| Warning | 1–2 features drifted | Log + Slack alert |
| Critical | 3+ features drifted | Trigger retrain pipeline |

The drift window is configurable via `DRIFT_WINDOW` environment variable (default: 100 images).

*Implementation: `src/monitoring/drift_detector.py`*

---

## 9. Retraining, Evaluation & Deployment Pipeline

### Trigger

Retraining is triggered by `scripts/check_retrain_needed.py` when any of:
- Drift severity = critical
- Overall accuracy drops > 5% below baseline
- Black bird F1 < 0.70
- Manual trigger via GitHub Actions

### Training

```bash
python3 train_color_classifier.py \
    --dataset-dir ./CUB_200_2011 \
    --epochs 15 --batch-size 32 \
    --output color_classifier_candidate.pt
```

DINOv2 backbone is frozen. Only the linear head trains (~3,500 parameters). Training completes in ~10 minutes on Mac MPS, ~3 minutes on CUDA GPU.

### Evaluation

```bash
python3 evaluate_cub200.py \
    --dataset-dir ./CUB_200_2011 \
    --max-images 500 --seed 42 \
    --color-model color_classifier_candidate.pt \
    --output-json metrics/candidate.json
```

Evaluates on 500 held-out test images across 4 random seeds for variance estimation.

### Promotion (safety gate)

The candidate model is promoted only if it beats the current production model:
- Overall accuracy ≥ current
- Black bird F1 ≥ current
- No class regresses more than 5%

### CI/CD Pipeline Strategy (GitHub Actions)

The retraining pipeline would be automated via GitHub Actions on a weekly schedule:

1. **Download** latest labelled data from S3 (corrections from Label Studio)
2. **Download** current production weights
3. **Train** candidate model (DINOv2 head only, ~10 min on GPU runner)
4. **Evaluate** candidate vs production on held-out test set
5. **Promote** if candidate wins → upload to S3, archive old weights
6. **Rolling restart** of API containers to pick up new weights

The broader CI/CD strategy would also include: linting, type-checking, and unit tests on every push; Docker image build and push on merge to main; staged deployment (staging first, production requires manual approval).

---

## 10. Performance Metrics

### Detection

| Metric | Definition | Target |
|--------|-----------|--------|
| mAP@50 | Mean Average Precision at IoU ≥ 0.50 | ≥ 0.65 |
| Precision | TP / (TP + FP) | ≥ 0.75 |
| Recall | TP / (TP + FN) | ≥ 0.70 |
| Counting MAE | Mean absolute error of bird count per frame | ≤ 1.5 |

### Classification

| Metric | Definition | Target |
|--------|-----------|--------|
| Black Bird Precision | TP_black / (TP_black + FP_black) | ≥ 0.85 |
| Black Bird Recall | TP_black / (TP_black + FN_black) | ≥ 0.75 |
| Black Bird F1 | Harmonic mean | ≥ 0.80 |
| Shadow FPR | Shadows misclassified as black | ≤ 0.05 |
| Overall Accuracy | Correct colour / total | ≥ 0.70 |

### System

| Metric | Target | Monitoring |
|--------|--------|-----------|
| Inference latency p95 | < 500ms per image | Prometheus |
| API uptime | > 99.5% | Health check |
| Throughput | > 500 images/hour per GPU | Grafana |

---

## 11. Project Structure

```
aero_watch/
├── src/
│   ├── models/
│   │   ├── bird_detector.py           # YOLO26 detector + shadow discriminator
│   │   └── color_classifier.py        # DINOv2 ViT-S/14 colour classifier
│   ├── inference/
│   │   └── inference_engine.py        # End-to-end pipeline orchestration
│   ├── data/
│   │   ├── data_pipeline.py           # Image preprocessing (white bal, CLAHE, resize)
│   │   └── cold_start_strategy.py     # Transfer learning, active learning, synthetic data
│   ├── monitoring/
│   │   └── drift_detector.py          # PSI, JS divergence, confidence tracking
│   ├── api/
│   │   └── app.py                     # FastAPI REST API with CORS + drift endpoints
│   ├── db.py                          # SQLite persistence (detections, drift state)
│   └── utils/
│       └── device.py                  # Auto-detect MPS / CUDA / CPU
├── scripts/
│   └── check_retrain_needed.py        # Drift + metric gate for retrain decisions
├── tests/
│   └── test_pipeline.py               # Unit & integration tests
├── run_pipeline.py                    # CLI: batch inference on folder of images
├── train_color_classifier.py          # Train DINOv2 colour classifier head
├── evaluate_cub200.py                 # Evaluate model on held-out test set
├── simulate_deployment.py             # End-to-end simulation (deploy → feed → observe)
├── dashboard.html                     # Offline results dashboard (open in browser)
├── requirements.txt                   # Python dependencies
├── MAC_SETUP_GUIDE.md                 # Step-by-step Mac setup instructions
└── README.md                          # This document
```

---

## 12. Quick Start

```bash
# 1. Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download the trained colour classifier
#    https://drive.google.com/file/d/1zyKomCduhhtVmWmaryxHCnRDh9sYTjaU/view?usp=sharing
#    Save as color_classifier_best.pt in the project root

# 3. Run pipeline on any folder of images
python3 run_pipeline.py -i ./my_bird_photos/ \
    --color-model color_classifier_best.pt \
    --output-json pipeline_results.json

# 4. View results in dashboard
open dashboard.html
# Drop pipeline_results.json onto the dashboard

# 5. Or run the full end-to-end simulation
python3 simulate_deployment.py --images ./my_bird_photos/ \
    --color-model color_classifier_best.pt \
    --num-images 50

# 6. (Optional) Download CUB-200 dataset for training/evaluation
wget https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz

# 7. Train the colour classifier
python3 train_color_classifier.py --dataset-dir ./CUB_200_2011 --epochs 15

# 8. Evaluate
python3 evaluate_cub200.py --dataset-dir ./CUB_200_2011 \
    --color-model color_classifier_best.pt
```

See `MAC_SETUP_GUIDE.md` for detailed setup instructions.

---
