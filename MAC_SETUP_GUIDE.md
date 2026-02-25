# Aero-Watch — Mac Setup Guide

## Step-by-step instructions to run the full pipeline on macOS

---

## Prerequisites

- **macOS 13+ (Ventura)** or later
- **Python 3.10+** (3.11–3.14 all work)
- **~5 GB disk space** for YOLO26x + DINOv2 + CLIP weights + dependencies
- **8+ GB RAM** (16 GB recommended)
- A folder of bird images (any JPEG/PNG photos)

---

## Step 1 — Install Python (if needed)

```bash
python3 --version

# If you need to install/update:
brew install python@3.11
```

---

## Step 2 — Extract the Project

```bash
cd ~/Downloads
unzip aero_watch_full_pipeline.zip
cd aero_watch_full_pipeline
```

---

## Step 3 — Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate

# Verify
which python
# Should show: .../aero_watch_full_pipeline/venv/bin/python
```

---

## Step 4 — Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

**Expected time:** 2–5 minutes.

**Verify PyTorch + MPS (Apple Silicon):**

```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
"
```

You should see `MPS available: True` on Apple Silicon Macs (M1/M2/M3/M4).

---

## Step 5 — Download the Trained Colour Classifier

Download the pre-trained DINOv2 colour classifier weights:

**[color_classifier_best.pt](https://drive.google.com/file/d/1zyKomCduhhtVmWmaryxHCnRDh9sYTjaU/view?usp=sharing)**

Save the file as `color_classifier_best.pt` in the project root directory.

This step is optional — without it, the system falls back to Hybrid mode (pixel-based black detection + CLIP zero-shot for other colours). With it, you get the best accuracy from the trained DINOv2 classifier.

---

## Step 6 — Run Pipeline on Any Images

### Option A — Quick test (small model, no extras)

```bash
python3 run_pipeline.py -i /path/to/bird_photos/ \
    --model yolo26n.pt --no-clip
```

Uses YOLO26-Nano (~6 MB download). Good for verifying everything works.

### Option B — Full pipeline with trained classifier

```bash
python3 run_pipeline.py -i /path/to/bird_photos/ \
    --model yolo26x.pt \
    --color-model color_classifier_best.pt \
    --output-json pipeline_results.json
```

Uses YOLO26x + DINOv2 colour classifier + CLIP verifier. Best accuracy.

### Option C — Single image

```bash
python3 run_pipeline.py -i photo.jpg \
    --color-model color_classifier_best.pt
```

### What happens on first run

1. **YOLO26 weights** auto-download from Ultralytics (~130 MB for yolo26x)
2. **DINOv2 backbone** downloads from PyTorch Hub (~85 MB, cached at `~/.cache/torch/hub/`)
3. **CLIP model** downloads from Hugging Face (~340 MB, cached at `~/.cache/huggingface/`)
4. Subsequent runs use cached weights — no re-download

### Expected output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📊  RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Images processed:  49
  Total birds:       43
  Black birds:       5
  Time:              58.2s (0.8 img/s)

  🎨 Birds by colour:
      mixed:  12  ████████████
      brown:   9  █████████
      white:   7  ███████
      black:   5  █████
      green:   4  ████
       grey:   3  ███
       blue:   2  ██
     yellow:   1  █
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Step 7 — View Results in Dashboard

```bash
# If you used --output-json in Step 5:
open dashboard.html
```

This opens an HTML file in your browser. Drag and drop `pipeline_results.json` onto it. The dashboard shows:
- Bird count stats (total, black, images processed, processing speed)
- Colour bar chart
- Per-image bird count timeline
- Colour breakdown with horizontal bars
- Full detection table with confidence scores

No server, no Node.js, no React required — just a browser.

---

## Step 8 — Run Full End-to-End Simulation

This starts the API server, feeds images through it, checks drift, and opens the dashboard — all in one command:

```bash
python3 simulate_deployment.py \
    --images /path/to/bird_photos/ \
    --color-model color_classifier_best.pt \
    --detector yolo26x.pt \
    --num-images 50
```

**What happens (5 stages):**

| Stage | What It Does |
|-------|-------------|
| 1. DEPLOY | Starts FastAPI server on port 8000 |
| 2. SIMULATE | Feeds images to `/detect` as camera frames |
| 3. OBSERVE | Queries `/health`, `/drift`, `/stats` |
| 4. DASHBOARD | Saves `pipeline_results.json`, opens `dashboard.html` |
| 5. RETRAIN CHECK | Evaluates drift report, decides if retraining needed |

The server shuts down automatically when done.

### Lower the drift window for small image sets

By default, drift detection needs 100 images to set a baseline. If you have fewer images:

```bash
DRIFT_WINDOW=20 python3 simulate_deployment.py \
    --images /path/to/bird_photos/ \
    --color-model color_classifier_best.pt \
    --num-images 49
```

---

## Step 9 — Run the API Server Manually

**Terminal 1 — Start server:**

```bash
source venv/bin/activate
MODEL_COLOR_PATH=color_classifier_best.pt \
MODEL_DETECTOR_PATH=yolo26x.pt \
USE_CLIP_VERIFIER=true \
    python3 -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete`.

**Terminal 2 — Send images:**

```bash
# Single image
curl -X POST "http://127.0.0.1:8000/detect?camera_id=CAM-01&sector=A" \
    -F "image=@photo.jpg" | python3 -m json.tool

# Batch
for img in ./photos/*.jpg; do
    curl -s -X POST "http://127.0.0.1:8000/detect?camera_id=CAM-01&sector=A" \
        -F "image=@$img" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'{d[\"total_birds\"]} birds, {d[\"black_bird_count\"]} black')
"
done
```

**API endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status, device, uptime |
| `/detect` | POST | Upload image, get detection results |
| `/drift` | GET | Current drift report |
| `/stats` | GET | Aggregated stats (total birds, colours, latency) |
| `/history` | GET | Recent detection results (for live dashboard) |
| `/metrics` | GET | Prometheus-format metrics |
| `/reset` | POST | Clear database for fresh simulation run |
| `/docs` | GET | Interactive Swagger UI |

---

## Step 10 — Train the Colour Classifier

Requires the CUB-200-2011 dataset:

```bash
# Download and extract CUB-200
wget https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz

# Train
python3 train_color_classifier.py \
    --dataset-dir ./CUB_200_2011 \
    --epochs 15 \
    --batch-size 32 \
    --output color_classifier_best.pt
```

Training uses DINOv2 ViT-S/14 with a frozen backbone. Only the linear head is trained (~3,500 parameters). Takes ~10 minutes on Mac MPS.

---

## Step 11 — Evaluate

```bash
python3 evaluate_cub200.py \
    --dataset-dir ./CUB_200_2011 \
    --max-images 500 \
    --seed 42 \
    --color-model color_classifier_best.pt \
    --output-json metrics/baseline.json
```

---

## Step 12 — Run Tests

```bash
pytest tests/test_pipeline.py -v
```

---

## Model Size & Speed Reference

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolo26n.pt | ~6 MB | ~30ms | Low |
| yolo26s.pt | ~22 MB | ~60ms | Medium |
| yolo26m.pt | ~50 MB | ~120ms | Good |
| yolo26x.pt | ~130 MB | ~300ms | Highest |

Speeds approximate for Apple M2 Pro.

**Recommendation:** Use `yolo26n.pt --no-clip` for fast iteration, `yolo26x.pt` for final results.

---

## Troubleshooting

### "No module named 'src'"

Run from the project root directory:
```bash
cd ~/Downloads/aero_watch_full_pipeline
python3 run_pipeline.py -i ...
```

### xFormers warnings from DINOv2

```
UserWarning: xFormers is not available (SwiGLU)
UserWarning: xFormers is not available (Attention)
```

These are harmless. DINOv2 uses standard PyTorch attention as fallback. No action needed.

### "MPS available: False" on Apple Silicon

```bash
pip install --upgrade torch torchvision
```

### HuggingFace token warning

```
Warning: You are sending unauthenticated requests to the HF Hub
```

Harmless. CLIP weights are public. For faster downloads: `huggingface-cli login`.

### No birds detected

1. Lower the confidence threshold: `--conf 0.10`
2. Try `--all-classes` to see if the model detects anything
3. Make sure birds are actually visible in the image

### Drift status shows "pending"

The drift detector needs `DRIFT_WINDOW` images (default 100) to set a baseline. Either send more images or lower the window:

```bash
DRIFT_WINDOW=20 python3 simulate_deployment.py --images ./photos/ --num-images 49
```

### Out of memory

```bash
python3 run_pipeline.py -i photo.jpg --model yolo26n.pt --no-clip --input-size 640
```

---

## What Each File Does

| File | Purpose |
|------|---------|
| `run_pipeline.py` | **Main entry point** — batch inference on images, save JSON results |
| `simulate_deployment.py` | **End-to-end demo** — start API, feed images, check drift, open dashboard |
| `train_color_classifier.py` | Train DINOv2 linear head on CUB-200 |
| `evaluate_cub200.py` | Evaluate model on held-out test set |
| `dashboard.html` | **Offline dashboard** — open in browser, drop JSON results |
| `src/models/bird_detector.py` | YOLO26 detector + shadow discriminator |
| `src/models/color_classifier.py` | DINOv2 ViT-S/14 colour classifier |
| `src/inference/inference_engine.py` | Orchestrates detect → verify → classify → count |
| `src/data/data_pipeline.py` | Image preprocessing (white balance, CLAHE, resize) |
| `src/data/cold_start_strategy.py` | Transfer learning, active learning, synthetic data |
| `src/monitoring/drift_detector.py` | PSI, JS divergence, confidence tracking |
| `src/api/app.py` | FastAPI REST API (detect, drift, stats, history) |
| `src/db.py` | SQLite persistence (detections, drift state) |
| `scripts/check_retrain_needed.py` | Drift + metric gate for retrain decisions |
