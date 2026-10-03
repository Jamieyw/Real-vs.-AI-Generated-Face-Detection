# 🕵️ Real or AI? Human vs. Model

An interactive game where you compete against a deep learning model to tell **real face photos** from **AI-generated faces**. Built for CS 6180 (Homework 1) with TensorFlow/Keras and Gradio, deployed on Render.

**Live demo:** https://<your-service>.onrender.com

> The app runs on Render's free tier, which sleeps after ~15 minutes of inactivity. The first visit may take about a minute to wake up; after that it responds instantly.

## How to play
1. A face from the held-out test set appears.
2. Click **Real** or **AI-generated**.
3. The model reveals its prediction and confidence, and both scores update.
4. After 10 rounds, the app shows who won. Click **Play again** for a new set of faces.

## Model
- **Backbone:** MobileNetV3Small pretrained on ImageNet (128×128 RGB input)
- **Head:** GlobalAveragePooling → Dense(64, ReLU) → Dropout(0.5) → Dense(1, sigmoid)
- **Training:** frozen-backbone transfer learning with data augmentation, then fine-tuning of the last 3 blocks (Adam 1e-5, BatchNorm frozen, label smoothing 0.1)
- **Test accuracy:** 74.0% (frozen MobileNet: 72.7%, CNN from scratch: 67.0%)

## Files
| File | Purpose |
|---|---|
| `app.py` | Gradio game |
| `best_model.keras` | Trained fine-tuned MobileNetV3Small |
| `test_images/` | Test-set faces (lossless PNG, 128×128) |
| `labels.json` | Ground-truth labels (1 = Real, 0 = AI), read server-side only |
| `requirements.txt` | Pinned dependencies (match the training environment) |
| `render.yaml` | Render deployment config |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.13
pip install -r requirements.txt
python app.py                                        # http://localhost:7860
```

## Deployment (Render)
- Web Service, Python runtime, free plan
- Build: `pip install -r requirements.txt`
- Start: `python app.py` (binds to `0.0.0.0:$PORT`)
- Env var: `PYTHON_VERSION=3.13.15`
