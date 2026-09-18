# GourNetV2 Mango Leaf Disease Detector

A Streamlit application for side-by-side comparison of two convolutional neural network
classifiers for mango leaf disease detection: the baseline **GourNet** and the enhanced
**GourNet v2**. Both models run on the same uploaded leaf image, so predictions, confidence
scores, and inference latency are directly comparable on a single screen.

Developed as a thesis prototype in connection with Alam et al. (2026),
"GourNet: A CNN-Based Model for Mango Leaf Disease Detection."

## Authors

- DM Rashid P. Ferrer
- Lean Adrian B. Murillo
- James Oliver C. Mendoza

## Features

- **Side-by-side comparison** — GourNet and GourNet v2 predict on the identical
  preprocessed input, with agreement/disagreement verdict, per-model latency, prediction
  margin, and entropy.
- **Two weight variants** — switch between Standard and 12k training weights for both
  architectures from the sidebar.
- **Probability analysis** — grouped per-class probability chart, v2-vs-baseline delta
  table, and top-k ranked class lists.
- **Batch benchmark** — upload multiple images, assign true labels, and compare agreement
  rate, per-model accuracy, and mean latency across the batch.

## Disease Classes

Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Healthy,
Powdery Mildew, Sooty Mould.

## Models

| Model | Architecture | Parameters |
|---|---|---|
| GourNet (baseline) | 4-conv-block CNN, faithful to the paper | 683,656 |
| GourNet v2 (enhanced) | GroupNorm + Global Average Pooling variant | 98,280 |

Each architecture ships with Standard and 12k trained weights (see `weights/`).

## Project Structure

```
├── app.py               # Streamlit UI (entry point)
├── models/
│   ├── __init__.py
│   ├── gournet.py       # Baseline architecture (build_gournet)
│   └── gournet_v2.py    # Enhanced architecture (build_gournet_v2)
├── weights/
│   ├── gournet_model.keras
│   ├── gournet_v2_model.keras
│   ├── gournet_model_12k.keras
│   └── gournet_v2_model_12k.keras
├── requirements.txt
├── runtime.txt
├── .python-version
└── README.md
```

## Setup

```bash
cd mango-gournet-streamlit
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub, including the `weights/*.keras` files.
2. Create a new app pointing at branch `main`, main file `app.py`.
3. Set the Python version to **3.12** in the app settings
   (`.python-version` already pins it; TensorFlow provides no Python 3.14 wheels).
4. Reboot after pushing dependency or version changes.

## Notes and Limitations

- Image preprocessing resizes to 224×224 and relies on the model's built-in
  `Rescaling(1/255)` layer, matching the training notebooks.
- Single-image confidence, margin, and latency demonstrate live behavior but do not
  rank the models; formal comparison (accuracy, macro F1, confusion matrix, calibration)
  is evaluated offline on a held-out test set.
- No authentication, persistence, or prediction logging is included.
