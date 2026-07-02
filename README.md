# Spot the Fake Photo — Screen Recapture Detector

Given a single image, predicts whether it's a **real photo (0)** or a
**photo of a screen/printout showing another image (1)** — i.e. a
"recapture."

```
python predict.py some_image.jpg
0.93
```

## Approach (short version)

No deep learning, no pretrained vision backbone. The pipeline extracts
~35 classical, physically-motivated features per image using OpenCV /
scikit-image / scipy — frequency-domain (FFT/moire/aliasing), texture
(LBP, wavelet energy), noise statistics, glare/specular/reflection
heuristics, sharpness, and color-moment features — then feeds them into
whichever of (Logistic Regression, small Random Forest) cross-validates
best. See `report.md` for the full write-up, honest accuracy numbers,
and the two required metrics (latency, cost-per-image).

**Training-only data augmentation**: each training photo also
contributes 4 augmented variants (small rotation, brightness/contrast
jitter, crop/zoom, saturation jitter, JPEG re-compression) to make the
classifier robust to minor shooting-condition differences. This is
leakage-safe — augmented copies of a photo are grouped so they can never
end up split across train/validation/test (see `train.py`'s
`GroupKFold` usage). Validation and the final held-out test set always
use single, un-augmented images, so the reported accuracy is honest and
reflects exactly what `predict.py` sees at inference time.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

```
dataset/
├── real/       # 84 photos (label 0)
└── screen/     # 51 photos (label 1)
```

Self-collected: real-world photos (including deliberate "decoys" —
glass, glossy floors, window grilles, a screen turned off) and
screen/laptop recaptures across varied lighting and angles. See
`report.md` for the full write-up.

## Train

```bash
python train.py
```

Prints 5-fold CV metrics per model, the auto-selected best model, tuned
decision threshold, final held-out test metrics (Accuracy / Precision /
Recall / F1 / ROC-AUC / Confusion Matrix), and latency/memory numbers.
Saves the model bundle to `model/best_model.joblib`.

## Predict

```bash
python predict.py path/to/image.jpg
```

Prints one float in `[0, 1]` to stdout (0 = real, 1 = screen). Logs and
errors go to stderr, so stdout is always machine-clean. Non-zero exit
code on failure (bad path, corrupt file, etc.).

## Live demo (optional)

```bash
python app.py
```

Then open **http://localhost:5000** and allow camera access. Point
your camera at something (a real object, or a screen showing another
photo) and the page continuously re-analyzes the live feed - the
targeting brackets and readout update every ~900ms with the live
probability.

This is a thin wrapper, not a separate implementation: `app.py`
imports `extract_features` from `utils.py` and loads the exact same
`model/best_model.joblib` bundle that `train.py` produces and
`predict.py` uses - a browser-facing view of the same pipeline, not a
different one. Runs entirely on localhost; no frame is ever written to
disk.

## Files

```
predict.py         # required one-line CLI predictor
train.py            # trains + evaluates + saves the model
utils.py             # feature extraction (the actual "secret sauce")
app.py                # optional live-demo backend (reuses utils.py + the saved model)
templates/index.html  # optional live-demo frontend (camera capture + live readout)
requirements.txt
report.md            # the half-page note: approach, accuracy, latency, cost, next steps
model/best_model.joblib   # produced by train.py
dataset/real, dataset/screen  # your photos go here
```
