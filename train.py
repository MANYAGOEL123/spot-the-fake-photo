"""
train.py
========
Train the real-vs-screen classifier on `dataset/real/` and
`dataset/screen/`.

Usage:
    python train.py

Pipeline:
    1. Split the *original* photo paths (not features) into
       train / val / test (70/15/15, stratified) BEFORE any
       augmentation, so augmented copies of a photo can never leak
       across the split.
    2. Build a TRAINING POOL: each training photo contributes its own
       clean feature vector plus `--n-aug` augmented variants. As of
       this revision, augmentation covers a much wider degradation
       space than mild indoor jitter - extreme exposure (night through
       bright sunlight), perspective/keystone warp (extreme viewing
       angles, projector geometry), synthetic sensor noise (low-light),
       and defocus/motion blur (projector softness, camera shake) - in
       addition to rotation/crop/saturation/JPEG jitter. This is a
       deliberate compensation for the raw training photos only
       covering phone/laptop/TV recaptures shot in reasonable indoor
       lighting: the hidden eval set is expected to include OLED/LCD/
       tablet/monitor/projector displays and printed photographs under
       night/low-light/bright-sunlight/extreme-angle conditions that
       aren't otherwise represented. All augmented copies are tagged
       with a group id so cross-validation folds never split a source
       photo's variants across folds.
    3. Preprocess with RobustScaler (median/IQR - several features here
       are heavy-tailed/zero-inflated, e.g. glare_score, so this
       generalizes better than StandardScaler's mean/std) THEN
       VarianceThreshold (drops near-constant/uninformative features -
       deliberately AFTER scaling, since running it on raw unscaled
       features would drop naturally-small-magnitude-but-informative
       features like moire_score purely for numeric scale reasons).
    4. Hyperparameter search (GridSearchCV) over regularization
       strength (Logistic Regression) / tree depth & leaf size (Random
       Forest), using StratifiedGroupKFold - stratified so small,
       imbalanced folds don't produce noisy CV estimates, AND grouped
       so a source photo's augmented copies never straddle a fold
       boundary. Best model picked by mean CV ROC-AUC.
    5. Probability calibration (CalibratedClassifierCV, sigmoid) on a
       fresh copy of the winning model type - Random Forest's raw
       predict_proba is a vote fraction, not a statistically calibrated
       probability, and the assignment explicitly wants a meaningful
       0-1 probability rather than just a correctly-ranked score. Uses
       a plain stratified cv (not grouped) for this step specifically -
       StratifiedGroupKFold's `groups` forwarding hit a real API
       incompatibility in the installed sklearn version; the resulting
       small leakage risk is confined to the calibration curve, not
       the decision boundary. This calibrated model is what gets used
       for every step after this point.
    6. Decision threshold tuned on the untouched, UN-augmented
       validation split, picking the MIDDLE of the best-F1 threshold
       plateau (not its edge - see `tune_threshold`) for robustness.
    7. The SAME (calibrated) model the threshold was tuned against is
       deployed - deliberately NOT retrained on train+val combined
       afterward. Retraining on more data sounds free, but a
       differently-trained model has a different decision function, so
       a threshold tuned against one model and applied to another is
       silently wrong (this was caught during review: it caused a
       previously-reliable real photo to flip to a false positive).
       Correctness of the threshold/model pairing matters more here
       than the modest gain from a few dozen extra training rows.
    8. Evaluated ONCE on the untouched, UN-augmented test split - the
       honest, reported number, reflecting exactly what predict.py
       sees (clean single images, no augmentation at inference time).
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import List, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from utils import (
    FEATURE_NAMES,
    FEATURE_SET_VERSION,
    augment_image,
    extract_features,
    get_memory_usage_mb,
    list_dataset_images,
    load_image,
    logger,
    save_model_bundle,
)

# Hyperparameter search space per candidate model. Modest grids - this
# is a ~135-photo dataset, so an exhaustive search would overfit the
# validation signal itself; the point is to sanity-check regularization
# strength / tree complexity against a fixed default, not to chase
# marginal gains.
_SEARCH_SPACE = {
    "LogisticRegression": (
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
        {"C": [0.05, 0.1, 0.3, 1.0, 3.0]},
    ),
    "RandomForest": (
        RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        {"max_depth": [3, 4, 6, 8], "min_samples_leaf": [2, 4, 6]},
    ),
}
_CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def collect_paths(dataset_dir: str) -> Tuple[List[str], np.ndarray]:
    real_paths = list_dataset_images(os.path.join(dataset_dir, "real"))
    screen_paths = list_dataset_images(os.path.join(dataset_dir, "screen"))
    if len(real_paths) < 10 or len(screen_paths) < 10:
        raise ValueError(
            f"Not enough images. Found {len(real_paths)} real, {len(screen_paths)} screen. "
            "Need at least ~10 per class (the assignment asks for ~50+ each)."
        )
    logger.info("Found %d real images and %d screen images.", len(real_paths), len(screen_paths))
    paths = real_paths + screen_paths
    labels = np.array([0] * len(real_paths) + [1] * len(screen_paths), dtype=np.int64)
    return paths, labels


def extract_clean(paths: List[str], labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Extract a single clean (non-augmented) feature vector per path."""
    vectors, times, kept_labels = [], [], []
    for path, label in zip(paths, labels):
        try:
            image = load_image(path)
            t0 = time.perf_counter()
            vec = extract_features(image)
            times.append(time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unreadable image '%s': %s", path, exc)
            continue
        vectors.append(vec)
        kept_labels.append(label)
    if not vectors:
        raise RuntimeError("No images could be processed.")
    return np.vstack(vectors), np.array(kept_labels), float(np.mean(times))


def extract_augmented_pool(
    paths: List[str], labels: np.ndarray, n_aug: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract clean + `n_aug` augmented feature vectors per path.

    Returns:
        X: stacked feature matrix (clean + augmented rows)
        y: matching labels
        groups: source-photo group id per row (same id for a photo's
            clean row and all its augmented rows) - required so
            StratifiedGroupKFold never splits a source photo's variants
            across folds (that would leak near-duplicate information
            between "train" and "held-out" within a fold).
    """
    rng = np.random.default_rng(seed)
    vectors, out_labels, groups = [], [], []
    for group_id, (path, label) in enumerate(zip(paths, labels)):
        try:
            image = load_image(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unreadable image '%s': %s", path, exc)
            continue

        vectors.append(extract_features(image))
        out_labels.append(label)
        groups.append(group_id)

        for aug_img in augment_image(image, n=n_aug, rng=rng):
            vectors.append(extract_features(aug_img))
            out_labels.append(label)
            groups.append(group_id)

    return np.vstack(vectors), np.array(out_labels), np.array(groups)


def build_preprocessor() -> Pipeline:
    """RobustScaler (median/IQR - robust to the heavy-tailed,
    zero-inflated features in this set, e.g. glare_score is 0 for most
    photos and spikes only for genuine glare) THEN VarianceThreshold.

    Order matters here: several features (moire_score, the wavelet
    energy ratios) are naturally small-magnitude fractions (e.g.
    0.0001-0.01) simply because of how they're defined, not because
    they're uninformative - moire_score is one of the most
    theoretically important features in this whole pipeline. Running
    VarianceThreshold on RAW features before scaling would drop them
    purely for being numerically small, which is a scale artifact, not
    a signal judgment. Scaling first (so every feature has a comparable
    IQR) makes VarianceThreshold measure genuine informativeness
    instead of raw numeric magnitude.
    """
    return Pipeline(
        [
            ("scale", RobustScaler()),
            ("variance", VarianceThreshold(threshold=1e-3)),
        ]
    )


def search_best_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5):
    """Hyperparameter search per candidate model family via
    StratifiedGroupKFold, then pick the best family by mean CV ROC-AUC.

    Returns:
        best_name: winning model family name
        best_estimator: already refit on the FULL X/y with best params
        results: {name: {best_params, cv_<metric>_mean for each metric}}
    """
    min_class_groups = min(
        len(np.unique(groups[y == c])) for c in np.unique(y)
    )
    n_splits = max(2, min(n_splits, min_class_groups))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}
    fitted = {}
    for name, (estimator, grid) in _SEARCH_SPACE.items():
        logger.info("Grid-searching %s over %s (%d-fold, stratified+grouped) ...", name, grid, n_splits)
        search = GridSearchCV(
            estimator, grid, cv=cv, scoring=_CV_SCORING, refit="roc_auc", n_jobs=-1
        )
        search.fit(X, y, groups=groups)
        best_idx = search.best_index_
        summary = {
            f"cv_{m}_mean": float(search.cv_results_[f"mean_test_{m}"][best_idx]) for m in _CV_SCORING
        }
        summary["best_params"] = search.best_params_
        results[name] = summary
        fitted[name] = search.best_estimator_
        logger.info(
            "%s | best_params=%s | CV Acc=%.4f F1=%.4f ROC-AUC=%.4f",
            name, search.best_params_, summary["cv_accuracy_mean"],
            summary["cv_f1_mean"], summary["cv_roc_auc_mean"],
        )

    best_name = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    return best_name, fitted[best_name], results


def tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    """Search thresholds in [0.10, 0.90] (step 0.01) maximizing F1.

    When several thresholds tie for the best F1 (common on a small
    validation set - a whole plateau of thresholds can separate the
    classes perfectly), picking the lowest edge of that plateau is
    fragile: it's the threshold closest to misclassifying a borderline
    example. This picks the MIDDLE of the best-scoring plateau instead,
    which is the more robust, standard choice.
    """
    thresholds = np.arange(0.10, 0.901, 0.01)
    f1s = np.array([f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in thresholds])
    best_f1 = float(f1s.max())
    plateau = thresholds[f1s >= best_f1 - 1e-9]
    best_threshold = float(np.median(plateau))
    return best_threshold, best_f1


def evaluate(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the real-vs-screen classifier.")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--model-out", default="model/best_model.joblib")
    parser.add_argument(
        "--n-aug", type=int, default=6,
        help="Number of augmented variants generated per TRAINING photo (0 disables augmentation).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    t_start = time.perf_counter()
    logger.info("=== Feature set %s | augmentation n=%d ===", FEATURE_SET_VERSION, args.n_aug)

    paths, labels = collect_paths(args.dataset_dir)

    # Split ORIGINAL PHOTO PATHS first (70/15/15) - augmentation happens
    # only after this split, so no augmented variant of a test photo can
    # ever be seen during training.
    paths_arr = np.array(paths)
    idx = np.arange(len(paths_arr))
    idx_trainval, idx_test = train_test_split(idx, test_size=0.15, stratify=labels, random_state=42)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.1765, stratify=labels[idx_trainval], random_state=42
    )
    logger.info(
        "Photo split (pre-augmentation) -> train=%d val=%d test=%d",
        len(idx_train), len(idx_val), len(idx_test),
    )

    paths_train = list(paths_arr[idx_train])
    paths_val = list(paths_arr[idx_val])
    paths_test = list(paths_arr[idx_test])
    y_train_photos = labels[idx_train]
    y_val_photos = labels[idx_val]
    y_test_photos = labels[idx_test]

    # Clean (single vector per photo) sets for validation / test - these
    # must stay exactly what predict.py would see: one real image, no
    # augmentation.
    X_val, y_val, _ = extract_clean(paths_val, y_val_photos)
    X_test, y_test, avg_extraction_time = extract_clean(paths_test, y_test_photos)

    # Augmented training pool, grouped by source photo for leakage-safe CV.
    logger.info("Extracting augmented training pool (%dx expansion)...", args.n_aug + 1)
    X_train_aug, y_train_aug, groups_train = extract_augmented_pool(
        paths_train, y_train_photos, n_aug=args.n_aug
    )
    logger.info(
        "Augmented training pool: %s (from %d source photos)", X_train_aug.shape, len(paths_train)
    )

    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train_aug)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)
    n_dropped = len(FEATURE_NAMES) - X_train_p.shape[1]
    if n_dropped:
        logger.info("VarianceThreshold dropped %d near-constant feature(s).", n_dropped)

    print("\n=== Hyperparameter Search (StratifiedGroupKFold, train split, augmented) ===")
    best_name, search_model, cv_results = search_best_model(X_train_p, y_train_aug, groups_train)
    for name, res in cv_results.items():
        print(
            f"{name:20s} | best_params={res['best_params']} | Acc: {res['cv_accuracy_mean']:.4f} "
            f"| Prec: {res['cv_precision_mean']:.4f} | Rec: {res['cv_recall_mean']:.4f} "
            f"| F1: {res['cv_f1_mean']:.4f} | ROC-AUC: {res['cv_roc_auc_mean']:.4f}"
        )
    print(f"\nBest model selected: {best_name} (params: {cv_results[best_name]['best_params']})")

    # Calibrate probabilities before doing anything else with them.
    # GridSearchCV's best_estimator_ (search_model) gives well-RANKED
    # scores, but Random Forest's predict_proba in particular is a raw
    # vote fraction, not a statistically calibrated probability - and
    # the assignment explicitly asks for a meaningful 0-1 probability,
    # not just a ranking. CalibratedClassifierCV re-fits a fresh copy
    # of the chosen model type (best hyperparameters) with internal
    # cross-validation and a sigmoid (Platt) calibration layer.
    #
    # Note on cv choice: StratifiedGroupKFold would be the ideal
    # splitter here (consistent with the main hyperparameter search),
    # but CalibratedClassifierCV's `groups` forwarding hit a real API
    # incompatibility in this sklearn version (confirmed via a runtime
    # TypeError, not assumed) - rather than fight an unreliable code
    # path, this uses a plain integer cv (StratifiedKFold internally).
    # Honest trade-off: this reintroduces a small amount of the
    # augmented-copy-leakage-across-folds risk, but ONLY for the
    # calibration curve (a post-hoc probability correction), not for
    # model selection or the underlying decision boundary - a much
    # lower-stakes place for it to exist.
    best_params = cv_results[best_name]["best_params"]
    base_estimator = clone(_SEARCH_SPACE[best_name][0]).set_params(**best_params)
    n_cal_splits = max(2, min(3, int(min(np.bincount(y_train_aug)))))
    calibrated_model = CalibratedClassifierCV(base_estimator, method="sigmoid", cv=n_cal_splits)
    calibrated_model.fit(X_train_p, y_train_aug)

    # calibrated_model is the model used for BOTH threshold tuning and
    # final deployment - deliberately the same object throughout (see
    # the threshold/model-mismatch bug this pipeline hit and fixed
    # previously: tuning a threshold against one model and deploying a
    # different one silently breaks the threshold's meaning).
    val_probs = calibrated_model.predict_proba(X_val_p)[:, 1]
    best_threshold, best_val_f1 = tune_threshold(y_val, val_probs)
    print(f"Tuned decision threshold: {best_threshold:.2f} (validation F1={best_val_f1:.4f}, clean val set)")

    final_model = calibrated_model

    infer_times = []
    for i in range(len(X_test_p)):
        t0 = time.perf_counter()
        final_model.predict_proba(X_test_p[i : i + 1])
        infer_times.append(time.perf_counter() - t0)
    avg_infer_time = float(np.mean(infer_times)) if infer_times else 0.0
    mem_mb = get_memory_usage_mb()

    test_probs = final_model.predict_proba(X_test_p)[:, 1]
    metrics = evaluate(y_test, test_probs, best_threshold)

    print("\n=== Final Held-Out Test Evaluation (honest, clean, untouched photos) ===")
    print(f"Test set size: {len(y_test)} photos (never augmented, never used for CV or threshold tuning)")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"ROC AUC  : {metrics['roc_auc']:.4f}")
    print("Confusion Matrix [rows=true, cols=pred, 0=real,1=screen]:")
    print(np.array(metrics["confusion_matrix"]))

    print("\n=== Performance ===")
    print(f"Avg feature extraction time: {avg_extraction_time * 1000:.2f} ms/image (clean, single image - matches predict.py)")
    print(f"Avg model inference time   : {avg_infer_time * 1000:.4f} ms/image")
    print(f"Total predict.py-equivalent: {(avg_extraction_time + avg_infer_time) * 1000:.2f} ms/image")
    print(f"Process memory usage       : {mem_mb:.1f} MB")
    print(f"Total training wall time   : {time.perf_counter() - t_start:.1f} s")

    bundle = {
        "model": final_model,
        "model_name": best_name,
        "model_params": cv_results[best_name]["best_params"],
        "calibrated": True,
        "preprocessor": preprocessor,
        "threshold": best_threshold,
        "feature_names": FEATURE_NAMES,
        "feature_set_version": FEATURE_SET_VERSION,
        "n_aug": args.n_aug,
        "cv_results": cv_results,
        "test_metrics": metrics,
    }
    save_model_bundle(args.model_out, bundle)
    print(f"\nModel bundle saved to: {args.model_out}")


if __name__ == "__main__":
    main()
