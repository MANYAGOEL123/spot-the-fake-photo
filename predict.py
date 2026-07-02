"""
predict.py
==========
Required usage (per assignment spec):

    python predict.py some_image.jpg
    -> 0.93

Prints exactly one float in [0, 1]: probability the image is a photo
of a screen/printout (a "recapture"). 0 = real photo, 1 = screen photo.
All logs/errors go to stderr so stdout stays a single clean number.
"""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn

import numpy as np

from utils import extract_features, load_image, load_model_bundle, logger


def _fail(message: str) -> NoReturn:
    logger.error(message)
    print("error", file=sys.stderr)
    sys.exit(1)


def predict_probability(image_path: str, model_path: str = "model/best_model.joblib") -> float:
    """Run the full pipeline on a single image and return P(screen photo)."""
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    expected_names = bundle["feature_names"]

    image = load_image(image_path)
    vector = extract_features(image)

    # Exact identity check, not just a length check: two feature sets
    # of the same length but different order (e.g. loading a bundle
    # trained against an older utils.py that happened to produce the
    # same feature count) would otherwise silently feed misaligned
    # values into the model.
    from utils import FEATURE_NAMES as current_feature_names

    if list(expected_names) != list(current_feature_names):
        _fail(
            "Feature schema mismatch between this model bundle and the current "
            "utils.py (same or different length, different feature identity/order). "
            "The model was trained against a different feature-extraction version - retrain with train.py."
        )

    X = preprocessor.transform(vector.reshape(1, -1))
    proba = float(model.predict_proba(X)[0, 1])
    return proba


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict whether an image is a real photo (0) or a screen/printout recapture (1)."
    )
    parser.add_argument("image", help="Path to the input image file.")
    parser.add_argument("--model", default="model/best_model.joblib", help="Path to trained model bundle.")
    args = parser.parse_args()

    try:
        proba = predict_probability(args.image, model_path=args.model)
    except FileNotFoundError as exc:
        _fail(str(exc))
    except ValueError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        _fail(f"Unexpected error during prediction: {exc}")
    else:
        print(f"{proba:.2f}")


if __name__ == "__main__":
    main()
