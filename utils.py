"""
utils.py
========
Feature extraction + shared helpers for the real-vs-screen-recapture
classifier.

Design goal: SMALL, FAST, CHEAP, and GENERALIZES across display/print
technology and capture conditions. Every feature below is a classical
OpenCV / scikit-image / scipy computation chosen to target a physical
artifact of photographing a screen or a printed photo. No deep-learning
backbone is used.

Generalization notes (this revision)
-------------------------------------
The training photos on hand only cover phone/laptop/TV recaptures shot
indoors in roughly-daylight lighting. The hidden evaluation set is
expected to also include OLED/LCD/tablet/monitor/projector displays,
printed photographs, and extreme conditions (night, low-light, bright
sunlight, reflections, extreme viewing angles) that AREN'T represented
in the raw training photos. Two changes address this:

1. Several features that were previously computed on raw pixel values
   with FIXED assumptions (a single characteristic frequency band for
   moire; a fixed absolute brightness threshold for glare/specular)
   are now either illumination-normalized, multi-scale, or adaptive -
   so they stay meaningful outside the lighting/pixel-pitch range the
   training photos happen to cover.
2. Training-only augmentation (`augment_image`, used exclusively by
   train.py - never at predict time) now covers a much wider
   degradation space: extreme exposure (night through bright
   sunlight), perspective/keystone warp (extreme viewing angles,
   projector geometry), synthetic sensor noise (low-light), and
   defocus/motion blur (projector softness, camera shake).

Honesty caveat, stated once here rather than scattered: none of this
is empirically validated against real OLED/tablet/monitor/projector/
printed-photo recaptures, because none exist in the training set.
These changes are principled (grounded in the actual physical
differences between these capture conditions) but unverified for the
specific new categories - report this accurately, don't oversell it.

Architecture
------------
All expensive per-image primitives are computed once per image into an
`_ImageContext` and every feature function reads from that shared
context - no feature recomputes a primitive another feature already
computed. Feature functions are registered via the `@feature(...)`
decorator, which declares each function's output name(s) inline;
`FEATURE_NAMES` is *derived* from that registry at import time rather
than hand-maintained, so it cannot drift out of sync with the
feature-assembly code.

Every feature function is defensive: a failure on one feature logs a
warning and returns its declared default instead of crashing the whole
extraction. The final feature vector is also sanitized (NaN/Inf -> 0.0)
as a last line of defense.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pywt
from scipy.fft import fft2, fftshift
from scipy.stats import skew
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logger = logging.getLogger("screen_detector")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Bumped: illumination-normalized preprocessing + multi-band frequency
# analysis + new features change every downstream numeric value, so a
# model trained on the previous feature set is NOT compatible with
# this one. predict.py checks this explicitly and fails loudly rather
# than silently mispredicting.
FEATURE_SET_VERSION = "lean-3.0"

_RESIZE_MAX_DIM = 700  # long-edge cap for the main pipeline
_LBP_MAX_DIM = 300  # texture/entropy stats are stable under heavier downsampling
_CHANNEL_MOIRE_MAX_DIM = 300  # per-channel frequency analysis - keep cheap


# --------------------------------------------------------------------------
# Image I/O
# --------------------------------------------------------------------------
def load_image(image_path: str) -> np.ndarray:
    """Load an image as BGR uint8.

    Routes through PIL first and applies EXIF orientation correction -
    phones routinely store portrait photos as landscape pixel data plus
    a rotation tag; `cv2.imread` ignores this entirely, which would
    silently feed a sideways/upside-down image into
    orientation-sensitive features. Falls back to raw `cv2.imread` for
    formats PIL can't decode.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: file cannot be decoded as an image, or is too small
            to analyze reliably.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = None
    try:
        from PIL import Image, ImageOps

        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img)  # applies EXIF rotation, if any
        pil_img = pil_img.convert("RGB")
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as exc:  # noqa: BLE001
        logger.debug("PIL load failed for '%s' (%s); falling back to OpenCV.", image_path, exc)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise ValueError(f"Could not decode image: {image_path}")

    h, w = image.shape[:2]
    if min(h, w) < 32:
        raise ValueError(
            f"Image too small to analyze reliably ({w}x{h}, minimum 32x32): {image_path}"
        )

    return image


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _resize_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale large images for speed (never upscale).

    Clamps the target dimensions to a minimum of 1px - an extreme
    aspect-ratio input (e.g. a 2px-tall x 1000px-wide image) can
    otherwise scale one dimension down to 0, which crashes cv2.resize.
    Found via adversarial testing, not organically - the training
    photos are all normal phone-camera aspect ratios."""
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def _fft_grid(gray_f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute FFT magnitude / log-magnitude / radial-distance / angle
    grid for an arbitrary single-channel float32 image. Shared by every
    frequency-domain feature (main multi-band moire, per-channel
    divergence) so the underlying math lives in exactly one place.

    Returns:
        (magnitude, log_magnitude, dist, theta, radius)
    """
    f = fft2(gray_f)
    fshift = fftshift(f)
    magnitude = np.abs(fshift)
    log_magnitude = np.log1p(magnitude)
    h, w = gray_f.shape
    cy, cx = h // 2, w // 2
    radius = float(min(cy, cx)) + 1e-8
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    theta = np.arctan2(np.broadcast_to(yy - cy, (h, w)), np.broadcast_to(xx - cx, (h, w)))
    return magnitude, log_magnitude, dist, theta, radius


def _band_peak_ratio(log_magnitude: np.ndarray, dist: np.ndarray, radius: float, r_lo: float, r_hi: float) -> float:
    """Fraction of pixels in an annular frequency band whose energy
    exceeds a local outlier threshold - the core "is there a strong
    periodic structure at this scale" test, reused across every
    moire-style feature."""
    band_vals = log_magnitude[(dist > radius * r_lo) & (dist < radius * r_hi)]
    if band_vals.size == 0:
        return 0.0
    band_mean, band_std = float(np.mean(band_vals)), float(np.std(band_vals))
    if band_std < 1e-6:
        return 0.0
    threshold = band_mean + 3.0 * band_std
    return float(np.count_nonzero(band_vals > threshold)) / float(band_vals.size)


# --------------------------------------------------------------------------
# Shared per-image context - every expensive primitive computed ONCE
# --------------------------------------------------------------------------
@dataclass
class _ImageContext:
    image: np.ndarray  # resized BGR, uint8 (RAW)
    gray: np.ndarray  # resized grayscale, uint8 (RAW)
    gray_f: np.ndarray  # grayscale as float32 (RAW)
    gray_norm: np.ndarray  # CLAHE-normalized grayscale, uint8
    gray_norm_f: np.ndarray  # CLAHE-normalized grayscale, float32
    hsv: np.ndarray  # HSV of resized image (RAW)
    gradient_x: np.ndarray  # Sobel dx (on gray_norm)
    gradient_y: np.ndarray  # Sobel dy (on gray_norm)
    gradient_mag: np.ndarray  # sqrt(dx^2 + dy^2) (on gray_norm)
    laplacian: np.ndarray  # cv2.Laplacian (on gray_norm)
    canny_edges: np.ndarray  # cv2.Canny (on gray_norm)
    fft_magnitude: np.ndarray  # |FFT|, shifted (on gray_norm)
    fft_log_magnitude: np.ndarray  # log1p(|FFT|) (on gray_norm)
    dist: np.ndarray  # radial distance grid, same shape as fft
    theta: np.ndarray  # angle grid (radians), same shape as fft
    radius: float  # normalization radius for dist
    lbp_gray: np.ndarray  # small (<=300px) CLAHE-normalized copy for LBP
    entropy_gray: np.ndarray  # small (<=300px) RAW copy for entropy


def _build_context(image: np.ndarray) -> _ImageContext:
    image = _resize_max_dim(image, _RESIZE_MAX_DIM)
    gray = _to_gray(image)
    gray_f = gray.astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # CLAHE-normalized grayscale: equalizes broad illumination gradients
    # (so night/low-light/bright-sunlight extremes don't swamp the fine
    # periodic/edge signal we actually care about) while preserving
    # local micro-texture. Brightness/color-cast/glare features
    # deliberately stay on the RAW gray/hsv above - normalizing those
    # away would discard a real, separate signal (genuine exposure,
    # backlight color cast).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_norm = clahe.apply(gray)
    gray_norm_f = gray_norm.astype(np.float32)

    gx = cv2.Sobel(gray_norm, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_norm, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gx ** 2 + gy ** 2)

    laplacian = cv2.Laplacian(gray_norm, cv2.CV_64F)
    canny_edges = cv2.Canny(gray_norm, 100, 200)

    fft_magnitude, fft_log_magnitude, dist, theta, radius = _fft_grid(gray_norm_f)

    lbp_gray = _resize_max_dim(gray_norm, _LBP_MAX_DIM)
    entropy_gray = _resize_max_dim(gray, _LBP_MAX_DIM)

    return _ImageContext(
        image=image, gray=gray, gray_f=gray_f, gray_norm=gray_norm, gray_norm_f=gray_norm_f, hsv=hsv,
        gradient_x=gx, gradient_y=gy, gradient_mag=gradient_mag,
        laplacian=laplacian, canny_edges=canny_edges,
        fft_magnitude=fft_magnitude, fft_log_magnitude=fft_log_magnitude,
        dist=dist, theta=theta, radius=radius, lbp_gray=lbp_gray, entropy_gray=entropy_gray,
    )


# --------------------------------------------------------------------------
# Feature registry - FEATURE_NAMES is DERIVED from this, never hand-listed
# --------------------------------------------------------------------------
_FEATURE_REGISTRY: List[Tuple[str, Callable[[_ImageContext], Dict[str, float]], List[str]]] = []


def feature(*names: str, default: Optional[Sequence[float]] = None):
    """Register a feature function under one or more output names.

    The wrapped function receives an `_ImageContext` and returns either
    a single float (if one name is given) or a tuple matching `names`.
    On any exception, logs a warning and substitutes `default` (or
    zeros) so a single bad feature never crashes extraction.
    """
    fallback = tuple(default) if default is not None else tuple(0.0 for _ in names)
    assert len(fallback) == len(names), "default length must match declared names"

    def decorator(fn: Callable[[_ImageContext], object]):
        def wrapper(ctx: _ImageContext) -> Dict[str, float]:
            try:
                result = fn(ctx)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Feature '%s' failed (%s); using default=%s", fn.__name__, exc, fallback)
                result = fallback
            if not isinstance(result, (tuple, list)):
                result = (result,)
            return dict(zip(names, (float(v) for v in result)))

        _FEATURE_REGISTRY.append((fn.__name__, wrapper, list(names)))
        return wrapper

    return decorator


# --------------------------------------------------------------------------
# Handcrafted features
# --------------------------------------------------------------------------
@feature("laplacian_variance")
def _f_laplacian_variance(ctx: _ImageContext) -> float:
    """Classic focus/blur measure."""
    return float(ctx.laplacian.var())


@feature("tenengrad_sharpness")
def _f_tenengrad_sharpness(ctx: _ImageContext) -> float:
    return float(np.mean(ctx.gradient_mag ** 2))


@feature("fft_high_freq_ratio", "fft_low_freq_ratio", "fft_spectral_centroid",
          default=(0.0, 0.0, 0.0))
def _f_fft_features(ctx: _ImageContext) -> Tuple[float, float, float]:
    """High/low-frequency energy ratios + radial spectral centroid.
    A screen's sub-pixel grid + refresh cycle, or a printed halftone
    dot pattern, recaptured by a camera, pushes energy into structured
    high-frequency bands that natural scenes rarely produce."""
    magnitude, dist, radius = ctx.fft_magnitude, ctx.dist, ctx.radius
    total = float(np.sum(magnitude)) + 1e-8
    low = float(np.sum(magnitude[dist <= radius * 0.15]))
    high = float(np.sum(magnitude[dist >= radius * 0.5]))
    weights = magnitude / total
    centroid = float(np.sum(dist * weights) / (np.sum(weights) + 1e-8)) / radius
    return high / total, low / total, centroid


@feature("moire_score_fine", "moire_score_mid", "moire_score_coarse", default=(0.0, 0.0, 0.0))
def _f_moire_multiband(ctx: _ImageContext) -> Tuple[float, float, float]:
    """Multi-scale moire detection across 3 frequency bands.

    A single fixed frequency band (as an earlier version of this file
    used) implicitly assumes one characteristic pixel pitch / viewing
    distance. The hidden evaluation set spans OLED phones held close,
    tablets, computer monitors, and projectors viewed from across a
    room - these have very different effective pixel pitches once
    photographed, so their moire signal lands in different frequency
    bands. Splitting into fine/mid/coarse bands lets the classifier
    learn which combination matters, instead of missing displays whose
    pitch falls outside one fixed band.
    """
    magnitude, dist, radius = ctx.fft_log_magnitude, ctx.dist, ctx.radius
    fine = _band_peak_ratio(magnitude, dist, radius, 0.08, 0.25)
    mid = _band_peak_ratio(magnitude, dist, radius, 0.25, 0.5)
    coarse = _band_peak_ratio(magnitude, dist, radius, 0.5, 0.8)
    return fine, mid, coarse


@feature("moire_axis_anisotropy")
def _f_moire_axis_anisotropy(ctx: _ImageContext) -> float:
    """Whether mid-band frequency peaks are split between the
    horizontal AND vertical axes (screen-like) vs concentrated on a
    single axis (e.g. window blinds/shutters/grilles - a real-world
    object with a strong 1-D repeating pattern).

    A digital display's sub-pixel grid is a 2-D lattice, periodic in
    both directions, so a screen recapture tends to show peak energy
    on both frequency axes. A physical object with a 1-D repeating
    pattern only produces peaks along the single axis perpendicular to
    its stripes. Band widened to 0.08-0.8 to match the multi-band
    moire detector above, so this stays consistent across pixel
    pitches too.

    Returns 0.0 when peak energy is confined to one axis (real-object
    evidence), rising toward 1.0 when it's balanced across both axes
    (screen-grid evidence).
    """
    magnitude, dist, theta, radius = ctx.fft_log_magnitude, ctx.dist, ctx.theta, ctx.radius
    mid_band = (dist > radius * 0.08) & (dist < radius * 0.8)
    band_vals = magnitude[mid_band]
    band_theta = theta[mid_band]
    if band_vals.size == 0:
        return 0.0
    band_mean, band_std = float(np.mean(band_vals)), float(np.std(band_vals))
    if band_std < 1e-6:
        return 0.0
    peak_mask = band_vals > (band_mean + 2.5 * band_std)
    if np.count_nonzero(peak_mask) < 4:
        return 0.0

    peak_theta = band_theta[peak_mask]
    theta_mod = np.mod(peak_theta, np.pi)  # axes repeat every 180deg
    dist_to_horizontal = np.minimum(theta_mod, np.pi - theta_mod)
    dist_to_vertical = np.abs(theta_mod - np.pi / 2)
    window = np.radians(15)

    h_count = int(np.count_nonzero(dist_to_horizontal < window))
    v_count = int(np.count_nonzero(dist_to_vertical < window))
    denom = max(h_count, v_count)
    if denom == 0:
        return 0.0
    return float(min(h_count, v_count) / denom)


@feature("channel_moire_divergence")
def _f_channel_moire_divergence(ctx: _ImageContext) -> float:
    """Independent periodicity score per R/G/B channel, then the
    spread across channels.

    Emissive RGB displays (phone/tablet/monitor/TV/OLED/LCD) share one
    underlying pixel geometry across channels, so their periodic
    signal tends to be fairly correlated between R, G, and B. Halftone
    print reproduction uses independently-angled screens per ink
    channel (classically distinct angles per CMYK plate), so a
    recaptured PRINTED PHOTOGRAPH tends to show more divergent
    per-channel periodicity than an emissive display does. This is the
    one feature in this file aimed specifically at the "printed
    photograph" recapture category.

    Honesty caveat: this is physically motivated but NOT empirically
    validated - no real printed-photo recaptures were available in the
    training set to confirm it behaves as expected. Treat it as a
    reasonable prior the classifier can weight appropriately, not a
    proven discriminator.
    """
    small = _resize_max_dim(ctx.image, _CHANNEL_MOIRE_MAX_DIM)
    scores = []
    for c in range(3):
        channel = small[:, :, c].astype(np.float32)
        _, log_mag, dist, _, radius = _fft_grid(channel)
        scores.append(_band_peak_ratio(log_mag, dist, radius, 0.15, 0.6))
    return float(np.std(scores))


@feature("edge_density")
def _f_edge_density(ctx: _ImageContext) -> float:
    edges = ctx.canny_edges
    return float(np.count_nonzero(edges)) / float(edges.size)


@feature("gradient_mean", "gradient_std", "gradient_skew", default=(0.0, 0.0, 0.0))
def _f_gradient_stats(ctx: _ImageContext) -> Tuple[float, float, float]:
    mag = ctx.gradient_mag
    mean_val = float(np.mean(mag))
    std_val = float(np.std(mag))
    # skew() on the full array is the expensive part of this feature;
    # a stride-4 subsample gives a near-identical distribution shape
    # estimate at a fraction of the cost.
    sample = mag.flatten()[::4]
    skew_val = float(skew(sample)) if std_val > 1e-6 else 0.0
    return mean_val, std_val, skew_val


@feature("saturation_mean", "saturation_std", default=(0.0, 0.0))
def _f_saturation_stats(ctx: _ImageContext) -> Tuple[float, float]:
    sat = ctx.hsv[:, :, 1].astype(np.float32) / 255.0
    return float(np.mean(sat)), float(np.std(sat))


@feature("brightness_mean", "brightness_std", "brightness_hist_entropy", default=(0.0, 0.0, 0.0))
def _f_brightness_stats(ctx: _ImageContext) -> Tuple[float, float, float]:
    g = ctx.gray_f / 255.0
    hist, _ = np.histogram(ctx.gray, bins=256, range=(0, 256))
    p = hist.astype(np.float32) / (hist.sum() + 1e-8)
    hist_entropy = float(-np.sum(p * np.log2(p + 1e-12)))
    return float(np.mean(g)), float(np.std(g)), hist_entropy


@feature("specular_highlight_ratio")
def _f_specular_highlight_ratio(ctx: _ImageContext) -> float:
    """Ratio of near-white, low-saturation pixels, using an ADAPTIVE
    (percentile-based) brightness threshold rather than a fixed
    absolute value. A fixed threshold like 'v > 240' is implicitly
    calibrated for daylight: in low-light/night photos almost nothing
    reaches it (the feature goes dead), and in bright sunlight almost
    everything does (the feature saturates and stops discriminating).
    Anchoring to the scene's own brightness distribution keeps this
    meaningful across the full lighting range."""
    v = ctx.hsv[:, :, 2].astype(np.float32)
    s = ctx.hsv[:, :, 1]
    threshold = max(200.0, float(np.percentile(v, 99)))
    mask = (v >= threshold) & (s < 30)
    return float(np.count_nonzero(mask)) / float(mask.size)


@feature("glare_score")
def _f_glare_score(ctx: _ImageContext) -> float:
    """Area ratio of the largest bright connected blob, using an
    adaptive (percentile-based) threshold for the same reason as
    `specular_highlight_ratio` above - stays meaningful in both very
    dark and very bright scenes instead of being calibrated only for
    daylight."""
    gray = ctx.gray
    threshold = max(200, int(np.percentile(gray, 99)))
    _, thresh_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(cv2.contourArea(c) for c in contours)
    return float(largest) / float(gray.size)


@feature("reflection_symmetry_score")
def _f_reflection_symmetry_score(ctx: _ImageContext) -> float:
    """Left-right symmetry of the BRIGHT REGION's bounding box only
    (not the whole frame). Threshold is already scene-relative
    (mean + 1 std of the image's own brightness), so this stays
    meaningful across lighting conditions without further changes."""
    gray = ctx.gray
    bright_mask = gray > (gray.mean() + gray.std())
    ys, xs = np.nonzero(bright_mask)
    if len(ys) < 50:
        return 0.0
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    region = gray[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    if region.size == 0:
        return 0.0
    flipped = np.fliplr(region)
    diff = np.abs(region - flipped)
    score = float(np.mean(diff)) / 255.0
    return float(max(0.0, 1.0 - score))


@feature("lbp_uniformity", "lbp_entropy", default=(0.0, 0.0))
def _f_lbp_features(ctx: _ImageContext) -> Tuple[float, float]:
    """Uniform LBP texture, summarized to 2 scalars. Runs on the
    pre-downsampled, CLAHE-normalized `ctx.lbp_gray` - normalization
    helps reveal fine micro-texture in poorly-lit regions that would
    otherwise be crushed into a handful of raw quantization levels."""
    lbp = local_binary_pattern(ctx.lbp_gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    uniformity = float(np.sum(hist ** 2))
    hist_safe = hist + 1e-12
    entropy_val = float(-np.sum(hist_safe * np.log2(hist_safe)))
    return uniformity, entropy_val


@feature("wavelet_ll_energy", "wavelet_lh_energy", "wavelet_hl_energy", "wavelet_hh_energy",
          default=(0.0, 0.0, 0.0, 0.0))
def _f_wavelet_energy(ctx: _ImageContext) -> Tuple[float, float, float, float]:
    """Haar wavelet subband energy on the CLAHE-normalized image, for
    the same illumination-robustness reason as the other fine-detail
    detectors above."""
    coeffs = pywt.dwt2(ctx.gray_norm_f, "haar")
    ll, (lh, hl, hh) = coeffs
    total = float(np.sum(ll ** 2) + np.sum(lh ** 2) + np.sum(hl ** 2) + np.sum(hh ** 2)) + 1e-8
    return (
        float(np.sum(ll ** 2)) / total,
        float(np.sum(lh ** 2)) / total,
        float(np.sum(hl ** 2)) / total,
        float(np.sum(hh ** 2)) / total,
    )


@feature("noise_std", "noise_mean_abs", default=(0.0, 0.0))
def _f_noise_stats(ctx: _ImageContext) -> Tuple[float, float]:
    """Raw median-filter noise residual, on the RAW (not CLAHE
    -normalized) image - normalization could distort the noise
    estimate non-uniformly across dark/bright regions."""
    median = cv2.medianBlur(ctx.gray, 3)
    residual = ctx.gray_f - median.astype(np.float32)
    return float(np.std(residual)), float(np.mean(np.abs(residual)))


@feature("noise_std_normalized")
def _f_noise_normalized(ctx: _ImageContext) -> float:
    """Noise residual normalized by the image's own mean brightness.

    Absolute sensor noise is naturally much higher in low-light/night
    photos than in daylight, regardless of whether the photo is a
    recapture - a raw noise threshold tuned on daylight photos would
    misfire constantly on night photos. Dividing by local brightness
    keeps this feature comparable across the full lighting range."""
    median = cv2.medianBlur(ctx.gray, 3)
    residual = ctx.gray_f - median.astype(np.float32)
    denom = float(np.mean(ctx.gray_f)) + 5.0  # floor avoids blow-up in near-black scenes
    return float(np.std(residual)) / denom


@feature("entropy")
def _f_entropy_score(ctx: _ImageContext) -> float:
    """Shannon entropy on a small RAW (not CLAHE-normalized) copy -
    CLAHE tends to flatten histograms toward uniform, which would
    compress this feature's useful range."""
    return float(shannon_entropy(ctx.entropy_gray))


@feature("contrast_std")
def _f_contrast_score(ctx: _ImageContext) -> float:
    """Raw global contrast - deliberately NOT normalized, since low
    apparent contrast (e.g. washed out in bright sunlight, or crushed
    blacks at night) is itself part of the genuine signal here."""
    return float(np.std(ctx.gray_f) / 255.0)


@feature(
    "color_mean_b", "color_mean_g", "color_mean_r",
    "color_std_b", "color_std_g", "color_std_r",
    default=(0.0,) * 6,
)
def _f_color_means_stds(ctx: _ImageContext) -> Tuple[float, ...]:
    """Per-channel (B,G,R) mean and std, on the RAW image - captures
    display/panel color cast vs natural illumination. Deliberately not
    normalized: the color cast itself is the signal."""
    means, stds = [], []
    for c in range(3):
        channel = ctx.image[:, :, c].astype(np.float32).flatten() / 255.0
        means.append(float(np.mean(channel)))
        stds.append(float(np.std(channel)))
    return (*means, *stds)


@feature("aliasing_score")
def _f_aliasing_score(ctx: _ImageContext) -> float:
    edges = ctx.canny_edges
    if np.count_nonzero(edges) == 0:
        return 0.0
    edge_energy = float(np.mean(np.abs(ctx.laplacian[edges > 0])))
    total_energy = float(np.mean(np.abs(ctx.laplacian))) + 1e-8
    return edge_energy / total_energy


# Derived once, at import time, directly from the registry above - this
# CANNOT drift out of sync with the assembly code the way two
# hand-maintained parallel lists could.
FEATURE_NAMES: List[str] = [name for _, _, names in _FEATURE_REGISTRY for name in names]


def extract_features(image: np.ndarray) -> np.ndarray:
    """Compute the full feature vector for a single BGR image, in the
    fixed order of `FEATURE_NAMES`.

    Args:
        image: BGR uint8 image.

    Returns:
        1-D float32 array, len == len(FEATURE_NAMES). NaN/Inf values
        (which can arise from degenerate images, e.g. a single flat
        color, or a near-black night photo) are sanitized to 0.0.
    """
    ctx = _build_context(image)
    values: Dict[str, float] = {}
    for _, fn, _ in _FEATURE_REGISTRY:
        values.update(fn(ctx))

    vector = np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return vector


# --------------------------------------------------------------------------
# Training-only augmentation (never used at predict time)
# --------------------------------------------------------------------------
def _rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)


def _perspective_warp(image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Random perspective (keystone) warp - simulates photographing a
    flat display or print at an extreme off-axis viewing angle, or the
    natural keystone distortion of a projected image. `strength` is a
    fraction of the shorter image dimension used as the max corner
    displacement."""
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    max_shift = strength * min(h, w)
    dst = (src + rng.uniform(-max_shift, max_shift, src.shape)).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)


def _random_crop_zoom(image: np.ndarray, keep_fraction: float) -> np.ndarray:
    h, w = image.shape[:2]
    ch, cw = int(h * keep_fraction), int(w * keep_fraction)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    cropped = image[y0 : y0 + ch, x0 : x0 + cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def _adjust_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _add_sensor_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise - simulates high-ISO sensor noise from
    low-light/night capture conditions, which real training photos
    (all shot in reasonable indoor/daylight lighting) don't cover."""
    noise = rng.normal(0, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _gaussian_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def _downscale_upscale(image: np.ndarray, factor: float) -> np.ndarray:
    """Downscale then upscale back - simulates a common way fine
    detail gets destroyed in practice (aggressive resizing, a cheap
    resave, a messaging-app compression pass) independent of blur."""
    h, w = image.shape[:2]
    small_w, small_h = max(1, int(w * factor)), max(1, int(h * factor))
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _jpeg_recompress(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return image
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def augment_image(
    image: np.ndarray, n: int = 4, rng: Optional[np.random.Generator] = None
) -> List[np.ndarray]:
    """Generate `n` augmented variants of an image for TRAINING-SET
    expansion only (never used at predict time).

    Covers a deliberately wide degradation space so the model sees
    conditions the raw training photos don't - extreme exposure (night
    through bright sunlight), perspective/keystone warp (extreme
    viewing angles, projector geometry), synthetic sensor noise
    (low-light), and defocus/motion blur/downscale-upscale laundering -
    in addition to rotation/crop/saturation/JPEG jitter.

    Adversarial-robustness note: an earlier version of this function
    only applied MILD blur (kernel <=9) at low probability. Testing the
    trained model directly showed it had learned "low noise + low
    sharpness -> screen" as a shortcut from that narrow range - pushed
    past kernel 9 (which the model had never seen at training time), a
    genuinely blurry REAL photo (e.g. ordinary motion blur or a focus
    miss - extremely common) got confidently misclassified as a screen
    recapture. The fix isn't a new feature, it's making sure BOTH
    classes are represented across a much wider blur/degradation range
    during training, so the model can't use "low detail" as a proxy for
    "screen" - it has to actually distinguish blurry-real from
    blurry-screen using signal that survives blur (color cast,
    residual periodicity) rather than sharpness alone.

    Args:
        image: Original BGR uint8 image.
        n: Number of augmented variants to produce.
        rng: Optional numpy random Generator for reproducibility.

    Returns:
        List of `n` augmented BGR uint8 images (same size as input).
    """
    if rng is None:
        rng = np.random.default_rng()

    variants: List[np.ndarray] = []
    for _ in range(n):
        aug = image.copy()

        # Geometric: mild rotation always; occasional stronger
        # perspective warp for extreme-angle / projector-keystone cases.
        aug = _rotate(aug, float(rng.uniform(-8, 8)))
        if rng.random() < 0.35:
            aug = _perspective_warp(aug, strength=float(rng.uniform(0.03, 0.12)), rng=rng)

        # Exposure: widened to span night/low-light through bright
        # sunlight, not just mild indoor variation.
        alpha = float(rng.uniform(0.4, 1.7))  # contrast/gain
        beta = float(rng.uniform(-70, 70))  # brightness offset
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

        # Synthetic sensor noise, more likely (and stronger) when the
        # exposure branch above pushed brightness down - mimics real
        # low-light/night sensor behavior.
        if alpha < 0.8 or rng.random() < 0.3:
            aug = _add_sensor_noise(aug, sigma=float(rng.uniform(3, 18)), rng=rng)

        if rng.random() < 0.7:
            aug = _random_crop_zoom(aug, float(rng.uniform(0.8, 0.98)))

        aug = _adjust_saturation(aug, float(rng.uniform(0.7, 1.3)))

        # Defocus/motion blur AND/OR downscale-upscale detail loss.
        # Range widened substantially (was kernel 3-9 at 30% chance) -
        # a meaningful fraction of BOTH classes now see genuinely heavy
        # degradation, not just mild jitter, so the model can't use
        # "low detail" alone as a class proxy. See adversarial-
        # robustness note above.
        degrade_roll = rng.random()
        if degrade_roll < 0.20:
            aug = _gaussian_blur(aug, ksize=int(rng.integers(3, 9)))  # mild
        elif degrade_roll < 0.35:
            aug = _gaussian_blur(aug, ksize=int(rng.integers(9, 23)))  # heavy
        elif degrade_roll < 0.50:
            aug = _downscale_upscale(aug, factor=float(rng.uniform(0.2, 0.6)))

        quality = int(rng.integers(20, 95))
        aug = _jpeg_recompress(aug, quality)

        variants.append(aug)
    return variants


def extract_features_from_path(image_path: str) -> Tuple[np.ndarray, float]:
    """Load + extract features, returning (vector, elapsed_seconds)."""
    t0 = time.perf_counter()
    image = load_image(image_path)
    vector = extract_features(image)
    return vector, time.perf_counter() - t0


# --------------------------------------------------------------------------
# Model persistence
# --------------------------------------------------------------------------
def save_model_bundle(path: str, bundle: dict) -> None:
    import joblib

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(bundle, path)
    logger.info("Saved model bundle to %s", path)


def load_model_bundle(path: str) -> dict:
    import joblib

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model bundle not found at '{path}'. Run train.py first.")
    return joblib.load(path)


def list_dataset_images(dataset_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def get_memory_usage_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        return -1.0
