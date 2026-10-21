"""
filters.py — Face quality filtering for keyframe selection.

Coarse filter:  Fast face detection to quickly drop frames with no face.
Fine filter:    Select top-K frames by (detection confidence × sharpness score).

Mathematical Design Note:
    Sharpness scoring uses the discrete 2D Laplacian differential operator
    convolution and variance computation from first principles.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import os
import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

_cascade_dir = getattr(cv2.data, "haarcascades", "")
_CASCADE_PATH = os.path.join(_cascade_dir, "haarcascade_frontalface_default.xml") if _cascade_dir else ""
_cascade: cv2.CascadeClassifier | None = None


def _get_cascade():
    global _cascade
    if _cascade is None and _CASCADE_PATH and os.path.exists(_CASCADE_PATH):
        try:
            _cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        except Exception:
            _cascade = None
    return _cascade


def coarse_has_face(frame: np.ndarray) -> bool:
    """True if face detector finds at least one face candidate in the BGR frame."""
    cas = _get_cascade()
    if cas is None or getattr(cas, "empty", lambda: True)():
        return True
    try:
        # Standard ITU-R BT.601 luminance conversion: Y = 0.299*R + 0.587*G + 0.114*B
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        return len(faces) > 0
    except Exception:
        return True


def compute_discrete_laplacian_2d(gray: np.ndarray) -> np.ndarray:
    """
    Computes the discrete 2D Laplacian operator via finite differences:
        ∇²I(x, y) = d²I/dx² + d²I/dy²
                  ≈ [I(x+1, y) - 2I(x, y) + I(x-1, y)] + [I(x, y+1) - 2I(x, y) + I(x, y-1)]
                  = I(x+1, y) + I(x-1, y) + I(x, y+1) + I(x, y-1) - 4*I(x, y)

    Corresponding 3x3 convolution kernel:
        [ 0,  1,  0 ]
        [ 1, -4,  1 ]
        [ 0,  1,  0 ]

    Applies reflection boundary padding (reflect101) to eliminate border artifacts.
    """
    img = np.asarray(gray, dtype=np.float64)
    # Symmetric reflection padding around borders
    pad = np.pad(img, 1, mode="reflect")
    # 4-connected spatial second derivatives
    lap = (
        pad[2:, 1:-1] +
        pad[:-2, 1:-1] +
        pad[1:-1, 2:] +
        pad[1:-1, :-2] -
        4.0 * pad[1:-1, 1:-1]
    )
    return lap


def sharpness_score(frame: np.ndarray) -> float:
    """
    Computes the focus/sharpness score using the variance of the 2D Laplacian:
        Score = Var(∇²I) = (1 / HW) * Sum (∇²I(x, y) - μ)²

    Sharp images have pronounced high-frequency gradients (high variance),
    whereas blurred or out-of-focus images have attenuated edges (low variance).
    """
    if frame is None or frame.size == 0:
        return 0.0
    if frame.ndim == 3:
        # Standard ITU-R BT.601 luminance conversion
        gray = 0.114 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.299 * frame[:, :, 2]
    else:
        gray = frame
    lap = compute_discrete_laplacian_2d(gray)
    return float(np.var(lap))


def select_keyframes(
    frames: List[np.ndarray],
    scores: List[float],
    k: int = 8,
    score_threshold: float = 0.0,
) -> List[np.ndarray]:
    """
    Return top-K frames ranked by (face confidence × sharpness).
    Frames with score < score_threshold are discarded before ranking.
    Falls back to all frames if nothing passes the threshold.
    """
    if not frames:
        return []
    pairs = list(zip(scores, frames))
    if score_threshold > 0.0:
        filtered = [(s, f) for s, f in pairs if s >= score_threshold]
        if not filtered:
            filtered = pairs   # fallback: threshold too strict, use all
    else:
        filtered = pairs
    ranked = sorted(filtered, key=lambda x: x[0], reverse=True)
    selected = [f for _, f in ranked[:k]]
    while len(selected) < k:
        selected.append(selected[-1])
    return selected[:k]


def frames_to_pil(frames: List[np.ndarray], size: int = 224) -> List[Image.Image]:
    """Convert BGR numpy frames to PIL Images resized to (size × size)."""
    result = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
        result.append(img)
    return result
