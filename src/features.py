"""Feature engineering for the exercise classifier.

Per frame the classifier sees:

* the :data:`~src.angle_engine.CLASSIFIER_METRICS` subset of the angle engine's
  output (27 values), and
* hip-centred, torso-normalised x/y for the 15 core landmarks (30 values).

Per *window* (default 45 frames ~ 1.5 s) those 57 channels are reduced to five
summary statistics each - mean, std, min, max and net drift (last - first) -
giving a 285-dim vector. Aggregating a window rather than feeding raw sequences
is what lets a Random Forest do this job on CPU: it captures both the posture
(mean/min/max) and the motion (std/drift) without a recurrent model.

Everything is normalised for body size and camera distance, so a vector taken
from a tall person 3 m from the webcam is comparable to a short person at 1.5 m.
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Iterable, Mapping, Sequence

import numpy as np

from . import landmarks as LM
from .angle_engine import CLASSIFIER_METRICS, metrics_to_vector
from .pose_extraction import aspect_correct

DEFAULT_WINDOW = 45
DEFAULT_STRIDE = 10

#: Aggregations applied per channel over a window.
AGGREGATIONS: tuple[str, ...] = ("mean", "std", "min", "max", "drift")

COORD_LANDMARKS: tuple[int, ...] = LM.CORE_LANDMARKS


def frame_feature_names() -> tuple[str, ...]:
    names = list(CLASSIFIER_METRICS)
    for idx in COORD_LANDMARKS:
        names += [f"{LM.LANDMARK_NAMES[idx]}_nx", f"{LM.LANDMARK_NAMES[idx]}_ny"]
    return tuple(names)


FRAME_FEATURE_NAMES: tuple[str, ...] = frame_feature_names()
N_FRAME_FEATURES = len(FRAME_FEATURE_NAMES)


def window_feature_names() -> tuple[str, ...]:
    return tuple(f"{agg}__{name}" for agg in AGGREGATIONS for name in FRAME_FEATURE_NAMES)


WINDOW_FEATURE_NAMES: tuple[str, ...] = window_feature_names()
N_WINDOW_FEATURES = len(WINDOW_FEATURE_NAMES)


def frame_features(
    pts: np.ndarray,
    image_size: tuple[int, int],
    metrics: Mapping[str, float],
) -> np.ndarray:
    """Build the per-frame feature vector.

    ``pts`` is the raw ``(33, 4)`` landmark array; ``metrics`` is the output of
    :func:`~src.angle_engine.compute_metrics` for the same frame (passed in
    rather than recomputed, since the pipeline already has it).
    """
    xy = aspect_correct(pts, image_size)
    hip = 0.5 * (xy[LM.LEFT_HIP] + xy[LM.RIGHT_HIP])
    sho = 0.5 * (xy[LM.LEFT_SHOULDER] + xy[LM.RIGHT_SHOULDER])
    torso = float(np.linalg.norm(sho - hip))
    if torso < 1e-6:
        coords = np.full(2 * len(COORD_LANDMARKS), np.nan, dtype=np.float32)
    else:
        coords = ((xy[list(COORD_LANDMARKS)] - hip) / torso).astype(np.float32).ravel()
    return np.concatenate([metrics_to_vector(metrics, CLASSIFIER_METRICS), coords])


def aggregate_window(frames: np.ndarray) -> np.ndarray:
    """Reduce a ``(T, N_FRAME_FEATURES)`` window to a single feature vector.

    NaN-aware: a channel that was never observed in the window becomes NaN and
    is imputed by the training pipeline, rather than silently becoming 0.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[0] == 0:
        return np.full(N_WINDOW_FEATURES, np.nan, dtype=np.float32)

    # An all-NaN channel is legitimate (a metric that is undefined for this
    # camera view). NaN out is the right answer; the training pipeline's imputer
    # handles it, so silence numpy's warnings rather than masking the values.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(frames, axis=0)
        std = np.nanstd(frames, axis=0)
        mn = np.nanmin(frames, axis=0)
        mx = np.nanmax(frames, axis=0)
    first = _first_finite(frames)
    last = _first_finite(frames[::-1])
    drift = last - first
    return np.concatenate([mean, std, mn, mx, drift]).astype(np.float32)


def _first_finite(frames: np.ndarray) -> np.ndarray:
    """First finite value per channel (NaN if the channel is all-NaN)."""
    out = np.full(frames.shape[1], np.nan, dtype=np.float32)
    finite = np.isfinite(frames)
    has = finite.any(axis=0)
    idx = np.argmax(finite, axis=0)
    cols = np.nonzero(has)[0]
    out[cols] = frames[idx[cols], cols]
    return out


class WindowFeaturizer:
    """Rolling window used at inference time."""

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        self.window = int(window)
        self._buf: deque[np.ndarray] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._buf.clear()

    @property
    def filled(self) -> int:
        return len(self._buf)

    @property
    def ready(self) -> bool:
        # 60% of a window is enough for a usable prediction and gets the first
        # label on screen inside ~1 s instead of ~1.5 s.
        return len(self._buf) >= max(8, int(0.6 * self.window))

    def push(self, frame_vec: np.ndarray) -> None:
        self._buf.append(np.asarray(frame_vec, dtype=np.float32))

    def features(self) -> np.ndarray | None:
        if not self.ready:
            return None
        return aggregate_window(np.vstack(self._buf))


def windows_from_sequence(
    frames: np.ndarray,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    min_fill: float = 1.0,
) -> np.ndarray:
    """Slice a ``(T, N_FRAME_FEATURES)`` recording into window feature vectors.

    Returns ``(n_windows, N_WINDOW_FEATURES)``. Empty array if the recording is
    shorter than ``window * min_fill``.
    """
    frames = np.asarray(frames, dtype=np.float32)
    need = int(window * min_fill)
    if frames.shape[0] < need:
        return np.empty((0, N_WINDOW_FEATURES), dtype=np.float32)
    out = [
        aggregate_window(frames[start:start + window])
        for start in range(0, frames.shape[0] - window + 1, max(1, stride))
    ]
    return np.vstack(out) if out else np.empty((0, N_WINDOW_FEATURES), dtype=np.float32)


def stack_frame_features(vectors: Sequence[Iterable[float]]) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float32)
