"""Session recording / loading (Stage 1 "record mode", Stage 3 dataset input).

A session is one ``.npz`` file holding the *whole-body* time series for a clip:
raw landmarks, every metric from the angle engine, the per-frame classifier
features, and timestamps - plus a JSON metadata blob (label, fault label,
subject, source, fps, image size).

Recording landmarks as well as derived values matters: if the angle engine
changes, old recordings can be reprocessed instead of re-filmed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .angle_engine import METRIC_NAMES, compute_metrics, metrics_to_vector
from .features import FRAME_FEATURE_NAMES, frame_features

SESSION_SUFFIX = ".npz"


@dataclass
class Session:
    """One recorded clip."""

    landmarks: np.ndarray        # (T, 33, 4)
    metrics: np.ndarray          # (T, len(METRIC_NAMES))
    features: np.ndarray         # (T, len(FRAME_FEATURE_NAMES))
    timestamps: np.ndarray       # (T,)
    meta: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    # -- properties -----------------------------------------------------------
    @property
    def label(self) -> str:
        return str(self.meta.get("label", "unknown"))

    @property
    def fault(self) -> str:
        """Intentional-fault tag, e.g. ``"good"``, ``"shallow"``, ``"valgus"``."""
        return str(self.meta.get("fault", "good"))

    @property
    def expected_reps(self) -> int | None:
        v = self.meta.get("expected_reps")
        return None if v is None else int(v)

    @property
    def expected_codes(self) -> tuple[str, ...]:
        return tuple(self.meta.get("expected_codes", ()) or ())

    @property
    def image_size(self) -> tuple[int, int]:
        w, h = self.meta.get("image_size", (0, 0))
        return int(w), int(h)

    @property
    def fps(self) -> float:
        if self.timestamps.size > 1:
            dt = float(np.median(np.diff(self.timestamps)))
            if dt > 1e-6:
                return 1.0 / dt
        return float(self.meta.get("fps", 30.0))

    @property
    def n_frames(self) -> int:
        return int(self.landmarks.shape[0])

    @property
    def subject(self) -> str:
        return str(self.meta.get("subject", "unknown"))

    def metric_series(self, name: str) -> np.ndarray:
        return self.metrics[:, METRIC_NAMES.index(name)]

    def metrics_at(self, i: int) -> dict[str, float]:
        return {n: float(self.metrics[i, j]) for j, n in enumerate(METRIC_NAMES)}

    def iter_frames(self) -> Iterator[tuple[int, float, dict[str, float]]]:
        for i in range(self.n_frames):
            yield i, float(self.timestamps[i]), self.metrics_at(i)

    # -- io -------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            landmarks=self.landmarks.astype(np.float32),
            metrics=self.metrics.astype(np.float32),
            features=self.features.astype(np.float32),
            timestamps=self.timestamps.astype(np.float64),
            metric_names=np.array(METRIC_NAMES),
            feature_names=np.array(FRAME_FEATURE_NAMES),
            meta=np.array(json.dumps(self.meta)),
        )
        self.path = path
        return path

    @staticmethod
    def load(path: str | Path) -> "Session":
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"].item())) if "meta" in z else {}
            stored = tuple(str(x) for x in z["metric_names"]) if "metric_names" in z else METRIC_NAMES
            metrics = z["metrics"]
            if stored != METRIC_NAMES:
                metrics = _remap_columns(metrics, stored, METRIC_NAMES)
            sess = Session(
                landmarks=z["landmarks"],
                metrics=metrics,
                features=z["features"],
                timestamps=z["timestamps"],
                meta=meta,
                path=path,
            )
        return sess

    def to_csv(self, path: str | Path) -> Path:
        """Human-inspectable dump of the metric time series."""
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "time"] + list(METRIC_NAMES))
            for i in range(self.n_frames):
                w.writerow([i, f"{self.timestamps[i]:.4f}"] + [f"{v:.4f}" for v in self.metrics[i]])
        return path


def _remap_columns(arr: np.ndarray, have: Sequence[str], want: Sequence[str]) -> np.ndarray:
    """Reorder/pad metric columns when a recording predates a metric change."""
    index = {n: i for i, n in enumerate(have)}
    out = np.full((arr.shape[0], len(want)), np.nan, dtype=np.float32)
    for j, name in enumerate(want):
        if name in index:
            out[:, j] = arr[:, index[name]]
    return out


class SessionRecorder:
    """Accumulates frames during a live/offline capture, then writes a Session."""

    def __init__(self, label: str, **meta: Any) -> None:
        self.meta: dict[str, Any] = {
            "label": label,
            "fault": "good",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.meta.update({k: v for k, v in meta.items() if v is not None})
        self._landmarks: list[np.ndarray] = []
        self._metrics: list[np.ndarray] = []
        self._features: list[np.ndarray] = []
        self._times: list[float] = []

    def __len__(self) -> int:
        return len(self._times)

    def add(
        self,
        pts: np.ndarray,
        metrics: Mapping[str, float],
        features: np.ndarray,
        timestamp: float,
    ) -> None:
        self._landmarks.append(np.asarray(pts, dtype=np.float32))
        self._metrics.append(metrics_to_vector(metrics, METRIC_NAMES))
        self._features.append(np.asarray(features, dtype=np.float32))
        self._times.append(float(timestamp))

    def build(self) -> Session:
        if not self._times:
            raise ValueError("nothing recorded")
        t = np.asarray(self._times, dtype=np.float64)
        t -= t[0]
        return Session(
            landmarks=np.stack(self._landmarks),
            metrics=np.vstack(self._metrics),
            features=np.vstack(self._features),
            timestamps=t,
            meta=dict(self.meta),
        )

    def save(self, directory: str | Path, name: str | None = None) -> Path:
        session = self.build()
        directory = Path(directory)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        fname = name or f"{session.label}_{session.fault}_{stamp}{SESSION_SUFFIX}"
        return session.save(directory / fname)


def session_from_landmarks(
    landmarks: np.ndarray,
    timestamps: np.ndarray | None = None,
    image_size: tuple[int, int] = (0, 0),
    **meta: Any,
) -> Session:
    """Build a Session from a landmark array by running the angle engine.

    Used by the synthetic generator and by ``tools/process_video.py``; keeps a
    single code path for "landmarks in -> session out".
    """
    landmarks = np.asarray(landmarks, dtype=np.float32)
    n = landmarks.shape[0]
    if timestamps is None:
        timestamps = np.arange(n, dtype=np.float64) / float(meta.get("fps", 30.0))
    rec = SessionRecorder(str(meta.pop("label", "unknown")), image_size=list(image_size), **meta)
    for i in range(n):
        m = compute_metrics(landmarks[i], image_size)
        f = frame_features(landmarks[i], image_size, m)
        rec.add(landmarks[i], m, f, float(timestamps[i]))
    return rec.build()


def iter_sessions(directory: str | Path, pattern: str = f"*{SESSION_SUFFIX}") -> Iterator[Session]:
    for p in sorted(Path(directory).rglob(pattern)):
        yield Session.load(p)


def load_sessions(directory: str | Path, pattern: str = f"*{SESSION_SUFFIX}") -> list[Session]:
    return list(iter_sessions(directory, pattern))


def summarise(sessions: Iterable[Session]) -> str:
    from collections import Counter

    sessions = list(sessions)
    by_label = Counter(s.label for s in sessions)
    frames = sum(s.n_frames for s in sessions)
    lines = [f"{len(sessions)} sessions, {frames} frames total"]
    for label, n in sorted(by_label.items()):
        f = sum(s.n_frames for s in sessions if s.label == label)
        lines.append(f"  {label:<18} {n:>3} clips  {f:>6} frames")
    return "\n".join(lines)
