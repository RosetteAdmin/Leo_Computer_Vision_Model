"""Stage 4 - Exercise classifier with an explicit "unknown" path.

Model: ``SimpleImputer -> StandardScaler -> RandomForest`` (or SVM / XGBoost) on
the 285-dim window features from :mod:`src.features`. Small, fast, and trains in
seconds on a laptop, which matters because you will retrain it every time you
record more data.

The interesting part is *not* the classifier, it is the rejection logic. A
Random Forest asked to label a movement it has never seen will happily return
one of its known classes with high confidence. Two independent gates stop that:

1. **Confidence gate** - ``max(predict_proba) < min_confidence`` -> unknown.
2. **Novelty gate** - an :class:`~sklearn.ensemble.IsolationForest` fitted on
   the training features rejects out-of-distribution windows even when the
   forest is confident.

On top of that, :class:`LiveClassifier` requires a label to win a majority vote
over a short history before it is shown, so the on-screen label does not flicker
between neighbouring classes.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .features import DEFAULT_STRIDE, DEFAULT_WINDOW, WINDOW_FEATURE_NAMES

UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Outcome for a single window."""

    label: Optional[str]          # None => unknown / generic movement
    confidence: float
    raw_label: str
    novelty_ok: bool
    reason: str = ""

    @property
    def is_known(self) -> bool:
        return self.label is not None


@dataclass
class ClassifierBundle:
    """Everything needed to run inference, saved as a single ``.pkl``."""

    pipeline: Any
    labels: tuple[str, ...]
    feature_names: tuple[str, ...] = WINDOW_FEATURE_NAMES
    window: int = DEFAULT_WINDOW
    stride: int = DEFAULT_STRIDE
    min_confidence: float = 0.6
    novelty: Any = None
    #: Above this probability the novelty gate is overridden. Necessary because
    #: bad form is legitimately out-of-distribution relative to well-executed
    #: training reps; without the override the system stops recognising the
    #: exercise precisely when the user's form is worst.
    novelty_override_confidence: float = 0.85
    model_type: str = "random_forest"
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        side = path.with_suffix(".json")
        side.write_text(
            json.dumps(
                {
                    "labels": list(self.labels),
                    "window": self.window,
                    "stride": self.stride,
                    "min_confidence": self.min_confidence,
                    "model_type": self.model_type,
                    "n_features": len(self.feature_names),
                    "metadata": self.metadata,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def load(path: str | Path) -> "ClassifierBundle":
        import joblib

        bundle = joblib.load(Path(path))
        if not isinstance(bundle, ClassifierBundle):
            raise TypeError(f"{path} does not contain a ClassifierBundle")
        return bundle

    # -- inference ------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        return self.pipeline.predict_proba(X)

    def classify(self, x: np.ndarray, min_confidence: float | None = None) -> ClassificationResult:
        """Classify one window feature vector, with both rejection gates."""
        thr = self.min_confidence if min_confidence is None else min_confidence
        X = np.atleast_2d(np.asarray(x, dtype=np.float32))
        proba = self.pipeline.predict_proba(X)[0]
        classes = list(self.pipeline.classes_)
        best = int(np.argmax(proba))
        raw_label = str(classes[best])
        conf = float(proba[best])

        novelty_ok = True
        if self.novelty is not None:
            novelty_ok = bool(self.novelty.predict(X)[0] == 1)

        if conf < thr:
            return ClassificationResult(None, conf, raw_label, novelty_ok, "low confidence")
        if not novelty_ok and conf < self.novelty_override_confidence:
            return ClassificationResult(None, conf, raw_label, novelty_ok, "out of distribution")
        return ClassificationResult(raw_label, conf, raw_label, novelty_ok, "")


class LiveClassifier:
    """Rolling-vote wrapper for real-time use.

    Parameters
    ----------
    classify_every:
        Run the forest only every Nth frame. At 30 FPS, N=3 gives 10 decisions
        per second, which is far faster than a human changes exercise, and keeps
        the classifier off the critical path.
    vote_window:
        Number of recent decisions considered. A label needs a strict majority
        of this window to be adopted.
    """

    def __init__(
        self,
        bundle: ClassifierBundle,
        min_confidence: float | None = None,
        classify_every: int = 3,
        vote_window: int = 7,
    ) -> None:
        self.bundle = bundle
        self.min_confidence = bundle.min_confidence if min_confidence is None else min_confidence
        self.classify_every = max(1, int(classify_every))
        self.vote_window = max(1, int(vote_window))
        self._votes: deque[str] = deque(maxlen=self.vote_window)
        self._frames = 0
        self.last_result: ClassificationResult | None = None
        self.label: Optional[str] = None
        self.confidence: float = 0.0

    def reset(self) -> None:
        self._votes.clear()
        self._frames = 0
        self.last_result = None
        self.label = None
        self.confidence = 0.0

    def update(self, window_features: np.ndarray | None) -> Optional[str]:
        """Feed the current window features. Returns the smoothed label."""
        self._frames += 1
        if window_features is None:
            return self.label
        if self._frames % self.classify_every != 0 and self.last_result is not None:
            return self.label

        res = self.bundle.classify(window_features, self.min_confidence)
        self.last_result = res
        self._votes.append(res.label or UNKNOWN)

        winner, votes = Counter(self._votes).most_common(1)[0]
        if votes * 2 > len(self._votes):
            new_label = None if winner == UNKNOWN else winner
            if new_label != self.label:
                self.label = new_label
        self.confidence = res.confidence
        return self.label
