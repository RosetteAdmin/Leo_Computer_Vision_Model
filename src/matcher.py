"""Training-free exercise recognition straight from the YAML configs.

The trained classifier in :mod:`src.classifier` needs labelled recordings of the
exact framing it will see. That is the right long-term answer, but it fails in two
common situations:

* you have no training data yet for a newly added exercise, and
* the camera shows only part of the body, so most of its features are missing.

:class:`ConfigMatcher` covers both. It uses information the configs already
contain - each exercise's ``primary_metric`` and its ``near``/``far`` thresholds -
and asks a simple question of the recent motion: *which exercise's rep signal does
this movement actually trace out?*

Three factors, multiplied:

``coverage``
    How much of the configured ``near -> far`` span the observed range covers.
    A movement that never approaches an exercise's working position is not it.
``fit``
    How much of the observed motion is *explained* by that span. A curl swings
    the elbow ~125 degrees; matching it against the push-up's narrower 50-degree
    window leaves most of the motion unexplained, so the curl wins.
``signature``
    Postural gates from the config (``signature:`` block). This is what separates
    exercises that share a joint: an elbow bending 90 degrees is a push-up if the
    torso is horizontal and a curl if it is upright.

Unmeasurable metrics are handled explicitly rather than silently: an exercise
whose primary metric cannot be seen is never proposed, and a signature that
cannot be evaluated yields a *partial* score, so the match is reported with lower
confidence instead of being asserted or discarded.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from .exercise_config import ExerciseConfig, ExerciseLibrary

#: Score given to a signature condition whose metric is not measurable, e.g.
#: `torso_lean` when the hips are out of frame. Deliberately mid-range: it means
#: "cannot confirm or deny", which is different from both pass and fail.
UNKNOWN_SIGNATURE_SCORE = 0.5


@dataclass
class MatchResult:
    exercise: Optional[str]
    score: float
    coverage: float = 0.0
    fit: float = 0.0
    signature: float = 0.0
    ambiguous_with: tuple[str, ...] = ()
    reason: str = ""

    @property
    def confident(self) -> bool:
        return self.exercise is not None and not self.ambiguous_with


class ConfigMatcher:
    """Recognise an exercise from recent motion, using configs only.

    Parameters
    ----------
    history_seconds:
        How much recent motion to consider. Needs to span at least one full
        repetition; 5 s covers a slow squat.
    min_score:
        Below this the movement is reported as unrecognised rather than guessed.
    min_samples:
        Minimum finite samples of a primary metric before that exercise is even
        considered.
    """

    def __init__(
        self,
        library: ExerciseLibrary,
        history_seconds: float = 5.0,
        min_score: float = 0.45,
        min_samples: int = 12,
        switch_frames: int = 10,
    ) -> None:
        self.library = library
        self.history_seconds = float(history_seconds)
        self.min_score = float(min_score)
        self.min_samples = int(min_samples)
        self.switch_frames = max(1, int(switch_frames))
        self._current: Optional[str] = None
        self._pending: Optional[str] = None
        self._pending_count = 0
        self._candidates = tuple(
            c for c in library.values() if c.rep_counter is not None
        )
        self._metrics = sorted({c.rep_counter.primary_metric for c in self._candidates}
                               | {g.metric for c in self._candidates for g in c.signature})
        self._buf: deque[tuple[float, dict[str, float]]] = deque(maxlen=600)
        self.last: MatchResult = MatchResult(None, 0.0, reason="no data yet")

    def reset(self) -> None:
        self._buf.clear()
        self._current = self._pending = None
        self._pending_count = 0
        self.last = MatchResult(None, 0.0, reason="no data yet")

    # -- scoring --------------------------------------------------------------
    @staticmethod
    def _span_overlap(lo: float, hi: float, a: float, b: float) -> float:
        """Length of the overlap between observed [lo, hi] and configured [a, b]."""
        c, d = (a, b) if a <= b else (b, a)
        return max(0.0, min(hi, d) - max(lo, c))

    _PERCENTILE = {"peak": 85.0, "trough": 15.0, "median": 50.0}

    def _statistic(self, metric: str, over: str) -> float:
        """A percentile of ``metric`` over the window, or NaN if unmeasurable.

        Signatures MUST be judged over the whole movement, not on the current
        frame. Sampled instantaneously, a shoulder press looks exactly like a
        bicep curl during the part of the rep where the arms are down, and a
        lunge looks like a squat while the feet are still together - so the label
        flips back and forth and neither rep counter ever completes a cycle.

        ``over`` selects which end of the motion the condition is about; see
        :class:`~src.exercise_config.Gate`.
        """
        vals = np.array([m.get(metric, np.nan) for _, m in self._buf], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size < self.min_samples:
            return float("nan")
        return float(np.percentile(finite, self._PERCENTILE.get(over, 85.0)))

    def _signature_score(self, cfg: ExerciseConfig) -> float:
        if not cfg.signature:
            return 1.0
        total = 0.0
        for gate in cfg.signature:
            v = self._statistic(gate.metric, gate.over)
            if v != v:
                total += UNKNOWN_SIGNATURE_SCORE
            elif gate.passes({gate.metric: v}):
                total += 1.0
            # a measurable condition that fails contributes 0
        return total / len(cfg.signature)

    def _score(self, cfg: ExerciseConfig) -> MatchResult:
        rc = cfg.rep_counter
        assert rc is not None
        series = np.array(
            [m.get(rc.primary_metric, np.nan) for _, m in self._buf], dtype=np.float64
        )
        finite = series[np.isfinite(series)]
        if finite.size < self.min_samples:
            return MatchResult(cfg.name, 0.0, reason=f"{rc.primary_metric} not measurable")

        lo, hi = (float(x) for x in np.percentile(finite, [4, 96]))
        observed = hi - lo
        span = abs(rc.far_threshold - rc.near_threshold)
        if span < 1e-6 or observed < 1e-6:
            return MatchResult(cfg.name, 0.0, reason="no movement")

        overlap = self._span_overlap(lo, hi, rc.near_threshold, rc.far_threshold)
        coverage = min(1.0, overlap / span)
        fit = overlap / observed          # how much of the motion this span explains
        sig = self._signature_score(cfg)
        return MatchResult(cfg.name, coverage * fit * sig, coverage, fit, sig)

    # -- main entry point -----------------------------------------------------
    def update(self, metrics: Mapping[str, float], timestamp: float) -> MatchResult:
        self._buf.append((float(timestamp), {k: metrics.get(k, np.nan) for k in self._metrics}))
        cutoff = float(timestamp) - self.history_seconds
        while len(self._buf) > 2 and self._buf[0][0] < cutoff:
            self._buf.popleft()

        results = [self._score(c) for c in self._candidates]
        results.sort(key=lambda r: -r.score)

        proposed: Optional[str] = None
        top = results[0] if results else MatchResult(None, 0.0)
        rivals: tuple[str, ...] = ()
        if results and top.score >= self.min_score:
            proposed = top.exercise
            # Flag a near-tie rather than pretending to be sure. This is what
            # happens when the distinguishing posture is not visible: a curl and
            # a push-up look identical at the elbow if the torso is out of frame.
            rivals = tuple(r.exercise for r in results[1:]
                           if r.exercise and r.score > 0.85 * top.score)

        # --- stability: a new label must persist before it is adopted ---
        # Without this the label chatters between neighbours on every frame and
        # each switch strands a half-finished repetition in a different counter.
        if proposed == self._pending:
            self._pending_count += 1
        else:
            self._pending, self._pending_count = proposed, 1
        if self._pending_count >= self.switch_frames:
            self._current = proposed

        if self._current is None:
            self.last = MatchResult(None, top.score,
                                    reason="no exercise matches this movement")
        elif self._current == top.exercise:
            self.last = MatchResult(top.exercise, top.score, top.coverage, top.fit,
                                    top.signature, rivals,
                                    "ambiguous" if rivals else "matched")
        else:
            held = next((r for r in results if r.exercise == self._current), top)
            self.last = MatchResult(held.exercise, held.score, held.coverage, held.fit,
                                    held.signature, rivals, "matched")
        return self.last

    def debug_table(self, metrics: Mapping[str, float] | None = None) -> str:
        """Per-exercise scores - useful when a match looks wrong."""
        rows = [self._score(c) for c in self._candidates]
        rows.sort(key=lambda r: -r.score)
        out = [f"{'exercise':<16}{'score':>7}{'cover':>7}{'fit':>7}{'sig':>7}  reason"]
        for r in rows:
            out.append(f"{str(r.exercise):<16}{r.score:>7.2f}{r.coverage:>7.2f}"
                       f"{r.fit:>7.2f}{r.signature:>7.2f}  {r.reason}")
        return "\n".join(out)
