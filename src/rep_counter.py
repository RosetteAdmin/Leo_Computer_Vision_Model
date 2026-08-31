"""Stage 5 - Repetition counting.

Two counters, one state machine:

* :class:`RepCounter` - config-driven. Uses the exercise's ``primary_metric``
  and its near/far thresholds.
* :class:`GenericRepCounter` - for movements the classifier does not recognise.
  It watches every metric in
  :data:`~src.angle_engine.GENERIC_TRACK_METRICS`, picks the one with the
  clearest periodic motion, derives near/far thresholds from that signal's own
  rolling min/max, and counts cycles with the identical state machine.

State machine (Section 9)::

    READY --> MOVING_AWAY --> EXTREME --> RETURNING --> READY (+1 rep)

Two details that matter in practice and are easy to get wrong:

1. **Direction independence.** A squat's knee angle *decreases* toward the
   working position, a shoulder press's abduction *increases*. The counter works
   on ``u = direction * value`` so ``u`` always grows toward the working
   position, and one set of comparisons covers both.
2. **Shallow reps still count.** A rep that peaks past ``count_ratio`` of the
   way to ``far_threshold`` but never reaches it is still counted and returned,
   so the mistake detector can flag it as ``PARTIAL_REP``. Silently ignoring
   short reps would make the counter look broken to anyone with sloppy form.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

from .angle_engine import GENERIC_TRACK_METRICS
from .exercise_config import RepCounterConfig

READY = "ready"
MOVING_AWAY = "moving_away"
EXTREME = "extreme"
RETURNING = "returning"

#: Metric-family scale factors so that angle metrics (degrees) and normalised
#: distance metrics (torso lengths) are comparable when generic mode ranks
#: candidate signals by amplitude.
_GENERIC_SCALE: dict[str, float] = {
    "hip_height": 90.0,
    "shoulder_height": 90.0,
    "wrist_height_mean": 90.0,
    "stance_width": 90.0,
}


@dataclass
class Rep:
    """One completed repetition, with everything the per-rep checks need."""

    index: int
    exercise: str
    metric: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    curve: np.ndarray                      # primary metric over the rep
    start_value: float
    extreme_value: float
    extreme_frame: int
    reached_target_zone: bool              # crossed far_threshold (full rep)
    metrics_at_extreme: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(self.end_time - self.start_time, 1e-6)

    @property
    def rom(self) -> float:
        """Total swing of the primary metric within the rep."""
        if self.curve.size == 0:
            return 0.0
        return float(np.nanmax(self.curve) - np.nanmin(self.curve))

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


class RepCounter:
    """Config-driven hysteresis state machine."""

    #: How much recent history, in seconds, is used to estimate where the body
    #: was resting before a repetition began.
    REST_WINDOW_SECONDS = 0.6

    def __init__(self, config: RepCounterConfig, exercise: str = "") -> None:
        self.config = config
        self.exercise = exercise
        self.count = 0
        self.reps: list[Rep] = []
        # (timestamp, value) samples observed while idle, used to estimate the
        # resting position. See _rest_estimate.
        self._rest_hist: deque[tuple[float, float]] = deque(maxlen=240)
        self._reset_cycle()
        self.state = READY

    # -- helpers --------------------------------------------------------------
    @property
    def primary_metric(self) -> str:
        return self.config.primary_metric

    @property
    def direction(self) -> int:
        return self.config.direction

    def _u(self, value: float) -> float:
        return self.direction * value

    def _reset_cycle(self) -> None:
        self.state = READY
        self._curve: list[float] = []
        self._start_frame = -1
        self._start_time = 0.0
        self._start_value = float("nan")
        self._rest_value = float("nan")
        self._peak_u = -float("inf")
        self._peak_frame = -1
        self._peak_metrics: dict[str, float] = {}
        self._reached_far = False

    def reset(self, keep_count: bool = False) -> None:
        if not keep_count:
            self.count = 0
            self.reps = []
        self._rest_hist.clear()
        self._reset_cycle()

    def observe_rest(self, timestamp: float, value: float) -> None:
        """Record one idle sample, exactly as :meth:`update` does while in READY.

        Lets a counter that was created part-way through a session inherit the
        resting history it would have gathered had it existed from the first frame.
        Without this, a counter created mid-movement has nothing to measure travel
        against and rejects a genuine repetition — measurably, the first rep of a
        set, because the classifier needs a window before it names the exercise.

        Samples already past the arming threshold are ignored: those describe the
        movement, not the rest before it. That distinction is what keeps a
        half-flexed hold (the bug in :meth:`_complete`) from being mistaken for a
        resting position just because it preceded a cycle.
        """
        if value != value:                      # NaN: nothing was measurable
            return
        u = self._u(value)
        if u > self._u(self.config.near_threshold) + abs(self.config.hysteresis):
            return
        self._rest_hist.append((timestamp, u))
        while (self._rest_hist
               and timestamp - self._rest_hist[0][0] > self.REST_WINDOW_SECONDS):
            self._rest_hist.popleft()

    def _rest_estimate(self, fallback: float) -> float:
        """Where the body was before this rep began: the least-worked idle sample.

        The threshold crossing that arms the counter is a poor stand-in. On a fast
        descent the signal can leap well past the threshold in a single frame, so
        the crossing understates how far the movement really travelled and the rep
        gets rejected - measurably, the first rep of a set.

        Taking the sample furthest from the working position answers "where did
        this movement start from", which is the question. A median of the idle
        window does not: during a quick descent most idle samples are already
        part-way down, so the median sits mid-movement and understates travel too.
        """
        if not self._rest_hist:
            return fallback
        return self.direction * min(u for _, u in self._rest_hist)

    @property
    def progress(self) -> float:
        """0..1 estimate of how far into the working range the user currently is."""
        cfg = self.config
        near_u, far_u = self._u(cfg.near_threshold), self._u(cfg.far_threshold)
        span = far_u - near_u
        if span <= 0 or self._peak_u == -float("inf"):
            return 0.0
        return float(np.clip((self._peak_u - near_u) / span, 0.0, 1.0))

    # -- main entry point -----------------------------------------------------
    def update(
        self,
        metrics: Mapping[str, float],
        frame_index: int,
        timestamp: float,
    ) -> Optional[Rep]:
        """Feed one frame. Returns a :class:`Rep` on the frame a rep completes."""
        cfg = self.config
        value = metrics.get(cfg.primary_metric, float("nan"))
        if value != value:  # NaN - pose lost; hold state rather than miscount
            return None

        u = self._u(value)
        near_u = self._u(cfg.near_threshold)
        far_u = self._u(cfg.far_threshold)
        h = abs(cfg.hysteresis)
        count_u = near_u + cfg.count_ratio * (far_u - near_u)

        if self.state != READY:
            self._curve.append(value)
            if u > self._peak_u:
                self._peak_u = u
                self._peak_frame = frame_index
                self._peak_metrics = dict(metrics)
            if timestamp - self._start_time > cfg.max_rep_seconds:
                self._reset_cycle()   # abandoned rep
                return None

        if self.state == READY:
            if u > near_u + h:
                self.state = MOVING_AWAY
                self._curve = [value]
                self._start_frame = frame_index
                self._start_time = timestamp
                self._start_value = value
                # Snapshot the resting position *before* this movement, while the
                # idle history still describes it.
                self._rest_value = self._rest_estimate(value)
                self._peak_u = u
                self._peak_frame = frame_index
                self._peak_metrics = dict(metrics)
                self._reached_far = False
            else:
                self._rest_hist.append((timestamp, u))
                while (self._rest_hist
                       and timestamp - self._rest_hist[0][0] > self.REST_WINDOW_SECONDS):
                    self._rest_hist.popleft()
            return None

        if self.state == MOVING_AWAY:
            if u >= far_u:
                self._reached_far = True
                self.state = EXTREME
            elif u <= near_u:
                self._reset_cycle()   # twitched and gave up: not a rep
            elif u < self._peak_u - h:
                self.state = RETURNING
            return None

        if self.state == EXTREME:
            if u < self._peak_u - h:
                self.state = RETURNING
            return None

        # RETURNING
        if u >= far_u:                    # bounced back into the working position
            self.state = EXTREME
            return None
        if u <= near_u + h:
            rep = self._complete(frame_index, timestamp, count_u)
            self._reset_cycle()
            return rep
        return None

    def _complete(self, frame_index: int, timestamp: float, count_u: float) -> Optional[Rep]:
        cfg = self.config
        # Duration, not frame count: identical behaviour at 11 FPS and 30 FPS.
        long_enough = (timestamp - self._start_time) >= cfg.min_rep_seconds
        # Did the signal get into the working zone at all?
        deep_enough = self._peak_u >= count_u
        # ...and did it actually *travel* there from where this cycle began?
        #
        # Depth alone is not a repetition. Holding the hands part-way up and then
        # jiggling scores as "deep" on an absolute scale while barely moving: a
        # real session held the elbow near 92 deg, dipped to 72 deg, and was
        # credited with a rep for 20 deg of movement, because 72 deg looks deep
        # against the configured 150 deg start. Measuring the excursion from the
        # resting position separates a performed rep from a wobble near the bottom.
        #
        # This does not suppress legitimate partial reps. Those are shallow at the
        # *top* - they never reach far_threshold - but still travel most of the way
        # from rest, so they are counted and then flagged as PARTIAL_REP.
        # Only demanded of reps that fell short of the working position. Reaching
        # far_threshold is strong evidence on its own, and insisting on travel
        # there costs real reps: the classifier needs a window to decide, so
        # end-to-end the counter is often created mid-descent with no idle history
        # to measure against, which silently dropped the first rep of every set.
        # Reps that fall short are exactly where wobbles live, so that is where the
        # extra evidence is required.
        if self._reached_far:
            moved_enough = True
        else:
            span = abs(cfg.far_threshold - cfg.near_threshold)
            rest = (self._rest_value if self._rest_value == self._rest_value
                    else self._start_value)
            travelled = abs(self.direction * self._peak_u - rest)
            moved_enough = travelled >= cfg.count_ratio * span
        if not (long_enough and deep_enough and moved_enough):
            return None
        self.count += 1
        rep = Rep(
            index=self.count,
            exercise=self.exercise,
            metric=self.config.primary_metric,
            start_frame=self._start_frame,
            end_frame=frame_index,
            start_time=self._start_time,
            end_time=timestamp,
            curve=np.asarray(self._curve, dtype=np.float32),
            start_value=self._start_value,
            extreme_value=self.direction * self._peak_u,
            extreme_frame=self._peak_frame,
            reached_target_zone=self._reached_far,
            metrics_at_extreme=self._peak_metrics,
        )
        self.reps.append(rep)
        return rep


class GenericRepCounter:
    """Cycle counter for unrecognised movements (Section 7c).

    Auto-selects the driving signal, auto-derives thresholds, then reuses
    :class:`RepCounter`. Selection is re-evaluated periodically; the running
    total survives a change of signal.
    """

    def __init__(
        self,
        candidates: tuple[str, ...] = GENERIC_TRACK_METRICS,
        history: int = 150,
        reselect_every: int = 20,
        min_amplitude: float = 18.0,
        min_periodicity: float = 0.30,
    ) -> None:
        self.candidates = candidates
        self.history = history
        self.reselect_every = reselect_every
        self.min_amplitude = min_amplitude
        self.min_periodicity = min_periodicity
        self._buf: dict[str, deque[float]] = {c: deque(maxlen=history) for c in candidates}
        self._frames = 0
        self.count = 0
        self.reps: list[Rep] = []
        self.selected: str | None = None
        self.amplitude = 0.0
        self._inner: RepCounter | None = None

    @property
    def state(self) -> str:
        return self._inner.state if self._inner else READY

    @property
    def primary_metric(self) -> str | None:
        return self.selected

    def reset(self) -> None:
        for d in self._buf.values():
            d.clear()
        self._frames = 0
        self.count = 0
        self.reps = []
        self.selected = None
        self._inner = None

    # -- signal selection -----------------------------------------------------
    @staticmethod
    def _periodicity(x: np.ndarray) -> float:
        """0..1 score: how strongly the signal repeats.

        Normalised autocorrelation peak over lags 8..len/2. Prefers genuine
        cyclic motion over slow drift or a one-off movement.
        """
        x = x[np.isfinite(x)]
        if x.size < 30:
            return 0.0
        x = x - x.mean()
        denom = float(np.dot(x, x))
        if denom < 1e-9:
            return 0.0
        ac = np.correlate(x, x, mode="full")[x.size - 1:] / denom
        lo, hi = 8, max(9, x.size // 2)
        seg = ac[lo:hi]
        return float(np.clip(seg.max(), 0.0, 1.0)) if seg.size else 0.0

    def _rank(self) -> tuple[str | None, float, float, float]:
        best: tuple[str | None, float, float, float] = (None, 0.0, 0.0, 0.0)
        best_score = -1.0
        for name, buf in self._buf.items():
            if len(buf) < 30:
                continue
            arr = np.asarray(buf, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if finite.size < 30:
                continue
            scale = _GENERIC_SCALE.get(name, 1.0)
            lo, hi = np.percentile(finite, [3, 97])
            amp = float(hi - lo) * scale
            if amp < self.min_amplitude:
                continue
            # Amplitude alone is not enough: landmark noise on a normalised
            # distance metric can look large once rescaled, and a slow drift
            # (person walking out of frame) has huge amplitude and no cycles.
            # Requiring real repetition stops both from being counted as reps.
            periodicity = self._periodicity(arr)
            if periodicity < self.min_periodicity:
                continue
            score = amp * (0.35 + 0.65 * periodicity)
            if score > best_score:
                best_score = score
                best = (name, amp, float(lo), float(hi))
        return best

    # -- main entry point -----------------------------------------------------
    def update(
        self,
        metrics: Mapping[str, float],
        frame_index: int,
        timestamp: float,
    ) -> Optional[Rep]:
        for name in self.candidates:
            self._buf[name].append(float(metrics.get(name, float("nan"))))
        self._frames += 1

        need_select = self._inner is None or (self._frames % self.reselect_every == 0)
        if need_select:
            name, amp, lo, hi = self._rank()
            self.amplitude = amp
            if name is None:
                self.selected = None
                self._inner = None
                return None
            rng = hi - lo
            cfg = RepCounterConfig(
                primary_metric=name,
                near_threshold=lo + 0.22 * rng,
                far_threshold=lo + 0.78 * rng,
                hysteresis=max(0.06 * rng, 1e-3),
                min_rep_seconds=0.27,
                max_rep_seconds=20.0,
                count_ratio=0.6,
            )
            if self._inner is None or self._inner.primary_metric != name:
                self._inner = RepCounter(cfg, exercise="generic")
            else:
                self._inner.config = cfg   # same signal, refreshed thresholds

        assert self._inner is not None
        rep = self._inner.update(metrics, frame_index, timestamp)
        self.selected = self._inner.primary_metric
        if rep is not None:
            self.count += 1
            rep.index = self.count
            self.reps.append(rep)
        return rep
