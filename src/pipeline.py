"""Stage 7 - the whole pipeline, wired together.

One class, :class:`MonitorPipeline`, owns the per-frame flow::

    landmarks -> angle engine -> smoothing -> classifier (or forced/generic)
              -> rep counter (per-exercise state kept separately)
              -> mistake detector (per-frame + per-rep)
              -> FrameState

``main.py`` drives it from a webcam and ``evaluate.py`` drives it from recorded
sessions, so the numbers in the metrics report come from exactly the code that
runs live - no separate offline reimplementation to drift out of sync.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

from .angle_engine import (DEFAULT_SMOOTHING_SECONDS, METRIC_NAMES, MetricSmoother,
                           compute_metrics)
from .classifier import ClassifierBundle, LiveClassifier
from .exercise_config import ExerciseConfig, ExerciseLibrary
from .features import WindowFeaturizer, frame_features
from .matcher import ConfigMatcher, MatchResult
from .mistake_detector import Mistake, MistakeDetector
from .rep_counter import GenericRepCounter, Rep, RepCounter

GENERIC_LABEL = "generic movement"


@dataclass
class FrameState:
    """Everything the renderer (or an evaluator) needs about one frame."""

    frame_index: int
    timestamp: float
    detected: bool
    metrics: dict[str, float] = field(default_factory=dict)
    raw_metrics: dict[str, float] = field(default_factory=dict)
    exercise: Optional[str] = None          # None => generic mode
    display_name: str = GENERIC_LABEL
    confidence: float = 0.0
    forced: bool = False
    #: How the label was decided: "forced", "model" (trained classifier),
    #: "matched" / "ambiguous" (ConfigMatcher), or "generic" (nothing matched).
    #: Shown on screen so a guess is never mistaken for a confident recognition.
    label_source: str = "generic"
    match: Optional[MatchResult] = None
    phase: str = "ready"
    primary_metric: Optional[str] = None
    rep_count: int = 0
    new_rep: Optional[Rep] = None
    new_mistakes: list[Mistake] = field(default_factory=list)
    active_mistakes: list[Mistake] = field(default_factory=list)
    view_ok: bool = True
    fps: float = 0.0
    landmarks: Optional[np.ndarray] = None

    @property
    def generic(self) -> bool:
        return self.exercise is None


class MonitorPipeline:
    """Stateful, single-person movement monitor.

    Parameters
    ----------
    library:
        Loaded exercise configs.
    bundle:
        Trained classifier. ``None`` runs everything in generic mode unless an
        exercise is forced, which is a perfectly usable configuration - joint
        tracking, generic cycle counting and anomaly checks all still work.
    forced_exercise:
        Skip classification and always use this config. Useful for recording
        clean training clips and for users who just want to drill one movement.
    """

    def __init__(
        self,
        library: ExerciseLibrary,
        bundle: ClassifierBundle | None = None,
        forced_exercise: str | None = None,
        smoothing_seconds: float = DEFAULT_SMOOTHING_SECONDS,
        min_confidence: float | None = None,
        classify_every: int = 3,
        vote_window: int = 7,
        use_config_matcher: bool = True,
    ) -> None:
        self.library = library
        if forced_exercise is not None and forced_exercise not in library:
            raise KeyError(
                f"no config for forced exercise '{forced_exercise}'. "
                f"Available: {', '.join(library.names)}"
            )
        self.forced_exercise = forced_exercise
        self.smoother = MetricSmoother(window_seconds=smoothing_seconds)
        self.featurizer = WindowFeaturizer(bundle.window if bundle else 45)
        self.classifier = (
            LiveClassifier(bundle, min_confidence, classify_every, vote_window)
            if bundle is not None and forced_exercise is None
            else None
        )
        # Training-free fallback. Runs whenever the trained model is absent or
        # unsure, which is the normal case for an exercise you have not recorded
        # yet or a camera framing the model never saw.
        self.matcher = (
            ConfigMatcher(library) if (use_config_matcher and forced_exercise is None) else None
        )
        self.detector = MistakeDetector()
        self.generic_counter = GenericRepCounter()
        self._counters: dict[str, RepCounter] = {}
        # Short rolling window of recent smoothed metrics, used to seed a rep
        # counter's resting history when it is created mid-session. Holds
        # references to the smoother's per-frame dicts, so it copies nothing.
        self._recent: deque[tuple[float, Mapping[str, float]]] = deque(maxlen=120)
        self.current_exercise: Optional[str] = forced_exercise
        self._fps_times: deque[float] = deque(maxlen=30)
        self.frames_seen = 0
        self.frames_detected = 0

    # -- helpers --------------------------------------------------------------
    def counter_for(self, name: str) -> RepCounter:
        if name not in self._counters:
            cfg = self.library[name]
            if cfg.rep_counter is None:
                raise ValueError(f"exercise '{name}' has no rep_counter section")
            counter = RepCounter(cfg.rep_counter, exercise=name)
            # Hand the new counter the recent past. It is created the moment the
            # exercise is first recognised, which is typically part-way into the
            # first repetition - the classifier needs a window of frames to decide.
            # A counter starting blind has no resting position to measure the
            # movement against and drops that first rep.
            metric = cfg.rep_counter.primary_metric
            for ts, metrics in self._recent:
                counter.observe_rest(ts, metrics.get(metric, float("nan")))
            self._counters[name] = counter
        return self._counters[name]

    def rep_totals(self) -> dict[str, int]:
        totals = {n: c.count for n, c in self._counters.items() if c.count}
        if self.generic_counter.count:
            totals["generic"] = self.generic_counter.count
        return totals

    def reset(self) -> None:
        self.smoother.reset()
        self.featurizer.reset()
        if self.classifier:
            self.classifier.reset()
        self.detector.reset()
        self.generic_counter.reset()
        for c in self._counters.values():
            c.reset()
        self._recent.clear()
        self.current_exercise = self.forced_exercise

    def _fps(self, timestamp: float) -> float:
        self._fps_times.append(timestamp)
        if len(self._fps_times) < 2:
            return 0.0
        span = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / span if span > 1e-6 else 0.0

    # -- main entry points ----------------------------------------------------
    def process_landmarks(
        self,
        pts: Optional[np.ndarray],
        image_size: tuple[int, int],
        frame_index: int,
        timestamp: float,
    ) -> FrameState:
        """Full path from raw landmarks (live camera or video)."""
        if pts is None:
            self.frames_seen += 1
            return FrameState(frame_index, timestamp, False, fps=self._fps(timestamp))
        raw = compute_metrics(pts, image_size)
        feats = frame_features(pts, image_size, raw)
        state = self.process_metrics(raw, feats, frame_index, timestamp)
        state.landmarks = pts
        return state

    def process_metrics(
        self,
        raw_metrics: Mapping[str, float],
        frame_feature_vec: np.ndarray | None,
        frame_index: int,
        timestamp: float,
    ) -> FrameState:
        """Path from precomputed metrics (recorded session replay / evaluation)."""
        self.frames_seen += 1
        self.frames_detected += 1
        metrics = self.smoother.update(raw_metrics, timestamp)
        fps = self._fps(timestamp)

        self._recent.append((timestamp, metrics))
        while (self._recent
               and timestamp - self._recent[0][0] > RepCounter.REST_WINDOW_SECONDS):
            self._recent.popleft()

        # --- 1. which exercise? ---
        label: Optional[str] = self.forced_exercise
        confidence = 1.0 if self.forced_exercise else 0.0
        source = "forced" if self.forced_exercise else "generic"
        match: MatchResult | None = None

        if self.forced_exercise is None:
            if self.classifier is not None:
                if frame_feature_vec is not None:
                    self.featurizer.push(frame_feature_vec)
                label = self.classifier.update(self.featurizer.features())
                confidence = self.classifier.confidence
                # A label with no config cannot be acted on; treat it as unknown.
                if label is not None and label not in self.library:
                    label = None
                if label is not None:
                    source = "model"

            if self.matcher is not None:
                match = self.matcher.update(metrics, timestamp)
                if label is None and match.exercise is not None:
                    label = match.exercise
                    confidence = match.score
                    source = "matched" if match.confident else "ambiguous"

        if label != self.current_exercise:
            self.detector.clear_active()
            self.current_exercise = label

        config: Optional[ExerciseConfig] = self.library[label] if label else None

        # --- 2. camera view sanity ---
        view_ok = config.view_ok(metrics) if config else True

        # --- 3. reps ---
        if config is not None and config.rep_counter is not None:
            counter = self.counter_for(config.name)
            new_rep = counter.update(metrics, frame_index, timestamp)
            phase = counter.state
            primary = counter.primary_metric
            rep_count = counter.count
        else:
            new_rep = self.generic_counter.update(metrics, frame_index, timestamp)
            phase = self.generic_counter.state
            primary = self.generic_counter.primary_metric
            rep_count = self.generic_counter.count

        # --- 4. mistakes ---
        new_mistakes = self.detector.check_frame(metrics, phase, config, frame_index, timestamp)
        if config is not None and not view_ok:
            got = self.detector._raise(  # noqa: SLF001 - same package, deliberate
                Mistake(
                    code="SUBOPTIMAL_CAMERA_VIEW",
                    scope="frame",
                    severity="info",
                    metric="view_frontality",
                    value=metrics.get("view_frontality"),
                    exercise=config.name,
                    frame_index=frame_index,
                    detail="stand at ~45 deg to the camera",
                )
            )
            if got:
                new_mistakes.append(got)
        if new_rep is not None:
            new_mistakes += self.detector.check_rep(new_rep, config, skip_rom=not view_ok)

        return FrameState(
            frame_index=frame_index,
            timestamp=timestamp,
            detected=True,
            metrics=metrics,
            raw_metrics=dict(raw_metrics),
            exercise=label,
            display_name=config.display_name if config else GENERIC_LABEL,
            confidence=confidence,
            forced=self.forced_exercise is not None,
            label_source=source,
            match=match,
            phase=phase,
            primary_metric=primary,
            rep_count=rep_count,
            new_rep=new_rep,
            new_mistakes=new_mistakes,
            active_mistakes=self.detector.active(frame_index),
            view_ok=view_ok,
            fps=fps,
        )

    # -- offline replay -------------------------------------------------------
    def replay(self, session, use_stored_features: bool = True) -> list[FrameState]:
        """Run a recorded :class:`~src.session.Session` through the pipeline.

        Same code path as the live loop, so evaluation measures the shipped
        behaviour. ``use_stored_features=False`` recomputes metrics from the
        stored landmarks instead (use it after changing the angle engine).
        """
        states: list[FrameState] = []
        for i in range(session.n_frames):
            ts = float(session.timestamps[i])
            if use_stored_features:
                states.append(
                    self.process_metrics(session.metrics_at(i), session.features[i], i, ts)
                )
            else:
                states.append(
                    self.process_landmarks(session.landmarks[i], session.image_size, i, ts)
                )
        return states

    # -- session summary ------------------------------------------------------
    def summary(self) -> dict[str, object]:
        return {
            "frames": self.frames_seen,
            "frames_with_pose": self.frames_detected,
            "reps": self.rep_totals(),
            "mistakes": self.detector.summary(),
        }
