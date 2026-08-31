"""Stage 6 - Mistake detection layer.

Runs in parallel with classification and rep counting, off the same metric
stream. Three families of check:

7a **Angle mistakes** (per frame)
    A metric outside its configured range for the current movement phase.
    Violations must persist for ``min_frames`` before being raised, which stops
    single-frame BlazePose glitches from generating warnings.

7b **Repetition mistakes** (once per completed rep)
    Range of motion, tempo (absolute and relative to the user's own median),
    left/right symmetry at the extreme, and curve-shape consistency against the
    user's own running average rep.

7c **Generic anomalies** (per frame, no exercise config needed)
    Anatomically implausible joint angles, gross left/right asymmetry, and jerky
    motion. This is what runs for movements the classifier has never seen.

Every mistake is emitted as a structured :class:`Mistake` with a machine code;
turning codes into English lives in :mod:`src.feedback`.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .exercise_config import ExerciseConfig
from .rep_counter import Rep

# --- exercise-agnostic safety envelope ---------------------------------------
#: Plausible ranges for 2D-projected joint angles, in degrees. Deliberately
#: generous: these exist to catch "that cannot be a human joint" situations
#: (usually a tracking failure or a genuinely dangerous position), not to
#: coach form. In oblique camera views a projected angle can legitimately read
#: lower than the true anatomical angle, so tight limits would false-positive.
ANATOMICAL_LIMITS: dict[str, tuple[float, float]] = {
    "knee_l": (15.0, 192.0),
    "knee_r": (15.0, 192.0),
    "elbow_l": (15.0, 192.0),
    "elbow_r": (15.0, 192.0),
    "hip_l": (18.0, 196.0),
    "hip_r": (18.0, 196.0),
    "shoulder_l": (0.0, 186.0),
    "shoulder_r": (0.0, 186.0),
    "neck": (75.0, 192.0),
}

#: Absolute left/right differences above this (degrees) are called out in
#: generic mode.
GENERIC_ASYMMETRY_LIMIT = 35.0
GENERIC_ASYMMETRY_METRICS: tuple[str, ...] = ("knee_diff", "elbow_diff", "hip_diff", "shoulder_diff")

#: Angular speed (deg/s) above which motion is considered jerky.
GENERIC_JERK_LIMIT = 800.0
GENERIC_JERK_METRICS: tuple[str, ...] = ("knee_mean", "elbow_mean", "hip_mean", "shoulder_mean")

_CURVE_SAMPLES = 32


@dataclass
class Mistake:
    """One structured form error."""

    code: str
    scope: str                       # "frame" | "rep" | "generic"
    severity: str = "warning"        # "info" | "warning" | "error"
    metric: str | None = None
    value: float | None = None
    limit: float | None = None
    exercise: str | None = None
    frame_index: int = -1
    rep_index: int | None = None
    detail: str = ""
    highlight: tuple[str, ...] = ()

    def __str__(self) -> str:  # pragma: no cover - debug aid
        v = "-" if self.value is None else f"{self.value:.1f}"
        lim = "-" if self.limit is None else f"{self.limit:.1f}"
        return f"{self.code}({self.metric}={v} limit={lim})"


def _resample(curve: np.ndarray, n: int = _CURVE_SAMPLES) -> np.ndarray:
    """Resample a rep curve to a fixed length so curves can be compared."""
    curve = np.asarray(curve, dtype=np.float64)
    curve = curve[np.isfinite(curve)]
    if curve.size < 2:
        return np.full(n, np.nan)
    src = np.linspace(0.0, 1.0, curve.size)
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, curve)


class MistakeDetector:
    """Stateful detector. One instance per session.

    Parameters
    ----------
    hold_frames:
        How long a raised warning stays on screen after its last violation.
    rep_hold_frames:
        Same, for per-rep verdicts (kept longer so the user can read them).
    """

    def __init__(self, hold_frames: int = 20, rep_hold_frames: int = 60) -> None:
        self.hold_frames = hold_frames
        self.rep_hold_frames = rep_hold_frames
        self._streak: dict[str, int] = defaultdict(int)
        self._active: dict[str, tuple[Mistake, int, int]] = {}   # code -> (mistake, last_frame, hold)
        self.counts: dict[str, int] = defaultdict(int)
        self.history: list[Mistake] = []
        # per-exercise rep memory for tempo / consistency baselines
        self._durations: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=15))
        self._speeds: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=15))
        self._curves: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=10))
        # previous frame state for velocity / jerk
        self._prev_metrics: Mapping[str, float] | None = None
        self._prev_time: float | None = None

    # -- bookkeeping ----------------------------------------------------------
    def reset(self) -> None:
        self._streak.clear()
        self._active.clear()
        self.counts.clear()
        self.history.clear()
        self._durations.clear()
        self._speeds.clear()
        self._curves.clear()
        self._prev_metrics = None
        self._prev_time = None

    def _raise(self, m: Mistake, hold: int | None = None) -> Optional[Mistake]:
        """Register a mistake. Returns it only if it is newly active."""
        hold = self.hold_frames if hold is None else hold
        already = m.code in self._active
        self._active[m.code] = (m, m.frame_index, hold)
        if already:
            return None
        self.counts[m.code] += 1
        self.history.append(m)
        return m

    def active(self, frame_index: int) -> list[Mistake]:
        """Currently displayable warnings, most severe first."""
        expired = [c for c, (_, last, hold) in self._active.items() if frame_index - last > hold]
        for c in expired:
            del self._active[c]
        order = {"error": 0, "warning": 1, "info": 2}
        return sorted(
            (m for m, _, _ in self._active.values()),
            key=lambda m: (order.get(m.severity, 3), m.code),
        )

    def clear_active(self) -> None:
        self._active.clear()

    # -- 7a: per-frame angle mistakes ----------------------------------------
    def check_frame(
        self,
        metrics: Mapping[str, float],
        phase: str,
        config: Optional[ExerciseConfig],
        frame_index: int,
        timestamp: float,
    ) -> list[Mistake]:
        """Run angle checks for a known exercise, or generic checks if none."""
        raised: list[Mistake] = []

        if config is not None:
            for check in config.angle_checks:
                if not check.applies_to_phase(phase):
                    self._streak[check.code] = 0
                    continue
                if not all(g.passes(metrics) for g in check.gates):
                    self._streak[check.code] = 0
                    continue
                value = metrics.get(check.metric, float("nan"))
                if check.violated(value):
                    self._streak[check.code] += 1
                    if self._streak[check.code] >= check.min_frames:
                        limit = check.max if (check.max is not None and value > check.max) else check.min
                        m = Mistake(
                            code=check.code,
                            scope="frame",
                            severity=check.severity,
                            metric=check.metric,
                            value=float(value),
                            limit=limit,
                            exercise=config.name,
                            frame_index=frame_index,
                            highlight=check.highlight,
                        )
                        got = self._raise(m)
                        if got:
                            raised.append(got)
                else:
                    self._streak[check.code] = 0
        else:
            raised += self.check_generic(metrics, frame_index, timestamp)

        self._prev_metrics = dict(metrics)
        self._prev_time = timestamp
        return raised

    # -- 7c: generic anomalies -----------------------------------------------
    def check_generic(
        self,
        metrics: Mapping[str, float],
        frame_index: int,
        timestamp: float,
    ) -> list[Mistake]:
        """Exercise-agnostic checks: implausible angles, asymmetry, jerk."""
        raised: list[Mistake] = []

        for metric, (lo, hi) in ANATOMICAL_LIMITS.items():
            v = metrics.get(metric, float("nan"))
            if v != v:
                continue
            if v < lo or v > hi:
                key = f"EXTREME_JOINT_ANGLE::{metric}"
                self._streak[key] += 1
                if self._streak[key] >= 4:
                    got = self._raise(
                        Mistake(
                            code="EXTREME_JOINT_ANGLE",
                            scope="generic",
                            severity="error",
                            metric=metric,
                            value=float(v),
                            limit=lo if v < lo else hi,
                            frame_index=frame_index,
                            detail=metric,
                        )
                    )
                    if got:
                        raised.append(got)
            else:
                self._streak[f"EXTREME_JOINT_ANGLE::{metric}"] = 0

        worst_diff, worst_metric = 0.0, None
        for metric in GENERIC_ASYMMETRY_METRICS:
            v = metrics.get(metric, float("nan"))
            if v == v and v > worst_diff:
                worst_diff, worst_metric = float(v), metric
        if worst_metric and worst_diff > GENERIC_ASYMMETRY_LIMIT:
            self._streak["LARGE_ASYMMETRY"] += 1
            if self._streak["LARGE_ASYMMETRY"] >= 6:
                got = self._raise(
                    Mistake(
                        code="LARGE_ASYMMETRY",
                        scope="generic",
                        severity="warning",
                        metric=worst_metric,
                        value=worst_diff,
                        limit=GENERIC_ASYMMETRY_LIMIT,
                        frame_index=frame_index,
                        detail=worst_metric,
                    )
                )
                if got:
                    raised.append(got)
        else:
            self._streak["LARGE_ASYMMETRY"] = 0

        if self._prev_metrics is not None and self._prev_time is not None:
            dt = timestamp - self._prev_time
            if 1e-4 < dt < 0.5:
                worst_v, worst_m = 0.0, None
                for metric in GENERIC_JERK_METRICS:
                    a = metrics.get(metric, float("nan"))
                    b = self._prev_metrics.get(metric, float("nan"))
                    if a == a and b == b:
                        speed = abs(a - b) / dt
                        if speed > worst_v:
                            worst_v, worst_m = speed, metric
                if worst_m and worst_v > GENERIC_JERK_LIMIT:
                    self._streak["JERKY_MOTION"] += 1
                    if self._streak["JERKY_MOTION"] >= 3:
                        got = self._raise(
                            Mistake(
                                code="JERKY_MOTION",
                                scope="generic",
                                severity="warning",
                                metric=worst_m,
                                value=worst_v,
                                limit=GENERIC_JERK_LIMIT,
                                frame_index=frame_index,
                                detail=worst_m,
                            )
                        )
                        if got:
                            raised.append(got)
                else:
                    self._streak["JERKY_MOTION"] = 0

        return raised

    # -- 7b: per-rep mistakes -------------------------------------------------
    def check_rep(
        self,
        rep: Rep,
        config: Optional[ExerciseConfig],
        skip_rom: bool = False,
    ) -> list[Mistake]:
        """Evaluate a completed rep. Call once, on the frame the rep closes.

        ``skip_rom`` suppresses the range-of-motion verdict, used when the camera
        view cannot support a depth measurement.
        """
        key = config.name if config else "generic"
        found: list[Mistake] = []

        def add(
            code: str,
            severity: str,
            metric: str | None,
            value: float | None,
            limit: float | None,
            detail: str = "",
        ) -> None:
            m = Mistake(
                code=code,
                scope="rep",
                severity=severity,
                metric=metric,
                value=value,
                limit=limit,
                exercise=key,
                frame_index=rep.end_frame,
                rep_index=rep.index,
                detail=detail,
            )
            got = self._raise(m, hold=self.rep_hold_frames)
            if got:
                found.append(got)

        if config is not None:
            # --- range of motion ---
            if config.rom is not None and not skip_rom:
                rom_cfg = config.rom
                direction = config.rep_counter.direction if config.rep_counter else -1
                extreme = float(rep.extreme_value)
                if rom_cfg.target_extreme is not None:
                    short = (
                        extreme > rom_cfg.target_extreme
                        if direction < 0
                        else extreme < rom_cfg.target_extreme
                    )
                    if short:
                        add(rom_cfg.partial_code, "error", rom_cfg.metric, extreme, rom_cfg.target_extreme)
                if rom_cfg.min_range is not None and rep.rom < rom_cfg.min_range:
                    add("PARTIAL_REP", "error", rom_cfg.metric, rep.rom, rom_cfg.min_range,
                        detail="range of motion")

            # --- tempo (angular speed, not duration - see TempoCheck) ---
            if config.tempo is not None:
                t = config.tempo
                dur = rep.duration
                speed = rep.rom / dur
                prev_speeds = list(self._speeds[key])
                flagged = False
                if dur < t.min_seconds:
                    add(t.fast_code, "warning", None, dur, t.min_seconds, detail="impossibly quick")
                    flagged = True
                elif t.max_speed is not None and speed > t.max_speed:
                    add(t.fast_code, "warning", rom_metric(config), speed, t.max_speed,
                        detail="deg/s")
                    flagged = True
                if not flagged and len(prev_speeds) >= 3:
                    median = float(np.median(prev_speeds))
                    limit = t.relative_max_speed_ratio * median
                    if median > 1e-6 and speed > limit:
                        add(t.fast_code, "warning", None, speed, limit, detail="vs your median")
                if t.max_seconds is not None and dur > t.max_seconds:
                    add(t.slow_code, "info", None, dur, t.max_seconds)
                self._speeds[key].append(speed)

            # --- left/right symmetry at the extreme ---
            if config.symmetry is not None and config.symmetry.pairs:
                worst, worst_pair = 0.0, None
                for a, b in config.symmetry.pairs:
                    va = rep.metrics_at_extreme.get(a, float("nan"))
                    vb = rep.metrics_at_extreme.get(b, float("nan"))
                    if va == va and vb == vb:
                        d = abs(va - vb)
                        if d > worst:
                            worst, worst_pair = d, (a, b)
                if worst_pair and worst > config.symmetry.max_diff:
                    add(config.symmetry.code, "warning", worst_pair[0], worst,
                        config.symmetry.max_diff, detail=f"{worst_pair[0]} vs {worst_pair[1]}")

            # --- shape consistency vs the user's own average rep ---
            if config.consistency is not None and config.consistency.metric:
                c = config.consistency
                curve = _resample(rep.curve)
                past = list(self._curves[key])
                if len(past) >= c.min_reps and np.isfinite(curve).all():
                    ref = np.nanmean(np.vstack(past), axis=0)
                    corr = _pearson(curve, ref)
                    if corr == corr and corr < c.min_correlation:
                        add(c.code, "warning", c.metric, corr, c.min_correlation,
                            detail="rep shape")
                if np.isfinite(curve).all():
                    self._curves[key].append(curve)

        self._durations[key].append(rep.duration)
        return found

    # -- session summary ------------------------------------------------------
    def summary(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))


def rom_metric(config: ExerciseConfig) -> str | None:
    """Metric whose swing defines a rep's range of motion."""
    if config.rom is not None:
        return config.rom.metric
    return config.rep_counter.primary_metric if config.rep_counter else None


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4:
        return float("nan")
    a, b = a[mask] - a[mask].mean(), b[mask] - b[mask].mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / denom)


def codes_from(mistakes: Iterable[Mistake]) -> tuple[str, ...]:
    """Unique codes in a mistake list, order preserved."""
    seen: dict[str, None] = {}
    for m in mistakes:
        seen.setdefault(m.code, None)
    return tuple(seen)


def unique_codes(sequences: Sequence[Iterable[Mistake]]) -> set[str]:
    return {m.code for seq in sequences for m in seq}
