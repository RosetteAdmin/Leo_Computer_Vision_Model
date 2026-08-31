"""Stage 2 - Exercise configuration system.

Everything exercise-specific lives in ``configs/<name>.yaml``. Adding a new
exercise means writing one YAML file: which joint drives rep counting, what a
correct rep looks like, and which angles to police. No pipeline code changes.

The loader validates aggressively and fails loudly, because a silently ignored
typo in a metric name would look like "the mistake detector doesn't work".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .angle_engine import METRIC_NAMES

VALID_PHASES: tuple[str, ...] = ("any", "ready", "moving_away", "extreme", "returning", "moving")


class ConfigError(ValueError):
    """Raised when an exercise config is malformed."""


def _require(d: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"{where}: missing required key '{key}'")
    return d[key]


def _check_metric(name: str, where: str) -> str:
    if name not in METRIC_NAMES:
        raise ConfigError(
            f"{where}: unknown metric '{name}'. "
            f"Valid metrics are listed in angle_engine.METRIC_NAMES."
        )
    return name


#: Which statistic of a metric's recent history a signature gate is judged on.
VALID_GATE_STATS: tuple[str, ...] = ("peak", "trough", "median")


@dataclass(frozen=True)
class Gate:
    """Precondition for a check. Skips the check when the view is unsuitable.

    Example: knee-valgus is only measurable from the front, so its check is
    gated on ``view_frontality >= 0.5``.

    ``over`` applies only in a ``signature`` block, where the gate is judged
    against a window of recent motion rather than one frame. It has to be stated
    explicitly because the useful end differs per condition: "the feet never got
    wide" is about the *peak* of stance width, while "the elbow bent at some
    point" is about the *trough* of the elbow angle. Guessing one convention for
    both silently breaks half the signatures.
    """

    metric: str
    min: float | None = None
    max: float | None = None
    over: str = "peak"

    def passes(self, metrics: Mapping[str, float]) -> bool:
        v = metrics.get(self.metric)
        if v is None or v != v:  # missing or NaN
            return False
        if self.min is not None and v < self.min:
            return False
        if self.max is not None and v > self.max:
            return False
        return True


@dataclass(frozen=True)
class AngleCheck:
    """Per-frame angle-range rule (Section 7a mistakes)."""

    code: str
    metric: str
    phases: tuple[str, ...] = ("any",)
    min: float | None = None
    max: float | None = None
    min_frames: int = 3
    severity: str = "warning"
    gates: tuple[Gate, ...] = ()
    highlight: tuple[str, ...] = ()

    def applies_to_phase(self, phase: str) -> bool:
        if "any" in self.phases:
            return True
        if "moving" in self.phases and phase in ("moving_away", "returning"):
            return True
        return phase in self.phases

    def violated(self, value: float) -> bool:
        if value != value:  # NaN
            return False
        if self.min is not None and value < self.min:
            return True
        if self.max is not None and value > self.max:
            return True
        return False


@dataclass(frozen=True)
class RepCounterConfig:
    """Drives the generic state machine (Section 9)."""

    primary_metric: str
    near_threshold: float
    far_threshold: float
    hysteresis: float = 6.0
    #: Minimum / maximum rep duration in **seconds**, not frames. Frame counts
    #: are not portable: a webcam in dim light drops to 11 FPS, at which point a
    #: "minimum 8 frames" rule silently rejects genuine 0.7 s repetitions.
    min_rep_seconds: float = 0.27
    max_rep_seconds: float = 20.0
    #: Fraction of the near->far span a movement must cover to be counted at
    #: all. Reps between this and ``far_threshold`` are counted *and* flagged
    #: as partial, instead of being silently dropped.
    count_ratio: float = 0.45

    @property
    def direction(self) -> int:
        """+1 when the worked position has a *larger* metric value, else -1.

        Squat/curl/push-up all *decrease* their primary angle when working;
        a shoulder press *increases* it. One sign handles both.
        """
        return 1 if self.far_threshold > self.near_threshold else -1


@dataclass(frozen=True)
class RomCheck:
    """Range-of-motion requirement evaluated on each completed rep."""

    metric: str
    target_extreme: float | None = None
    min_range: float | None = None
    partial_code: str = "PARTIAL_REP"


@dataclass(frozen=True)
class TempoCheck:
    """Tempo limits for a completed rep.

    The primary criterion is **angular speed** (``rom / duration``), not
    duration. Duration alone is confounded with range of motion: a deliberately
    shallow rep covers less angle, so it crosses the counter's two hysteresis
    thresholds sooner and *measures* as fast even when the person is moving
    slowly. Normalising by the ROM the rep actually covered separates "rushed"
    from "shallow", which matters because the coaching cues are opposites.

    ``min_seconds`` remains as a hard floor for physically impossible reps.
    """

    max_speed: float | None = None        # degrees per second
    min_seconds: float = 0.25
    max_seconds: float | None = None
    relative_max_speed_ratio: float = 1.9  # vs the user's own median speed
    fast_code: str = "TOO_FAST"
    slow_code: str = "TOO_SLOW"


@dataclass(frozen=True)
class SymmetryCheck:
    pairs: tuple[tuple[str, str], ...] = ()
    max_diff: float = 15.0
    code: str = "ASYMMETRIC_MOVEMENT"


@dataclass(frozen=True)
class ConsistencyCheck:
    metric: str | None = None
    min_correlation: float = 0.85
    min_reps: int = 3
    code: str = "INCONSISTENT_FORM"


@dataclass(frozen=True)
class ExerciseConfig:
    name: str
    display_name: str
    description: str = ""
    rep_counter: RepCounterConfig | None = None
    angle_checks: tuple[AngleCheck, ...] = ()
    rom: RomCheck | None = None
    tempo: TempoCheck | None = None
    symmetry: SymmetryCheck | None = None
    consistency: ConsistencyCheck | None = None
    tracked_metrics: tuple[str, ...] = ()
    #: Conditions the camera view must satisfy for range-of-motion verdicts to
    #: be trustworthy. A single 2D camera cannot measure sagittal joint angles
    #: from dead in front, so instead of quietly reporting nonsense the pipeline
    #: suppresses ROM checks and raises SUBOPTIMAL_CAMERA_VIEW.
    view_gates: tuple[Gate, ...] = ()
    #: Postural conditions that identify this movement, used by the training-free
    #: :class:`~src.matcher.ConfigMatcher`. Needed because several exercises share
    #: a primary joint: a bicep curl and a push-up both swing the elbow through
    #: ~90 degrees, and only the torso orientation tells them apart.
    signature: tuple[Gate, ...] = ()
    source_path: Path | None = field(default=None, compare=False)

    def view_ok(self, metrics: Mapping[str, float]) -> bool:
        return all(g.passes(metrics) for g in self.view_gates)

    @property
    def primary_metric(self) -> str | None:
        return self.rep_counter.primary_metric if self.rep_counter else None


# --- parsing -----------------------------------------------------------------
def _parse_gates(raw: Any, where: str) -> tuple[Gate, ...]:
    if not raw:
        return ()
    out = []
    for i, g in enumerate(raw):
        w = f"{where}.gates[{i}]"
        over = str(g.get("over", "peak"))
        if over not in VALID_GATE_STATS:
            raise ConfigError(f"{w}: invalid 'over' value '{over}'. Valid: {VALID_GATE_STATS}")
        out.append(
            Gate(
                metric=_check_metric(str(_require(g, "metric", w)), w),
                min=_as_float(g.get("min")),
                max=_as_float(g.get("max")),
                over=over,
            )
        )
    return tuple(out)


def _as_float(v: Any) -> float | None:
    return None if v is None else float(v)


def _parse_angle_check(raw: Mapping[str, Any], where: str) -> AngleCheck:
    code = str(_require(raw, "code", where))
    metric = _check_metric(str(_require(raw, "metric", where)), where)
    phases_raw = raw.get("phases", raw.get("phase", "any"))
    phases = tuple(phases_raw) if isinstance(phases_raw, (list, tuple)) else (str(phases_raw),)
    for p in phases:
        if p not in VALID_PHASES:
            raise ConfigError(f"{where}: invalid phase '{p}'. Valid: {VALID_PHASES}")
    if raw.get("min") is None and raw.get("max") is None:
        raise ConfigError(f"{where}: angle check '{code}' needs at least one of min/max")
    return AngleCheck(
        code=code,
        metric=metric,
        phases=phases,
        min=_as_float(raw.get("min")),
        max=_as_float(raw.get("max")),
        min_frames=int(raw.get("min_frames", 3)),
        severity=str(raw.get("severity", "warning")),
        gates=_parse_gates(raw.get("gates"), f"{where}[{code}]"),
        highlight=tuple(raw.get("highlight", ())),
    )


def parse_exercise_config(data: Mapping[str, Any], source: Path | None = None) -> ExerciseConfig:
    """Validate a parsed YAML mapping into an :class:`ExerciseConfig`."""
    where = str(source) if source else "<config>"
    name = str(_require(data, "name", where))

    rc_raw = data.get("rep_counter")
    rep_counter = None
    if rc_raw:
        w = f"{where}.rep_counter"
        rep_counter = RepCounterConfig(
            primary_metric=_check_metric(str(_require(rc_raw, "primary_metric", w)), w),
            near_threshold=float(_require(rc_raw, "near_threshold", w)),
            far_threshold=float(_require(rc_raw, "far_threshold", w)),
            hysteresis=float(rc_raw.get("hysteresis", 6.0)),
            # `min_rep_frames` / `max_rep_frames` are accepted for backward
            # compatibility and converted assuming the 30 FPS they were written
            # for. New configs should use the *_seconds keys.
            min_rep_seconds=float(
                rc_raw.get("min_rep_seconds", float(rc_raw.get("min_rep_frames", 8)) / 30.0)
            ),
            max_rep_seconds=float(
                rc_raw.get("max_rep_seconds", float(rc_raw.get("max_rep_frames", 600)) / 30.0)
            ),
            count_ratio=float(rc_raw.get("count_ratio", 0.45)),
        )
        if rep_counter.near_threshold == rep_counter.far_threshold:
            raise ConfigError(f"{w}: near_threshold and far_threshold must differ")

    angle_checks = tuple(
        _parse_angle_check(c, f"{where}.angle_checks[{i}]")
        for i, c in enumerate(data.get("angle_checks", ()) or ())
    )

    rep_raw = data.get("rep_checks", {}) or {}

    rom = None
    if rep_raw.get("rom"):
        w = f"{where}.rep_checks.rom"
        r = rep_raw["rom"]
        rom = RomCheck(
            metric=_check_metric(str(r.get("metric") or (rep_counter.primary_metric if rep_counter else "")), w),
            target_extreme=_as_float(r.get("target_extreme")),
            min_range=_as_float(r.get("min_range")),
            partial_code=str(r.get("code", "PARTIAL_REP")),
        )

    tempo = None
    if rep_raw.get("tempo"):
        t = rep_raw["tempo"]
        tempo = TempoCheck(
            max_speed=_as_float(t.get("max_speed")),
            min_seconds=float(t.get("min_seconds", 0.25)),
            max_seconds=_as_float(t.get("max_seconds")),
            relative_max_speed_ratio=float(t.get("relative_max_speed_ratio", 1.9)),
            fast_code=str(t.get("fast_code", "TOO_FAST")),
            slow_code=str(t.get("slow_code", "TOO_SLOW")),
        )

    symmetry = None
    if rep_raw.get("symmetry"):
        w = f"{where}.rep_checks.symmetry"
        s = rep_raw["symmetry"]
        pairs = []
        for pair in s.get("pairs", ()) or ():
            if len(pair) != 2:
                raise ConfigError(f"{w}: each symmetry pair needs exactly 2 metrics")
            pairs.append((_check_metric(str(pair[0]), w), _check_metric(str(pair[1]), w)))
        symmetry = SymmetryCheck(
            pairs=tuple(pairs),
            max_diff=float(s.get("max_diff", 15.0)),
            code=str(s.get("code", "ASYMMETRIC_MOVEMENT")),
        )

    consistency = None
    if rep_raw.get("consistency"):
        w = f"{where}.rep_checks.consistency"
        c = rep_raw["consistency"]
        metric = c.get("metric") or (rep_counter.primary_metric if rep_counter else None)
        consistency = ConsistencyCheck(
            metric=_check_metric(str(metric), w) if metric else None,
            min_correlation=float(c.get("min_correlation", 0.85)),
            min_reps=int(c.get("min_reps", 3)),
            code=str(c.get("code", "INCONSISTENT_FORM")),
        )

    tracked = tuple(
        _check_metric(str(t), f"{where}.tracked_metrics") for t in (data.get("tracked_metrics", ()) or ())
    )

    return ExerciseConfig(
        name=name,
        display_name=str(data.get("display_name", name.replace("_", " ").title())),
        description=str(data.get("description", "")),
        rep_counter=rep_counter,
        angle_checks=angle_checks,
        rom=rom,
        tempo=tempo,
        symmetry=symmetry,
        consistency=consistency,
        tracked_metrics=tracked,
        view_gates=_parse_gates(data.get("view"), f"{where}.view"),
        signature=_parse_gates(data.get("signature"), f"{where}.signature"),
        source_path=source,
    )


def load_exercise_config(path: str | Path) -> ExerciseConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: top level of an exercise config must be a mapping")
    cfg = parse_exercise_config(data, path)
    if cfg.name != path.stem:
        raise ConfigError(f"{path}: 'name' ({cfg.name}) must match the filename stem ({path.stem})")
    return cfg


class ExerciseLibrary(Mapping[str, ExerciseConfig]):
    """All exercise configs found in a directory, keyed by name."""

    def __init__(self, configs: Iterable[ExerciseConfig]) -> None:
        self._by_name = {c.name: c for c in configs}

    @classmethod
    def from_dir(cls, directory: str | Path) -> "ExerciseLibrary":
        directory = Path(directory)
        if not directory.is_dir():
            raise ConfigError(f"config directory not found: {directory}")
        files = sorted(list(directory.glob("*.yaml")) + list(directory.glob("*.yml")))
        if not files:
            raise ConfigError(f"no *.yaml exercise configs in {directory}")
        return cls([load_exercise_config(f) for f in files])

    def __getitem__(self, key: str) -> ExerciseConfig:
        return self._by_name[key]

    def __iter__(self):
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))
