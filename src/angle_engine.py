"""Stage 1b - Universal Angle Engine.

Computes a flat dictionary of **whole-body** kinematic metrics for every frame,
independently of which exercise (if any) is being performed. This is the layer
that makes "monitor every movement" realistic: the tracking is exercise-
agnostic, only the *interpretation* layers (classifier, mistake configs) are
exercise-specific.

Two kinds of metric are produced, both keyed by plain strings so that YAML
configs can reference any of them without code changes:

* **Angles** in degrees (``knee_l``, ``elbow_r``, ``torso_lean``, ...).
* **Normalised distances** in torso-length units (``knee_valgus_l``,
  ``wrist_height_r``, ``hip_height``, ...). Dividing by torso length makes them
  independent of the person's size and distance from the camera.

For every metric that exists as a left/right pair, two virtual metrics are added
automatically:

* ``<name>_mean`` - average of the two sides (what rep counting usually wants).
* ``<name>_diff`` - absolute left/right difference (what symmetry checks want).

Coordinates are aspect-corrected before any geometry is done, otherwise angles
are skewed on non-square frames.
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Iterable, Mapping

import numpy as np

from . import landmarks as LM
from .pose_extraction import aspect_correct

# --- low level geometry ------------------------------------------------------
_EPS = 1e-9


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at ``b`` formed by ``a-b-c``, in degrees (0..180)."""
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < _EPS or n2 < _EPS:
        return float("nan")
    cos = float(np.dot(v1, v2)) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def vector_angle_from_vertical(v: np.ndarray) -> float:
    """Angle (0..180) between ``v`` and screen-up. 0 = pointing straight up."""
    n = float(np.linalg.norm(v))
    if n < _EPS:
        return float("nan")
    # screen-up is (0, -1) because image y grows downward
    cos = float(-v[1]) / n
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def signed_angle_from_vertical(v: np.ndarray) -> float:
    """Signed lean of ``v`` from screen-up. Positive = tilted toward +x."""
    n = float(np.linalg.norm(v))
    if n < _EPS:
        return float("nan")
    return float(np.degrees(np.arctan2(float(v[0]), float(-v[1]))))


def line_tilt_from_horizontal(p: np.ndarray, q: np.ndarray) -> float:
    """Absolute tilt (0..90) of segment ``p-q`` away from horizontal."""
    d = q - p
    if float(np.linalg.norm(d)) < _EPS:
        return float("nan")
    tilt = abs(float(np.degrees(np.arctan2(float(d[1]), float(d[0])))))
    return float(min(tilt, 180.0 - tilt))


# --- metric catalogue --------------------------------------------------------
#: Paired metrics; ``_l``/``_r`` suffixes are appended by the engine.
PAIRED_METRICS: tuple[str, ...] = (
    "elbow",
    "shoulder",
    "hip",
    "knee",
    "ankle",
    "wrist",
    "body_align",
    "knee_valgus",
    "elbow_offset",
    "wrist_height",
)

#: Extra single metrics appended after the paired ones.
EXTRA_METRICS: tuple[str, ...] = ("knee_ankle_ratio",)

#: Single-valued metrics.
SINGLE_METRICS: tuple[str, ...] = (
    "neck",
    "torso_lean",
    "torso_lean_signed",
    "shoulder_tilt",
    "hip_tilt",
    "hip_height",
    "shoulder_height",
    "stance_width",
    "shoulder_width",
    "view_frontality",
    "hip_drop",
)


def _all_metric_names() -> tuple[str, ...]:
    names: list[str] = []
    for base in PAIRED_METRICS:
        names += [f"{base}_l", f"{base}_r", f"{base}_mean", f"{base}_diff"]
    names += list(SINGLE_METRICS)
    names += list(EXTRA_METRICS)
    return tuple(names)


#: Every metric the engine emits, in a stable order (feature vectors depend on
#: this being deterministic across runs and across train/inference).
METRIC_NAMES: tuple[str, ...] = _all_metric_names()
METRIC_INDEX = {name: i for i, name in enumerate(METRIC_NAMES)}

#: Subset fed to the classifier. Left out: ``*_diff`` (symmetry is a form issue,
#: not an exercise identity cue) and raw signed lean (view dependent).
CLASSIFIER_METRICS: tuple[str, ...] = (
    "elbow_l", "elbow_r", "elbow_mean",
    "shoulder_l", "shoulder_r", "shoulder_mean",
    "hip_l", "hip_r", "hip_mean",
    "knee_l", "knee_r", "knee_mean",
    "ankle_mean",
    "wrist_mean",
    "body_align_mean",
    "knee_valgus_mean",
    "elbow_offset_mean",
    "wrist_height_mean",
    "neck",
    "torso_lean",
    "shoulder_tilt",
    "hip_tilt",
    "hip_height",
    "shoulder_height",
    "stance_width",
    "shoulder_width",
    "view_frontality",
    # `knee_ankle_ratio` and `hip_drop` are deliberately excluded: both are
    # undefined (NaN) for whole classes of camera view or body orientation, and
    # a feature that is missing for an entire clip teaches the classifier about
    # the camera rather than about the movement.
)

#: Angles that generic mode watches for anatomically implausible values, and
#: that the rep counter may auto-select as a periodic signal.
GENERIC_TRACK_METRICS: tuple[str, ...] = (
    "knee_mean",
    "elbow_mean",
    "hip_mean",
    "shoulder_mean",
    "ankle_mean",
    "body_align_mean",
    "torso_lean",
    "neck",
    "hip_height",
    "wrist_height_mean",
)


def _hip_drop(
    sho_mid: np.ndarray,
    hip_l: np.ndarray,
    hip_r: np.ndarray,
    ank_mid: np.ndarray,
    torso_len: float,
) -> float:
    """Gravity-referenced hip sag, in torso lengths. Positive = hips too low.

    Measures how far the hips sit *below* the straight shoulder->ankle line.
    Unlike the unsigned ``body_align`` angle this distinguishes a sagging
    push-up (positive) from a piked one (negative), because "below" is defined
    by the image's y axis rather than by which way the person is facing.

    Only defined for a roughly horizontal body (plank, push-up). For an upright
    person the shoulder->ankle line is near-vertical and the measure is
    numerically unstable, so it returns NaN and any check on it is skipped.
    """
    hip_mid = 0.5 * (hip_l + hip_r)
    d = ank_mid - sho_mid
    span = float(np.linalg.norm(d))
    if span < _EPS or abs(float(d[0])) < 0.35 * span:
        return float("nan")
    t = (float(hip_mid[0]) - float(sho_mid[0])) / float(d[0])
    y_line = float(sho_mid[1]) + t * float(d[1])
    return float((float(hip_mid[1]) - y_line) / torso_len)


#: Landmarks below this MediaPipe ``visibility`` are treated as not observed.
#: BlazePose still *reports a position* for an occluded joint - it infers one
#: from the body model - and that inferred position is often badly wrong. Using
#: it produces confident nonsense: in a 45-degree view the far leg is partly
#: hidden, so its "angle" disagrees with the near leg and every left/right
#: symmetry check fires on a perfectly symmetric squat. Treating low-visibility
#: landmarks as missing (NaN) makes the affected checks skip instead.
DEFAULT_MIN_VISIBILITY = 0.55

#: Stricter bar for comparing one side against the other. A side-by-side
#: comparison is only as good as the *worse* of the two estimates, so it needs
#: more confidence than a single-sided reading does.
SYMMETRY_MIN_VISIBILITY = 0.75


def compute_metrics(
    pts: np.ndarray,
    image_size: tuple[int, int] = (0, 0),
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
    symmetry_min_visibility: float = SYMMETRY_MIN_VISIBILITY,
) -> dict[str, float]:
    """Compute every whole-body metric for one frame.

    Parameters
    ----------
    pts:
        ``(33, 4)`` landmark array, normalised image coordinates. Column 3 is
        MediaPipe's visibility and **is** used - see ``min_visibility``.
    image_size:
        ``(width, height)`` in pixels, used for aspect correction. Pass
        ``(0, 0)`` if the coordinates are already isotropic (e.g. synthetic
        data generated in a square space).
    min_visibility:
        A metric is ``NaN`` if any landmark it depends on is less visible than
        this. Set to 0 to disable (synthetic data has visibility 1.0 anyway).
    symmetry_min_visibility:
        Higher bar applied to the ``*_diff`` metrics only.

    Returns
    -------
    dict
        ``{metric_name: value}`` for all of :data:`METRIC_NAMES`. Values that
        cannot be computed are ``nan`` rather than a fake number, so downstream
        checks can skip them instead of firing on garbage.
    """
    xy = aspect_correct(pts, image_size)
    vis = pts[:, 3] if pts.shape[1] > 3 else np.ones(pts.shape[0], dtype=np.float32)

    def seen(*idx: int) -> bool:
        return all(float(vis[i]) >= min_visibility for i in idx)

    def ang(i: int, j: int, k: int) -> float:
        """Angle at ``j``, or NaN if any of the three joints is not observed."""
        if not seen(i, j, k):
            return float("nan")
        return joint_angle(xy[i], xy[j], xy[k])

    def midpoint(i: int, j: int) -> np.ndarray:
        """Midpoint of a left/right pair, from whichever sides are observed.

        Falling back to one side rather than giving up keeps the whole engine
        alive when a subject is slightly cropped or turned, which is common and
        recoverable. Only when *neither* side is visible does everything that
        depends on this midpoint become NaN.
        """
        li, lj = seen(i), seen(j)
        if li and lj:
            return 0.5 * (xy[i] + xy[j])
        if li:
            return xy[i]
        if lj:
            return xy[j]
        return np.array([np.nan, np.nan], dtype=np.float64)

    hip_mid = midpoint(LM.LEFT_HIP, LM.RIGHT_HIP)
    sho_mid = midpoint(LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
    ank_mid = midpoint(LM.LEFT_ANKLE, LM.RIGHT_ANKLE)
    torso_vec = sho_mid - hip_mid
    torso_len = float(np.linalg.norm(torso_vec))
    if not np.isfinite(torso_len) or torso_len < _EPS:
        torso_len = float("nan")

    m: dict[str, float] = {}

    sides = (
        ("l", LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST, LM.LEFT_INDEX,
         LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE, LM.LEFT_FOOT_INDEX),
        ("r", LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST, LM.RIGHT_INDEX,
         LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE, LM.RIGHT_FOOT_INDEX),
    )

    for tag, i_sho, i_elb, i_wri, i_idx, i_hip, i_kne, i_ank, i_foot in sides:
        elb, sho, kne, ank = xy[i_elb], xy[i_sho], xy[i_kne], xy[i_ank]

        # --- joint angles ---
        m[f"elbow_{tag}"] = ang(i_sho, i_elb, i_wri)
        m[f"shoulder_{tag}"] = ang(i_elb, i_sho, i_hip)       # arm-vs-torso abduction
        m[f"hip_{tag}"] = ang(i_sho, i_hip, i_kne)
        m[f"knee_{tag}"] = ang(i_hip, i_kne, i_ank)
        m[f"ankle_{tag}"] = ang(i_kne, i_ank, i_foot)
        m[f"wrist_{tag}"] = ang(i_elb, i_wri, i_idx)
        # 180 = shoulder/hip/ankle in a straight line (plank & push-up alignment)
        m[f"body_align_{tag}"] = ang(i_sho, i_hip, i_ank)

        # --- normalised distances ---
        # Knee valgus: how far the knee has drifted toward the body midline
        # relative to the ankle. Positive = caving in. Frontal-view metric;
        # gate any check on `view_frontality`.
        if seen(i_kne, i_ank) and np.isfinite(hip_mid[0]):
            side_sign = np.sign(ank[0] - hip_mid[0])
            if side_sign == 0:
                side_sign = 1.0 if tag == "l" else -1.0
            m[f"knee_valgus_{tag}"] = float(side_sign * (ank[0] - kne[0]) / torso_len)
        else:
            m[f"knee_valgus_{tag}"] = float("nan")
        m[f"elbow_offset_{tag}"] = (
            float(abs(elb[0] - sho[0]) / torso_len) if seen(i_elb, i_sho) else float("nan")
        )
        m[f"wrist_height_{tag}"] = (
            float((sho_mid[1] - xy[i_wri][1]) / torso_len)
            if seen(i_wri) and np.isfinite(sho_mid[1]) else float("nan")
        )

    # --- single metrics ---
    both_sho = seen(LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
    both_hip = seen(LM.LEFT_HIP, LM.RIGHT_HIP)
    both_ank = seen(LM.LEFT_ANKLE, LM.RIGHT_ANKLE)
    nan = float("nan")

    m["neck"] = (
        joint_angle(xy[LM.NOSE], sho_mid, hip_mid)
        if seen(LM.NOSE) and np.isfinite(sho_mid[0]) and np.isfinite(hip_mid[0]) else nan
    )
    m["torso_lean"] = vector_angle_from_vertical(torso_vec)
    m["torso_lean_signed"] = signed_angle_from_vertical(torso_vec)
    # Tilt of a left/right line is meaningless unless both ends are observed.
    m["shoulder_tilt"] = (
        line_tilt_from_horizontal(xy[LM.LEFT_SHOULDER], xy[LM.RIGHT_SHOULDER])
        if both_sho else nan
    )
    m["hip_tilt"] = (
        line_tilt_from_horizontal(xy[LM.LEFT_HIP], xy[LM.RIGHT_HIP]) if both_hip else nan
    )
    m["hip_height"] = float((ank_mid[1] - hip_mid[1]) / torso_len)
    m["shoulder_height"] = float((ank_mid[1] - sho_mid[1]) / torso_len)
    m["stance_width"] = (
        float(abs(xy[LM.LEFT_ANKLE][0] - xy[LM.RIGHT_ANKLE][0]) / torso_len)
        if both_ank else nan
    )
    shoulder_width = (
        float(np.linalg.norm(xy[LM.LEFT_SHOULDER] - xy[LM.RIGHT_SHOULDER])) if both_sho else nan
    )
    m["shoulder_width"] = shoulder_width / torso_len
    # 1.0-ish = square to the camera, ~0.2 = side-on. Used to gate frontal-only
    # checks such as knee valgus.
    m["view_frontality"] = float(np.clip(m["shoulder_width"] / 0.85, 0.0, 1.5))
    m["hip_drop"] = (
        _hip_drop(sho_mid, xy[LM.LEFT_HIP], xy[LM.RIGHT_HIP], ank_mid, torso_len)
        if both_hip and np.isfinite(sho_mid[0]) and np.isfinite(ank_mid[0]) else nan
    )

    # Knee-vs-ankle track width. The single most reliable monocular valgus cue:
    # both knees travel forward together during a squat, so taking the *ratio*
    # of the knee span to the ankle span cancels that sagittal travel, leaving
    # only genuine frontal-plane narrowing. It is also invariant to the camera
    # azimuth (both spans shrink by the same cosine), though it becomes
    # noise-dominated in a pure side view - hence the `view_frontality` gate on
    # the check that uses it, and NaN when the feet are nearly together.
    # All four landmarks must be confidently visible: this compares two limbs in
    # the frontal plane, so an *inferred* far knee would invent valgus outright.
    # The 0.10 floor only guards against dividing by a near-zero span (feet
    # together, or a body seen edge-on). Judging whether the ratio is *usable*
    # is the job of the `view_frontality` gate on the check itself.
    if seen(LM.LEFT_ANKLE, LM.RIGHT_ANKLE, LM.LEFT_KNEE, LM.RIGHT_KNEE):
        ankle_span = float(abs(xy[LM.LEFT_ANKLE][0] - xy[LM.RIGHT_ANKLE][0]))
        knee_span = float(abs(xy[LM.LEFT_KNEE][0] - xy[LM.RIGHT_KNEE][0]))
        m["knee_ankle_ratio"] = knee_span / ankle_span if ankle_span > 0.10 * torso_len else nan
    else:
        m["knee_ankle_ratio"] = nan

    # --- virtual paired metrics ---
    # `_mean` falls back to whichever side is observed, because a one-sided
    # reading is still a usable rep signal. `_diff` does not: comparing a
    # measured joint against an inferred one is how you invent an asymmetry that
    # is not there. It requires BOTH sides above the stricter symmetry bar.
    limb_pairs = {
        "elbow": (LM.LEFT_ELBOW, LM.RIGHT_ELBOW),
        "shoulder": (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
        "hip": (LM.LEFT_HIP, LM.RIGHT_HIP),
        "knee": (LM.LEFT_KNEE, LM.RIGHT_KNEE),
        "ankle": (LM.LEFT_ANKLE, LM.RIGHT_ANKLE),
        "wrist": (LM.LEFT_WRIST, LM.RIGHT_WRIST),
        "body_align": (LM.LEFT_HIP, LM.RIGHT_HIP),
        "knee_valgus": (LM.LEFT_KNEE, LM.RIGHT_KNEE),
        "elbow_offset": (LM.LEFT_ELBOW, LM.RIGHT_ELBOW),
        "wrist_height": (LM.LEFT_WRIST, LM.RIGHT_WRIST),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # both sides NaN -> NaN
        for base in PAIRED_METRICS:
            left, right = m[f"{base}_l"], m[f"{base}_r"]
            m[f"{base}_mean"] = float(np.nanmean([left, right]))
            li, ri = limb_pairs[base]
            comparable = (float(vis[li]) >= symmetry_min_visibility
                          and float(vis[ri]) >= symmetry_min_visibility)
            m[f"{base}_diff"] = float(abs(left - right)) if comparable else nan

    return m


def metrics_to_vector(metrics: Mapping[str, float], names: Iterable[str]) -> np.ndarray:
    """Extract ``names`` from a metric dict into a float32 vector."""
    return np.array([metrics.get(n, np.nan) for n in names], dtype=np.float32)


#: Smoothing window in *seconds*. ~0.15 s removes BlazePose jitter while staying
#: short enough not to blunt the peak of a fast repetition.
DEFAULT_SMOOTHING_SECONDS = 0.15


class MetricSmoother:
    """Causal moving average over metric dictionaries, windowed by **time**.

    Deliberately time-based rather than frame-based. A fixed 5-frame average is
    0.17 s at 30 FPS but 0.45 s on a webcam delivering 11 FPS in poor light -
    long enough to flatten the bottom of a squat and make the rep counter miss
    it. Averaging over a fixed duration behaves the same at any frame rate.

    NaNs are ignored rather than poisoning the window.
    """

    def __init__(
        self,
        window_seconds: float = DEFAULT_SMOOTHING_SECONDS,
        names: Iterable[str] | None = None,
        max_samples: int = 30,
    ) -> None:
        self.window_seconds = max(0.0, float(window_seconds))
        self.names = tuple(names) if names is not None else METRIC_NAMES
        self._buf: deque[tuple[float, np.ndarray]] = deque(maxlen=max(1, int(max_samples)))

    def reset(self) -> None:
        self._buf.clear()

    @property
    def samples(self) -> int:
        return len(self._buf)

    def update(
        self,
        metrics: Mapping[str, float],
        timestamp: float | None = None,
    ) -> dict[str, float]:
        ts = float(len(self._buf)) if timestamp is None else float(timestamp)
        self._buf.append((ts, metrics_to_vector(metrics, self.names)))
        cutoff = ts - self.window_seconds
        # always retain the newest sample, so a long stall cannot empty the buffer
        while len(self._buf) > 1 and self._buf[0][0] < cutoff:
            self._buf.popleft()
        stack = np.vstack([v for _, v in self._buf])
        # Metrics that are undefined for the current view are NaN by design
        # (e.g. `hip_drop` while standing). NaN out is correct; numpy's warning
        # about it is noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(stack, axis=0)
        return {name: float(mean[i]) for i, name in enumerate(self.names)}
