"""Synthetic kinematic data generator (development / CI test bench).

**Read this before trusting any number produced from it.** This module fabricates
BlazePose-shaped landmark streams from a 2D forward-kinematic stick figure. It
exists so that the whole pipeline - angle engine, classifier, rep counter,
mistake detector, metrics report - can be built, exercised and regression-tested
without a camera and without hand-labelling video. It is **not** a substitute for
real recordings:

* it has no pose-estimation error model beyond additive Gaussian jitter,
* it is a single articulated plane, so genuine out-of-plane rotation, occlusion
  and perspective are absent,
* its "correct form" is whatever the keyframes below say it is.

Accuracy numbers measured on synthetic data therefore describe *the logic*, not
*the product*. Record real clips with ``main.py --record`` and re-run
``evaluate.py`` before quoting anything to a user. See
``docs/metrics_report.md``.

How it works
------------
Each exercise is two keyframes of joint *angles* (a rest pose and a working
pose) plus an optional fault modifier. A trapezoidal, smoothstep-eased phase
signal interpolates between them, which produces flat "ready" plateaus at both
ends - exactly the shape the rep-counting state machine expects. Landmarks are
then produced by forward kinematics, rigidly rotated (push-ups), fitted into the
frame with one similarity transform per clip (angles unaffected), and finally
perturbed with jitter and drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

import numpy as np

from . import landmarks as LM
from .session import Session, session_from_landmarks

# --- segment lengths, in units of a 1.0-tall figure ---------------------------
SHANK = 0.20
THIGH = 0.20
TORSO = 0.26
UPPER_ARM = 0.135
FOREARM = 0.135
HEAD = 0.115
HAND = 0.045
LATERAL_SHOULDER = 0.098      # half the shoulder width  (~41 cm on a 1.75 m person)
LATERAL_LEG = 0.046           # half the hip/ankle width (~19 cm)


def _dir_up(deg: float) -> np.ndarray:
    """Unit vector ``deg`` away from screen-up, positive toward +x."""
    r = math.radians(deg)
    return np.array([math.sin(r), -math.cos(r)], dtype=np.float64)


def _dir_down(deg: float) -> np.ndarray:
    """Unit vector ``deg`` away from screen-down, positive toward +x."""
    r = math.radians(deg)
    return np.array([math.sin(r), math.cos(r)], dtype=np.float64)


@dataclass(frozen=True)
class PoseParams:
    """Joint angles (degrees) describing one keyframe.

    Angle conventions chosen so the derived metrics are analytic, which makes
    the expected values in the exercise configs verifiable by hand:

    * ``knee  = 180 - |thigh - shank|``
    * ``hip   = 180 - |torso - thigh|``
    * ``elbow = 180 - |upper_arm - forearm|``
    * ``shoulder abduction = |upper_arm + torso|``
    """

    shank: tuple[float, float] = (2.0, 2.0)      # from up; knee->? shin tilt
    thigh: tuple[float, float] = (0.0, 0.0)      # from up; knee->hip direction
    torso: float = 4.0                           # from up; hip->shoulder
    upper_arm: tuple[float, float] = (2.0, 2.0)  # from down; shoulder->elbow
    forearm: tuple[float, float] = (4.0, 4.0)    # from down; elbow->wrist
    neck: float = 6.0                            # extra head tilt on top of torso
    ankle_dx: tuple[float, float] = (0.0, 0.0)
    ankle_dy: tuple[float, float] = (0.0, 0.0)
    knee_in: tuple[float, float] = (0.0, 0.0)    # valgus shift toward the midline
    stance: float = 1.0                          # scales the leg lateral offset
    global_rot: float = 0.0                      # rigid rotation about the ankles


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_pair(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (_lerp(a[0], b[0], t), _lerp(a[1], b[1], t))


def interpolate(rest: PoseParams, work: PoseParams, t: float, side_t: tuple[float, float] | None = None
                ) -> PoseParams:
    """Blend two keyframes. ``side_t`` lets left/right progress differ (asymmetry)."""
    tl, tr = (t, t) if side_t is None else side_t

    def pair(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
        return (_lerp(a[0], b[0], tl), _lerp(a[1], b[1], tr))

    return PoseParams(
        shank=pair(rest.shank, work.shank),
        thigh=pair(rest.thigh, work.thigh),
        torso=_lerp(rest.torso, work.torso, t),
        upper_arm=pair(rest.upper_arm, work.upper_arm),
        forearm=pair(rest.forearm, work.forearm),
        neck=_lerp(rest.neck, work.neck, t),
        ankle_dx=pair(rest.ankle_dx, work.ankle_dx),
        ankle_dy=pair(rest.ankle_dy, work.ankle_dy),
        knee_in=pair(rest.knee_in, work.knee_in),
        stance=_lerp(rest.stance, work.stance, t),
        global_rot=_lerp(rest.global_rot, work.global_rot, t),
    )


def build_landmarks(p: PoseParams, azimuth: float = 70.0) -> np.ndarray:
    """Forward kinematics + camera projection -> ``(33, 4)`` landmark array.

    The figure is articulated in one sagittal plane ``(u, y)`` and carries a
    separate lateral offset ``w`` per landmark. A camera at ``azimuth`` degrees
    around the subject then projects them::

        x_image = 0.5 + (u - 0.5) * sin(azimuth) + w * cos(azimuth)

    ``azimuth = 90`` is a pure side view (full joint flexion visible, left/right
    separation collapsed); ``azimuth = 0`` is dead in front (left/right visible,
    sagittal flexion invisible). Modelling this properly matters: it is the
    reason a monocular system cannot measure squat depth from the front, and it
    lets the pipeline's ``view_frontality`` gates be tested rather than assumed.
    """
    ankle_base_y = 0.90
    ankle_base_x = 0.50

    # --- legs (sagittal chain, per side) ---
    ankles, knees, hips = [], [], []
    for s in (0, 1):
        ank = np.array([ankle_base_x + p.ankle_dx[s], ankle_base_y + p.ankle_dy[s]])
        kne = ank + SHANK * _dir_up(p.shank[s])
        hip = kne + THIGH * _dir_up(p.thigh[s])
        ankles.append(ank)
        knees.append(kne)
        hips.append(hip)

    hip_mid = 0.5 * (hips[0] + hips[1])
    sho_mid = hip_mid + TORSO * _dir_up(p.torso)
    nose = sho_mid + HEAD * _dir_up(p.torso + p.neck)

    elbows, wrists = [], []
    for s in (0, 1):
        elb = sho_mid + UPPER_ARM * _dir_down(p.upper_arm[s])
        wri = elb + FOREARM * _dir_down(p.forearm[s])
        elbows.append(elb)
        wrists.append(wri)

    pts = np.zeros((LM.NUM_LANDMARKS, 2), dtype=np.float64)

    def put(idx: int, v: np.ndarray) -> None:
        pts[idx] = v

    # head cluster - small offsets around the nose, enough for the overlay and
    # for the neck angle to be meaningful.
    up = _dir_up(p.torso + p.neck)
    side = np.array([-up[1], up[0]])
    put(LM.NOSE, nose)
    put(LM.LEFT_EYE_INNER, nose + 0.012 * side + 0.012 * up)
    put(LM.LEFT_EYE, nose + 0.024 * side + 0.014 * up)
    put(LM.LEFT_EYE_OUTER, nose + 0.034 * side + 0.014 * up)
    put(LM.RIGHT_EYE_INNER, nose - 0.012 * side + 0.012 * up)
    put(LM.RIGHT_EYE, nose - 0.024 * side + 0.014 * up)
    put(LM.RIGHT_EYE_OUTER, nose - 0.034 * side + 0.014 * up)
    put(LM.LEFT_EAR, nose + 0.045 * side + 0.008 * up)
    put(LM.RIGHT_EAR, nose - 0.045 * side + 0.008 * up)
    put(LM.MOUTH_LEFT, nose + 0.014 * side - 0.020 * up)
    put(LM.MOUTH_RIGHT, nose - 0.014 * side - 0.020 * up)

    put(LM.LEFT_SHOULDER, sho_mid)
    put(LM.RIGHT_SHOULDER, sho_mid)
    put(LM.LEFT_ELBOW, elbows[0])
    put(LM.RIGHT_ELBOW, elbows[1])
    put(LM.LEFT_WRIST, wrists[0])
    put(LM.RIGHT_WRIST, wrists[1])
    for idx, s, k in (
        (LM.LEFT_INDEX, 0, 1.0), (LM.RIGHT_INDEX, 1, 1.0),
        (LM.LEFT_PINKY, 0, 0.85), (LM.RIGHT_PINKY, 1, 0.85),
        (LM.LEFT_THUMB, 0, 0.6), (LM.RIGHT_THUMB, 1, 0.6),
    ):
        put(idx, wrists[s] + k * HAND * _dir_down(p.forearm[s]))

    put(LM.LEFT_HIP, hips[0])
    put(LM.RIGHT_HIP, hips[1])
    put(LM.LEFT_KNEE, knees[0])
    put(LM.RIGHT_KNEE, knees[1])
    put(LM.LEFT_ANKLE, ankles[0])
    put(LM.RIGHT_ANKLE, ankles[1])
    foot_dir = np.array([0.055, 0.012])
    heel_dir = np.array([-0.026, 0.018])
    put(LM.LEFT_FOOT_INDEX, ankles[0] + foot_dir)
    put(LM.RIGHT_FOOT_INDEX, ankles[1] + foot_dir)
    put(LM.LEFT_HEEL, ankles[0] + heel_dir)
    put(LM.RIGHT_HEEL, ankles[1] + heel_dir)

    # --- rigid rotation about the ankles (push-ups / planks) ---
    if abs(p.global_rot) > 1e-6:
        pivot = 0.5 * (ankles[0] + ankles[1])
        th = math.radians(p.global_rot)
        c, s = math.cos(th), math.sin(th)
        rel = pts - pivot
        pts = pivot + np.stack([rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c], axis=1)

    # --- lateral (left/right) offsets, in the frontal plane ---
    w = np.zeros(LM.NUM_LANDMARKS, dtype=np.float64)
    sho_off = LATERAL_SHOULDER
    leg_off = LATERAL_LEG * p.stance
    head_off = 0.55 * sho_off
    for idx in (LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST,
                LM.LEFT_INDEX, LM.LEFT_PINKY, LM.LEFT_THUMB):
        w[idx] = sho_off
    for idx in (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST,
                LM.RIGHT_INDEX, LM.RIGHT_PINKY, LM.RIGHT_THUMB):
        w[idx] = -sho_off
    for idx in (LM.LEFT_HIP, LM.LEFT_ANKLE, LM.LEFT_HEEL, LM.LEFT_FOOT_INDEX):
        w[idx] = leg_off
    for idx in (LM.RIGHT_HIP, LM.RIGHT_ANKLE, LM.RIGHT_HEEL, LM.RIGHT_FOOT_INDEX):
        w[idx] = -leg_off
    w[LM.LEFT_KNEE] = leg_off - p.knee_in[0]
    w[LM.RIGHT_KNEE] = -(leg_off - p.knee_in[1])
    for idx in (LM.LEFT_EYE_INNER, LM.LEFT_EYE, LM.LEFT_EYE_OUTER, LM.LEFT_EAR, LM.MOUTH_LEFT):
        w[idx] = head_off
    for idx in (LM.RIGHT_EYE_INNER, LM.RIGHT_EYE, LM.RIGHT_EYE_OUTER, LM.RIGHT_EAR, LM.MOUTH_RIGHT):
        w[idx] = -head_off

    # --- camera projection ---
    az = math.radians(azimuth)
    out = np.zeros((LM.NUM_LANDMARKS, 4), dtype=np.float32)
    out[:, 0] = ankle_base_x + (pts[:, 0] - ankle_base_x) * math.sin(az) + w * math.cos(az)
    out[:, 1] = pts[:, 1]
    out[:, 3] = 1.0
    return out


# --- phase signal -------------------------------------------------------------
def trapezoid(phase: float, rest: float = 0.30, ramp: float = 0.22, hold: float = 0.13) -> float:
    """Smoothed trapezoid on ``phase in [0, 1)``: rest -> work -> hold -> rest.

    The flat sections are what let the rep-counting state machine see a clean
    READY plateau; a pure sinusoid never settles and makes hysteresis tuning
    look better than it is.
    """
    phase = phase % 1.0
    up_end = rest + ramp
    hold_end = up_end + hold
    down_end = hold_end + ramp
    if phase < rest:
        return 0.0
    if phase < up_end:
        t = (phase - rest) / ramp
    elif phase < hold_end:
        return 1.0
    elif phase < down_end:
        t = 1.0 - (phase - hold_end) / ramp
    else:
        return 0.0
    return float(t * t * (3.0 - 2.0 * t))


# --- exercise definitions -----------------------------------------------------
@dataclass(frozen=True)
class ExerciseMotion:
    name: str
    rest: PoseParams
    work: PoseParams
    period: float = 60.0            # frames per rep at 30 FPS
    in_training_set: bool = True
    #: Plausible camera azimuths for this movement, degrees. 90 = pure side
    #: view. Leg/arm flexion needs obliquity to be visible at all, so nobody
    #: films a squat from dead in front and expects depth numbers.
    azimuth_range: tuple[float, float] = (40.0, 85.0)


_STAND = PoseParams()

MOTIONS: dict[str, ExerciseMotion] = {
    # knee 177 -> 80, hip 176 -> 60, torso lean 4 -> 42
    "squat": ExerciseMotion(
        "squat",
        _STAND,
        PoseParams(shank=(22, 22), thigh=(-78, -78), torso=42.0,
                   upper_arm=(40, 40), forearm=(56, 56), neck=-4.0),
        period=62,
        azimuth_range=(40.0, 82.0),
    ),
    # elbow 176 -> 88, shoulder abduction 88 -> 52, body straight
    "pushup": ExerciseMotion(
        "pushup",
        PoseParams(shank=(1, 1), thigh=(0, 0), torso=0.0,
                   upper_arm=(88, 88), forearm=(92, 92), neck=-6.0, global_rot=78.0),
        PoseParams(shank=(1, 1), thigh=(0, 0), torso=0.0,
                   upper_arm=(52, 52), forearm=(-40, -40), neck=-6.0, global_rot=82.0),
        period=46,
        azimuth_range=(72.0, 90.0),
    ),
    # elbow 177 -> 43, upper arm pinned (shoulder abduction ~11)
    "bicep_curl": ExerciseMotion(
        "bicep_curl",
        PoseParams(upper_arm=(2, 2), forearm=(5, 5)),
        PoseParams(upper_arm=(8, 8), forearm=(145, 145), torso=5.0),
        period=42,
    ),
    # split stance: front knee 90, rear knee 100, knee_mean ~95
    "lunge": ExerciseMotion(
        "lunge",
        _STAND,
        PoseParams(shank=(12, -35), thigh=(-78, 45), torso=12.0,
                   upper_arm=(6, 6), forearm=(8, 8),
                   ankle_dx=(0.09, -0.09), ankle_dy=(0.0, -0.012)),
        period=68,
        azimuth_range=(45.0, 85.0),
    ),
    # shoulder abduction 55 -> 158, elbow 80 -> 170  (signal INCREASES)
    "shoulder_press": ExerciseMotion(
        "shoulder_press",
        PoseParams(upper_arm=(52, 52), forearm=(152, 152)),
        PoseParams(upper_arm=(155, 155), forearm=(165, 165)),
        period=46,
    ),
    # --- out-of-set movements: never trained on, used to test the unknown gate
    "toe_touch": ExerciseMotion(
        "toe_touch",
        _STAND,
        PoseParams(shank=(4, 4), thigh=(-8, -8), torso=82.0,
                   upper_arm=(-4, -4), forearm=(-6, -6), neck=10.0),
        period=64,
        in_training_set=False,
    ),
    "star_jump": ExerciseMotion(
        "star_jump",
        PoseParams(upper_arm=(4, 4), forearm=(6, 6), stance=1.0),
        PoseParams(upper_arm=(150, 150), forearm=(160, 160), stance=3.4,
                   shank=(0, 0), thigh=(0, 0)),
        period=34,
        in_training_set=False,
    ),
    "side_bend": ExerciseMotion(
        "side_bend",
        _STAND,
        PoseParams(torso=38.0, upper_arm=(20, 20), forearm=(24, 24), neck=8.0),
        period=56,
        in_training_set=False,
    ),
    "arm_swing": ExerciseMotion(
        "arm_swing",
        PoseParams(upper_arm=(2, 2), forearm=(4, 4)),
        PoseParams(upper_arm=(140, 140), forearm=(142, 142)),
        period=38,
        in_training_set=False,
    ),
    # A "standing still / between sets" class, deliberately IN the training set.
    # Without an explicit rest class the classifier must assign idle frames to
    # some exercise, and it will - a person standing motionless gets labelled
    # "squat" and the rep counter starts hunting for reps in the noise. There is
    # intentionally no `configs/idle.yaml`, so the pipeline maps this label to
    # generic mode: joints are tracked, nothing is counted or judged.
    "idle": ExerciseMotion(
        "idle",
        _STAND,
        PoseParams(torso=6.0, neck=8.0),
        period=150,
    ),
}

TRAINING_EXERCISES: tuple[str, ...] = tuple(k for k, v in MOTIONS.items() if v.in_training_set)
UNSEEN_EXERCISES: tuple[str, ...] = tuple(k for k, v in MOTIONS.items() if not v.in_training_set)


# --- faults -------------------------------------------------------------------
@dataclass(frozen=True)
class Fault:
    """A deliberate form error, with the codes it is expected to trigger."""

    name: str
    expected_codes: tuple[str, ...]
    depth: float = 1.0                              # work-pose interpolation scale
    side_depth: tuple[float, float] | None = None   # asymmetric interpolation
    period_scale: float = 1.0
    work_override: dict[str, object] | None = None
    rest_override: dict[str, object] | None = None
    #: Overrides the motion's azimuth range. Valgus is a frontal-plane fault, so
    #: it is only observable from a roughly 45-degree view.
    azimuth_range: tuple[float, float] | None = None


GOOD = Fault("good", ())

#: A rep spans roughly 57% of a period (both ramps plus the hold), so a
#: period_scale of 0.45 yields a rep well under every config's `min_seconds`
#: while still lasting more frames than `min_rep_frames`.
_FAST = 0.45

FAULTS: dict[str, tuple[Fault, ...]] = {
    "squat": (
        GOOD,
        Fault("shallow", ("INSUFFICIENT_DEPTH",), depth=0.68),
        Fault("valgus", ("KNEE_ALIGNMENT_POOR",), work_override={"knee_in": (0.022, 0.022)},
              azimuth_range=(42.0, 60.0)),
        Fault("lean", ("EXCESSIVE_TORSO_LEAN",), work_override={"torso": 72.0}),
        Fault("asymmetric", ("ASYMMETRIC_MOVEMENT",), side_depth=(1.0, 0.72)),
        Fault("fast", ("TOO_FAST",), period_scale=_FAST),
    ),
    "pushup": (
        GOOD,
        Fault("partial", ("PARTIAL_REP",), depth=0.62),
        Fault("sag", ("HIP_SAG",), work_override={"torso": -26.0}, rest_override={"torso": -26.0}),
        Fault("flare", ("ELBOW_FLARE",), work_override={"upper_arm": (88.0, 88.0),
                                                        "forearm": (-4.0, -4.0)}),
        Fault("asymmetric", ("ASYMMETRIC_MOVEMENT",), side_depth=(1.0, 0.70)),
        Fault("fast", ("TOO_FAST",), period_scale=_FAST),
    ),
    "bicep_curl": (
        GOOD,
        Fault("partial", ("PARTIAL_REP",), depth=0.60),
        Fault("swing", ("SHOULDER_SWING",), work_override={"upper_arm": (52.0, 52.0)}),
        # Lean applied to the WORK pose only. Shifting the rest pose too would
        # move the rep signal's baseline outside the config's near_threshold,
        # which stops rep counting entirely - a real failure mode, but not the
        # one this clip is meant to test.
        Fault("body_swing", ("TORSO_SWING",), work_override={"torso": 40.0},
              azimuth_range=(50.0, 85.0)),
        Fault("asymmetric", ("ASYMMETRIC_MOVEMENT",), side_depth=(1.0, 0.62)),
        Fault("fast", ("TOO_FAST",), period_scale=_FAST),
    ),
    "lunge": (
        GOOD,
        Fault("shallow", ("INSUFFICIENT_DEPTH",), depth=0.70),
        Fault("lean", ("EXCESSIVE_TORSO_LEAN",), work_override={"torso": 44.0}),
        Fault("fast", ("TOO_FAST",), period_scale=_FAST),
    ),
    "shoulder_press": (
        GOOD,
        Fault("partial", ("PARTIAL_REP",), depth=0.70),
        Fault("no_lockout", ("INCOMPLETE_LOCKOUT",), work_override={"forearm": (100.0, 100.0)}),
        Fault("lean", ("TORSO_SWING",), work_override={"torso": 40.0},
              azimuth_range=(50.0, 85.0)),
        Fault("asymmetric", ("ASYMMETRIC_MOVEMENT",), side_depth=(1.0, 0.62)),
        Fault("fast", ("TOO_FAST",), period_scale=_FAST),
    ),
}


def _apply_override(p: PoseParams, override: dict[str, object] | None) -> PoseParams:
    return p if not override else replace(p, **override)  # type: ignore[arg-type]


# --- clip generation ----------------------------------------------------------
def _fit_transform(frames: np.ndarray, margin: float = 0.07, rng: np.random.Generator | None = None
                   ) -> tuple[float, np.ndarray]:
    """One similarity transform per clip so the figure fills the frame.

    Uniform scale + translation, so every joint angle and every torso-normalised
    distance is unchanged.
    """
    xy = frames[:, :, :2].reshape(-1, 2)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    extent = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-6))
    target = 1.0 - 2.0 * margin
    scale = target / extent
    if rng is not None:
        scale *= float(rng.uniform(0.82, 1.0))
    centre = 0.5 * (lo + hi)
    offset = np.array([0.5, 0.5]) - scale * centre
    if rng is not None:
        offset += rng.uniform(-0.035, 0.035, size=2)
    return scale, offset


def generate_clip(
    exercise: str,
    reps: int = 8,
    fault: Fault = GOOD,
    fps: float = 30.0,
    noise: float = 0.0030,
    seed: int | None = None,
    mirror: bool | None = None,
    subject: str = "synthetic",
    azimuth: float | None = None,
    depth_scale: float = 1.0,
    period_jitter: tuple[float, float] = (0.88, 1.14),
    pose_jitter: float = 0.0,
) -> Session:
    """Generate one labelled synthetic clip as a :class:`~src.session.Session`.

    ``noise`` is the per-landmark Gaussian jitter in normalised image units;
    0.003 is roughly the frame-to-frame wobble BlazePose shows on a static
    subject at 480p.

    ``depth_scale``, ``period_jitter`` and ``pose_jitter`` widen the range of
    motion, tempo and posture. Training clips should use them: a classifier
    trained only on textbook-depth, metronome-tempo, perfectly-upright reps
    treats a shallow, hurried or leaning rep as a *different movement*, and the
    novelty gate then rejects it as unknown exactly when the user most needs
    feedback. Leave them at their defaults for evaluation clips so the fault
    labels stay meaningful.
    """
    if exercise not in MOTIONS:
        raise KeyError(f"unknown synthetic motion '{exercise}'. Known: {sorted(MOTIONS)}")
    motion = MOTIONS[exercise]
    rng = np.random.default_rng(seed)

    rest = _apply_override(motion.rest, fault.rest_override)
    work = _apply_override(motion.work, fault.work_override)

    if pose_jitter > 0.0:
        d_torso = float(rng.uniform(-pose_jitter, pose_jitter))
        d_neck = float(rng.uniform(-0.5 * pose_jitter, 0.5 * pose_jitter))
        rest = replace(rest, torso=rest.torso + d_torso, neck=rest.neck + d_neck)
        work = replace(work, torso=work.torso + d_torso, neck=work.neck + d_neck)

    az_range = fault.azimuth_range or motion.azimuth_range
    if azimuth is None:
        azimuth = float(rng.uniform(*az_range))

    period = motion.period * fault.period_scale * float(rng.uniform(*period_jitter))
    period = max(14.0, period)
    lead_in = int(round(0.6 * fps))               # a moment of stillness first
    n_frames = int(round(period * reps)) + lead_in

    poses: list[np.ndarray] = []
    for i in range(n_frames):
        if i < lead_in:
            t = 0.0
        else:
            t = trapezoid((i - lead_in) / period)
        amp = fault.depth * depth_scale * float(rng.uniform(0.97, 1.03))
        if fault.side_depth is None:
            side_t = (t * amp, t * amp)
        else:
            side_t = (t * amp * fault.side_depth[0], t * amp * fault.side_depth[1])
        p = interpolate(rest, work, t * amp, side_t)
        poses.append(build_landmarks(p, azimuth))

    frames = np.stack(poses)
    scale, offset = _fit_transform(frames, rng=rng)
    frames[:, :, :2] = frames[:, :, :2] * scale + offset

    if mirror is None:
        mirror = bool(rng.integers(0, 2))
    if mirror:
        frames[:, :, 0] = 1.0 - frames[:, :, 0]

    # jitter + a slow drift, so window statistics are not artificially clean
    frames[:, :, :2] += rng.normal(0.0, noise, size=frames[:, :, :2].shape).astype(np.float32)
    drift = np.cumsum(rng.normal(0.0, noise * 0.28, size=(n_frames, 1, 2)), axis=0)
    frames[:, :, :2] += drift.astype(np.float32)
    frames[:, :, 3] = np.clip(rng.normal(0.95, 0.03, size=frames.shape[:2]), 0.0, 1.0)

    timestamps = np.arange(n_frames, dtype=np.float64) / fps
    return session_from_landmarks(
        frames,
        timestamps=timestamps,
        image_size=(0, 0),          # already isotropic; no aspect correction needed
        label=exercise,
        fault=fault.name,
        expected_reps=reps,
        expected_codes=list(fault.expected_codes),
        depth_scale=round(float(depth_scale), 3),
        subject=subject,
        fps=fps,
        source="synthetic",
        seed=seed,
        mirror=bool(mirror),
        azimuth=round(float(azimuth), 1),
        synthetic=True,
    )


def generate_dataset(
    exercises: Sequence[str] = TRAINING_EXERCISES,
    clips_per_exercise: int = 6,
    reps: int = 8,
    seed: int = 0,
    faults: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Iterable[Session]:
    """Yield a set of clips. ``faults=True`` produces the bad-form test set."""
    counter = 0
    for ex in exercises:
        options = FAULTS.get(ex, (GOOD,)) if faults else (GOOD,)
        for fault in options:
            n = clips_per_exercise if fault is GOOD else max(1, clips_per_exercise // 2)
            for k in range(n):
                counter += 1
                s = generate_clip(
                    ex,
                    reps=reps,
                    fault=fault,
                    seed=seed * 1000 + counter,
                    subject=f"synthetic_{k % 3}",
                )
                if progress:
                    progress(f"{ex:<15} {fault.name:<12} reps={reps} frames={s.n_frames}")
                yield s
