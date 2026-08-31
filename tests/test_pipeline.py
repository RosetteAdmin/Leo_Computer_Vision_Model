"""Regression tests for the analysis layers.

    python -m pytest tests -q

Deliberately covers the parts where a silent bug would be invisible in the live
demo: angle sign conventions, aspect correction, rep counting in both signal
directions, threshold-rule gating, and the per-rep verdicts. The classifier is
exercised end-to-end by ``src/evaluate.py`` instead - training a model inside a
unit test would make the suite slow and flaky.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src import landmarks as LM
from src.angle_engine import (METRIC_NAMES, MetricSmoother, compute_metrics,
                              joint_angle, line_tilt_from_horizontal,
                              signed_angle_from_vertical, vector_angle_from_vertical)
from src.exercise_config import (ConfigError, ExerciseLibrary, RepCounterConfig,
                                 parse_exercise_config)
from src.features import (N_FRAME_FEATURES, N_WINDOW_FEATURES, aggregate_window,
                          frame_features, windows_from_sequence)
from src.mistake_detector import MistakeDetector
from src.pipeline import MonitorPipeline
from src.rep_counter import EXTREME, READY, GenericRepCounter, RepCounter
from src.synthetic import (FAULTS, GOOD, MOTIONS, PoseParams, build_landmarks,
                           generate_clip, interpolate, trapezoid)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# --- geometry -----------------------------------------------------------------
def test_joint_angle_right_angle_and_straight():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    assert joint_angle(a, b, c) == pytest.approx(90.0)
    assert joint_angle(np.array([0.0, 1.0]), b, np.array([0.0, -1.0])) == pytest.approx(180.0)


def test_joint_angle_degenerate_is_nan():
    p = np.array([0.5, 0.5])
    assert math.isnan(joint_angle(p, p, np.array([0.1, 0.1])))


def test_vertical_angle_conventions():
    # image y grows downward, so "up" is (0, -1)
    assert vector_angle_from_vertical(np.array([0.0, -1.0])) == pytest.approx(0.0)
    assert vector_angle_from_vertical(np.array([1.0, 0.0])) == pytest.approx(90.0)
    assert signed_angle_from_vertical(np.array([1.0, -1.0])) == pytest.approx(45.0)
    assert signed_angle_from_vertical(np.array([-1.0, -1.0])) == pytest.approx(-45.0)


def test_line_tilt_is_folded_to_0_90():
    p, q = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert line_tilt_from_horizontal(p, q) == pytest.approx(0.0)
    assert line_tilt_from_horizontal(p, np.array([1.0, 1.0])) == pytest.approx(45.0)
    assert line_tilt_from_horizontal(p, np.array([-1.0, 1.0])) == pytest.approx(45.0)


def test_aspect_correction_changes_measured_angle():
    """A 16:9 frame skews angles unless x is rescaled - this is easy to forget."""
    pts = np.zeros((LM.NUM_LANDMARKS, 4), dtype=np.float32)
    pts[:, 3] = 1.0
    # a limb at 45 degrees in *pixel* space on a 1280x720 frame
    pts[LM.LEFT_SHOULDER, :2] = (0.5, 0.5)
    pts[LM.LEFT_ELBOW, :2] = (0.5, 0.6)
    # forearm at 45 degrees in pixel space: dx == dy in PIXELS, so dx in
    # normalised units must be scaled down by the aspect ratio
    pts[LM.LEFT_WRIST, :2] = (0.5 + 0.1 * 720 / 1280, 0.7)
    square = compute_metrics(pts, (0, 0))["elbow_l"]
    corrected = compute_metrics(pts, (1280, 720))["elbow_l"]
    assert corrected == pytest.approx(135.0, abs=0.5)
    assert abs(square - 135.0) > 10.0


# --- metric engine ------------------------------------------------------------
def test_engine_emits_every_declared_metric():
    pts = build_landmarks(PoseParams(), azimuth=60.0)
    m = compute_metrics(pts, (0, 0))
    assert set(m) == set(METRIC_NAMES)


def test_paired_mean_and_diff_are_consistent():
    p = PoseParams(shank=(20.0, 0.0), thigh=(-70.0, 0.0))
    m = compute_metrics(build_landmarks(p, azimuth=85.0), (0, 0))
    assert m["knee_mean"] == pytest.approx((m["knee_l"] + m["knee_r"]) / 2, abs=1e-3)
    assert m["knee_diff"] == pytest.approx(abs(m["knee_l"] - m["knee_r"]), abs=1e-3)
    assert m["knee_diff"] > 60.0


def test_knee_ankle_ratio_detects_valgus_and_is_azimuth_stable():
    """The valgus cue must survive a change of camera angle."""
    good = PoseParams(shank=(20.0, 20.0), thigh=(-70.0, -70.0))
    bad = PoseParams(shank=(20.0, 20.0), thigh=(-70.0, -70.0), knee_in=(0.022, 0.022))
    for az in (30.0, 45.0, 60.0):
        g = compute_metrics(build_landmarks(good, az), (0, 0))["knee_ankle_ratio"]
        b = compute_metrics(build_landmarks(bad, az), (0, 0))["knee_ankle_ratio"]
        assert g == pytest.approx(1.0, abs=0.05)
        assert b < 0.8


def test_hip_drop_sign_and_nan_when_upright():
    upright = compute_metrics(build_landmarks(PoseParams(), 70.0), (0, 0))
    assert math.isnan(upright["hip_drop"]), "hip_drop is unmeasurable on a standing body"
    plank = MOTIONS["pushup"].rest
    flat = compute_metrics(build_landmarks(plank, 85.0), (0, 0))["hip_drop"]
    sag = compute_metrics(
        build_landmarks(PoseParams(**{**plank.__dict__, "torso": -26.0}), 85.0), (0, 0)
    )["hip_drop"]
    assert abs(flat) < 0.05
    assert sag > 0.15, "a sagging plank must read positive"


def test_occluded_landmarks_produce_nan_not_a_guess():
    """BlazePose invents positions for hidden joints; we must not believe them.

    Regression test for real-world false alarms: at a 45-degree camera angle the
    far leg is partly occluded, and using its inferred position made every
    left/right symmetry check fire on a symmetric squat.
    """
    pts = build_landmarks(PoseParams(shank=(15.0, 15.0), thigh=(-60.0, -60.0)), 55.0)
    pts[LM.RIGHT_KNEE, 3] = 0.2         # far knee hidden
    m = compute_metrics(pts, (0, 0))
    assert math.isnan(m["knee_r"]), "an unobserved knee must not report an angle"
    assert not math.isnan(m["knee_l"]), "the visible side must still work"
    assert not math.isnan(m["knee_mean"]), "mean should fall back to the visible side"
    assert math.isnan(m["knee_diff"]), "symmetry needs both sides actually observed"
    assert math.isnan(m["knee_ankle_ratio"]), "valgus compares limbs; needs both knees"


def test_symmetry_needs_higher_confidence_than_a_single_reading():
    pts = build_landmarks(PoseParams(shank=(15.0, 15.0), thigh=(-60.0, -60.0)), 55.0)
    pts[LM.RIGHT_KNEE, 3] = 0.65        # above DEFAULT_MIN_VISIBILITY, below symmetry bar
    m = compute_metrics(pts, (0, 0))
    assert not math.isnan(m["knee_r"])
    assert math.isnan(m["knee_diff"])


def test_partial_occlusion_does_not_kill_the_whole_engine():
    """One hidden hip must not blank every metric - the rep signal must survive."""
    pts = build_landmarks(PoseParams(shank=(15.0, 15.0), thigh=(-60.0, -60.0)), 55.0)
    pts[LM.RIGHT_HIP, 3] = 0.1
    m = compute_metrics(pts, (0, 0))
    assert not math.isnan(m["knee_l"])
    assert not math.isnan(m["torso_lean"])
    assert not math.isnan(m["knee_mean"])


def test_generic_asymmetry_does_not_fire_on_occluded_side():
    """The exact live failure: 20 LARGE_ASYMMETRY warnings on good form."""
    det = MistakeDetector()
    pts = build_landmarks(PoseParams(shank=(15.0, 0.0), thigh=(-60.0, 0.0)), 55.0)
    pts[LM.RIGHT_KNEE, 3] = 0.3         # asymmetry is real in the data but unobserved
    m = compute_metrics(pts, (0, 0))
    for i in range(20):
        det.check_frame(m, READY, None, i, i / 30.0)
    assert "LARGE_ASYMMETRY" not in det.counts
    assert "EXTREME_JOINT_ANGLE" not in det.counts


def test_metrics_are_scale_and_translation_invariant():
    pts = build_landmarks(PoseParams(shank=(15.0, 15.0), thigh=(-60.0, -60.0)), 70.0)
    moved = pts.copy()
    moved[:, :2] = moved[:, :2] * 0.6 + 0.2
    a, b = compute_metrics(pts, (0, 0)), compute_metrics(moved, (0, 0))
    for name in ("knee_mean", "hip_mean", "torso_lean", "hip_height", "stance_width"):
        assert a[name] == pytest.approx(b[name], abs=0.3), name


def test_smoother_ignores_nan_and_averages():
    sm = MetricSmoother(window_seconds=0.15, names=("knee_l",))
    assert sm.update({"knee_l": 100.0}, 0.00)["knee_l"] == pytest.approx(100.0)
    assert sm.update({"knee_l": float("nan")}, 0.03)["knee_l"] == pytest.approx(100.0)
    assert sm.update({"knee_l": 120.0}, 0.06)["knee_l"] == pytest.approx(110.0)


def test_smoother_window_is_time_based_not_frame_based():
    """Prevents the bug that made a real session under-count.

    A fixed 5-frame average is 0.17 s at 30 FPS but 0.45 s on a webcam limited to
    11 FPS in dim light - long enough to flatten the bottom of a squat so the rep
    counter never sees the depth. A time window behaves the same at any rate.
    """
    fast = MetricSmoother(window_seconds=0.15, names=("knee_l",))
    slow = MetricSmoother(window_seconds=0.15, names=("knee_l",))
    for i in range(10):
        fast.update({"knee_l": 100.0}, i / 30.0)   # 0.15 s == 5 samples
        slow.update({"knee_l": 100.0}, i / 10.0)   # 0.15 s == 2 samples
    assert fast.samples > slow.samples
    # a sudden dip must be preserved at least as well at the lower frame rate
    dip_fast = fast.update({"knee_l": 0.0}, 10 / 30.0)["knee_l"]
    dip_slow = slow.update({"knee_l": 0.0}, 10 / 10.0)["knee_l"]
    assert dip_slow < dip_fast


def test_rep_counting_is_frame_rate_independent():
    """The *same real movement* must count the same however fast we sample it.

    Regression test for a live under-count: a webcam limited to ~11 FPS in dim
    light missed reps, because `min_rep_frames` and the smoothing window were
    frame counts calibrated at 30 FPS. Here one rep always takes 2.0 seconds of
    wall time and only the sampling rate changes.
    """
    cfg = RepCounterConfig(primary_metric="knee_mean", near_threshold=158,
                           far_threshold=110, hysteresis=7, min_rep_seconds=0.27)
    seconds_per_rep, n_reps = 2.0, 5
    for fps in (8.0, 11.0, 15.0, 30.0, 60.0):
        per_rep = int(round(seconds_per_rep * fps))
        values = [178.0 + (82.0 - 178.0) * trapezoid((i % per_rep) / per_rep)
                  for i in range(n_reps * per_rep)]
        counter = RepCounter(cfg, "squat")
        smoother = MetricSmoother(window_seconds=0.15, names=("knee_mean",))
        counted = 0
        for i, v in enumerate(values):
            t = i / fps
            sm = smoother.update({"knee_mean": v}, t)
            if counter.update(sm, i, t) is not None:
                counted += 1
        assert counted == n_reps, f"counted {counted}/{n_reps} at {fps} FPS"


# --- features -----------------------------------------------------------------
def test_feature_shapes():
    pts = build_landmarks(PoseParams(), 60.0)
    m = compute_metrics(pts, (0, 0))
    f = frame_features(pts, (0, 0), m)
    assert f.shape == (N_FRAME_FEATURES,)
    seq = np.tile(f, (60, 1))
    assert aggregate_window(seq).shape == (N_WINDOW_FEATURES,)
    assert windows_from_sequence(seq, window=45, stride=10).shape[1] == N_WINDOW_FEATURES


def test_aggregate_window_all_nan_channel_stays_nan():
    frames = np.full((20, N_FRAME_FEATURES), 1.0, dtype=np.float32)
    frames[:, 3] = np.nan
    out = aggregate_window(frames)
    assert math.isnan(out[3])
    assert not math.isnan(out[0])


# --- config -------------------------------------------------------------------
def test_shipped_configs_load_and_validate():
    lib = ExerciseLibrary.from_dir(CONFIGS)
    assert set(lib.names) == {"bicep_curl", "lunge", "pushup", "shoulder_press", "squat"}
    for cfg in lib.values():
        assert cfg.rep_counter is not None
        assert cfg.angle_checks


def test_unknown_metric_is_rejected():
    with pytest.raises(ConfigError, match="unknown metric"):
        parse_exercise_config({
            "name": "x",
            "angle_checks": [{"code": "C", "metric": "not_a_metric", "max": 1}],
        })


def test_angle_check_needs_a_bound():
    with pytest.raises(ConfigError, match="min/max"):
        parse_exercise_config({
            "name": "x", "angle_checks": [{"code": "C", "metric": "knee_mean"}],
        })


def test_rep_counter_direction_sign():
    dec = RepCounterConfig(primary_metric="knee_mean", near_threshold=160, far_threshold=100)
    inc = RepCounterConfig(primary_metric="shoulder_mean", near_threshold=55, far_threshold=145)
    assert dec.direction == -1
    assert inc.direction == 1


# --- rep counter --------------------------------------------------------------
def _drive(counter, values, fps=30.0):
    reps = []
    for i, v in enumerate(values):
        r = counter.update({counter.primary_metric: float(v)}, i, i / fps)
        if r is not None:
            reps.append(r)
    return reps


def _cycle(rest, work, n_reps, period=40):
    out = []
    for i in range(n_reps * period):
        t = trapezoid((i % period) / period)
        out.append(rest + (work - rest) * t)
    return out


def test_counts_reps_on_a_decreasing_signal():
    cfg = RepCounterConfig(primary_metric="knee_mean", near_threshold=158,
                           far_threshold=110, hysteresis=7, min_rep_seconds=0.27)
    c = RepCounter(cfg, "squat")
    assert len(_drive(c, _cycle(178.0, 82.0, 6))) == 6
    assert c.count == 6
    assert c.state == READY


def test_counts_reps_on_an_increasing_signal():
    cfg = RepCounterConfig(primary_metric="shoulder_mean", near_threshold=55,
                           far_threshold=145, hysteresis=8, min_rep_seconds=0.27)
    c = RepCounter(cfg, "shoulder_press")
    assert len(_drive(c, _cycle(50.0, 165.0, 5))) == 5


def test_jitter_at_rest_does_not_produce_reps():
    cfg = RepCounterConfig(primary_metric="knee_mean", near_threshold=158,
                           far_threshold=110, hysteresis=7)
    c = RepCounter(cfg, "squat")
    rng = np.random.default_rng(0)
    _drive(c, 178.0 + rng.normal(0, 2.5, 400))
    assert c.count == 0


def test_shallow_reps_are_still_counted_so_they_can_be_flagged():
    """A partial rep must reach the counter, otherwise it can never be coached."""
    cfg = RepCounterConfig(primary_metric="knee_mean", near_threshold=158,
                           far_threshold=110, hysteresis=7, min_rep_seconds=0.27,
                           count_ratio=0.45)
    c = RepCounter(cfg, "squat")
    reps = _drive(c, _cycle(178.0, 128.0, 4))   # never reaches 110
    assert len(reps) == 4
    assert all(not r.reached_target_zone for r in reps)


def _curl_cfg():
    return ExerciseLibrary.from_dir(CONFIGS)["bicep_curl"].rep_counter


def _hold(value, seconds, fps=14.0):
    return [value] * max(1, int(round(seconds * fps)))


def _ramp(a, b, seconds, fps=14.0):
    n = max(2, int(round(seconds * fps)))
    return [a + (b - a) * (i / (n - 1)) for i in range(n)]


def test_wobble_from_a_half_flexed_hold_is_not_a_rep():
    """Reported live: holding the hands up and jiggling scored repetitions.

    The elbow was held near 92 deg, dipped to 72 and came back. 72 deg is past
    this config's `far_threshold` of 70-ish, so it looked like a deep rep on an
    absolute scale, but the arm only moved 20 deg. A repetition has to travel
    from where the body was resting, not merely reach a deep-looking value.
    """
    c = RepCounter(_curl_cfg(), "bicep_curl")
    seq = _hold(92.0, 1.5) + _ramp(92.0, 72.0, 0.5) + _ramp(72.0, 145.0, 0.6)
    assert len(_drive(c, seq, fps=14.0)) == 0
    assert c.count == 0


def test_lifting_into_position_then_curling_counts_once():
    """The user's exact sequence: get set, curl once, lower. That is ONE rep.

    Raising the hands into the start position is an elbow flexion and cannot be
    told apart from a curl by the elbow angle alone, so it must not be credited
    separately - the rep completes only when the arm returns to rest.
    """
    c = RepCounter(_curl_cfg(), "bicep_curl")
    seq = (
        _hold(175.0, 1.0)                 # standing, arms hanging
        + _ramp(175.0, 92.0, 0.7)         # bring the hands up towards the shoulders
        + _hold(92.0, 0.8)                # settle there
        + _ramp(92.0, 10.0, 0.7)          # the actual curl
        + _hold(10.0, 0.3)
        + _ramp(10.0, 175.0, 1.0)         # lower all the way back down
        + _hold(175.0, 0.6)
    )
    reps = _drive(c, seq, fps=14.0)
    assert len(reps) == 1, f"expected 1 rep, counted {len(reps)}"
    assert reps[0].extreme_value == pytest.approx(10.0, abs=2.0)


def test_full_curls_from_rest_still_count_every_rep():
    """The guard must not cost real repetitions at a low frame rate."""
    c = RepCounter(_curl_cfg(), "bicep_curl")
    seq = []
    for _ in range(6):
        seq += _ramp(175.0, 12.0, 0.8) + _hold(12.0, 0.2) + _ramp(12.0, 175.0, 0.8)
        seq += _hold(175.0, 0.4)
    assert len(_drive(c, seq, fps=14.0)) == 6


def test_partial_curl_from_rest_is_counted_and_flagged():
    """A shallow-at-the-top rep is still a rep - it must reach the detector."""
    c = RepCounter(_curl_cfg(), "bicep_curl")
    seq = []
    for _ in range(4):
        seq += _ramp(175.0, 95.0, 0.8) + _hold(95.0, 0.2) + _ramp(95.0, 175.0, 0.8)
        seq += _hold(175.0, 0.4)
    reps = _drive(c, seq, fps=14.0)
    assert len(reps) == 4
    assert all(not r.reached_target_zone for r in reps), "should be partial, not full"


def test_rep_starting_mid_movement_is_measured_from_where_it_rested():
    """Pose acquired mid-curl must not inflate the measured excursion."""
    c = RepCounter(_curl_cfg(), "bicep_curl")
    # No idle history at all: the very first sample is already half flexed, and
    # the dip stops short of the working position.
    seq = _ramp(95.0, 75.0, 0.5) + _ramp(75.0, 148.0, 0.6)
    assert len(_drive(c, seq, fps=14.0)) == 0


def test_a_rep_reaching_the_working_position_is_trusted_on_depth_alone():
    """The travel guard must not cost reps when the counter starts mid-movement.

    End to end the classifier needs a window before it names the exercise, so the
    counter is frequently created part-way into the first descent with no idle
    history. A rep that genuinely reaches `far_threshold` is evidence enough.
    """
    c = RepCounter(_curl_cfg(), "bicep_curl")
    seq = _ramp(120.0, 15.0, 0.7) + _ramp(15.0, 170.0, 0.9)   # starts mid-descent
    assert len(_drive(c, seq, fps=14.0)) == 1


def test_tiny_twitches_below_count_ratio_are_ignored():
    cfg = RepCounterConfig(primary_metric="knee_mean", near_threshold=158,
                           far_threshold=110, hysteresis=7, count_ratio=0.45)
    c = RepCounter(cfg, "squat")
    _drive(c, _cycle(178.0, 150.0, 5))
    assert c.count == 0


def test_nan_frames_do_not_break_the_state_machine():
    cfg = RepCounterConfig(primary_metric="knee_mean", near_threshold=158,
                           far_threshold=110, hysteresis=7, min_rep_seconds=0.27)
    c = RepCounter(cfg, "squat")
    vals = _cycle(178.0, 82.0, 3)
    vals[45] = float("nan")
    assert len(_drive(c, vals)) == 3


def test_rep_records_rom_and_extreme():
    cfg = RepCounterConfig(primary_metric="knee_mean", near_threshold=158,
                           far_threshold=110, hysteresis=7, min_rep_seconds=0.27)
    c = RepCounter(cfg, "squat")
    rep = _drive(c, _cycle(178.0, 82.0, 2))[0]
    assert rep.extreme_value == pytest.approx(82.0, abs=1.0)
    # The curve starts at the hysteresis crossing (~151), not at the resting
    # 178, so the recorded ROM is the *counted* excursion. Config `min_range`
    # values are calibrated against this same quantity.
    assert 60.0 < rep.rom < 96.0
    assert rep.duration > 0.0


def test_generic_counter_finds_the_moving_joint():
    gc = GenericRepCounter()
    knee = _cycle(178.0, 95.0, 8, period=45)
    reps = 0
    for i, v in enumerate(knee):
        m = {n: 175.0 for n in METRIC_NAMES}
        m["knee_mean"] = v
        m["elbow_mean"] = 176.0
        if gc.update(m, i, i / 30.0) is not None:
            reps += 1
    assert gc.selected == "knee_mean"
    assert reps >= 6, f"generic mode counted only {reps} of 8 cycles"


def test_generic_counter_stays_quiet_when_nothing_moves():
    gc = GenericRepCounter()
    rng = np.random.default_rng(1)
    for i in range(300):
        m = {n: 175.0 + float(rng.normal(0, 0.5)) for n in METRIC_NAMES}
        gc.update(m, i, i / 30.0)
    assert gc.count == 0


# --- mistake detection --------------------------------------------------------
def _squat_cfg():
    return ExerciseLibrary.from_dir(CONFIGS)["squat"]


def test_angle_check_requires_min_frames_then_fires_once():
    cfg = _squat_cfg()
    det = MistakeDetector()
    bad = {n: float("nan") for n in METRIC_NAMES}
    bad.update({"torso_lean": 70.0, "view_frontality": 0.5, "knee_ankle_ratio": 1.0,
                "neck": 170.0, "hip_mean": 120.0})
    raised = [det.check_frame(bad, EXTREME, cfg, i, i / 30.0) for i in range(12)]
    codes = [m.code for frame in raised for m in frame]
    assert codes.count("EXCESSIVE_TORSO_LEAN") == 1
    assert not any(c for frame in raised[:4] for c in frame if c.code == "EXCESSIVE_TORSO_LEAN")


def test_gate_blocks_a_frontal_only_check_in_a_side_view():
    cfg = _squat_cfg()
    side = {n: float("nan") for n in METRIC_NAMES}
    side.update({"knee_ankle_ratio": 0.4, "view_frontality": 0.1,
                 "torso_lean": 20.0, "neck": 170.0, "hip_mean": 120.0})
    det = MistakeDetector()
    for i in range(20):
        det.check_frame(side, EXTREME, cfg, i, i / 30.0)
    assert "KNEE_ALIGNMENT_POOR" not in det.counts

    front = dict(side, view_frontality=0.7)
    det2 = MistakeDetector()
    for i in range(20):
        det2.check_frame(front, EXTREME, cfg, i, i / 30.0)
    assert det2.counts["KNEE_ALIGNMENT_POOR"] == 1


def test_generic_mode_flags_impossible_angles():
    det = MistakeDetector()
    m = {n: float("nan") for n in METRIC_NAMES}
    m.update({"knee_l": 5.0, "knee_r": 175.0})
    for i in range(10):
        det.check_frame(m, READY, None, i, i / 30.0)
    assert det.counts["EXTREME_JOINT_ANGLE"] >= 1


def test_generic_mode_flags_asymmetry():
    det = MistakeDetector()
    m = {n: float("nan") for n in METRIC_NAMES}
    m.update({"knee_diff": 60.0})
    for i in range(12):
        det.check_frame(m, READY, None, i, i / 30.0)
    assert det.counts["LARGE_ASYMMETRY"] == 1


def test_rom_check_flags_a_shallow_rep_and_passes_a_deep_one():
    cfg = _squat_cfg()
    counter = RepCounter(cfg.rep_counter, "squat")
    det = MistakeDetector()
    shallow = _drive(counter, _cycle(178.0, 125.0, 1))[0]
    codes = {m.code for m in det.check_rep(shallow, cfg)}
    assert "INSUFFICIENT_DEPTH" in codes

    counter2 = RepCounter(cfg.rep_counter, "squat")
    det2 = MistakeDetector()
    # period 62 frames matches a realistic ~2 s squat; a shorter cycle here
    # would trip the (legitimate) tempo check and mask what is being tested.
    deep = _drive(counter2, _cycle(178.0, 82.0, 1, period=62))[0]
    assert not {m.code for m in det2.check_rep(deep, cfg)}


def test_rom_check_is_suppressed_for_a_bad_camera_view():
    cfg = _squat_cfg()
    counter = RepCounter(cfg.rep_counter, "squat")
    rep = _drive(counter, _cycle(178.0, 125.0, 1))[0]
    det = MistakeDetector()
    assert not {m.code for m in det.check_rep(rep, cfg, skip_rom=True)}


def test_symmetry_check_uses_metrics_at_the_extreme():
    cfg = _squat_cfg()
    counter = RepCounter(cfg.rep_counter, "squat")
    rep = _drive(counter, _cycle(178.0, 82.0, 1))[0]
    rep.metrics_at_extreme = {"knee_l": 80.0, "knee_r": 115.0, "hip_l": 70.0, "hip_r": 75.0}
    det = MistakeDetector()
    assert "ASYMMETRIC_MOVEMENT" in {m.code for m in det.check_rep(rep, cfg)}


def test_tempo_uses_angular_speed_not_duration():
    """A shallow-but-slow rep must not be reported as rushed."""
    cfg = _squat_cfg()
    det = MistakeDetector()
    c1 = RepCounter(cfg.rep_counter, "squat")
    shallow_slow = _drive(c1, _cycle(178.0, 125.0, 1, period=62))[0]
    assert "TOO_FAST" not in {m.code for m in det.check_rep(shallow_slow, cfg)}

    det2 = MistakeDetector()
    c2 = RepCounter(cfg.rep_counter, "squat")
    quick = _drive(c2, _cycle(178.0, 82.0, 1, period=62), fps=90.0)  # same ROM, 3.7x faster
    assert "TOO_FAST" in {m.code for m in det2.check_rep(quick[0], cfg)}


def test_warnings_expire_after_the_hold_window():
    cfg = _squat_cfg()
    det = MistakeDetector(hold_frames=5)
    bad = {n: float("nan") for n in METRIC_NAMES}
    bad.update({"torso_lean": 70.0, "neck": 170.0, "hip_mean": 120.0})
    for i in range(10):
        det.check_frame(bad, EXTREME, cfg, i, i / 30.0)
    assert det.active(10)
    assert not det.active(30)


# --- end to end ---------------------------------------------------------------
@pytest.mark.parametrize("exercise", ["squat", "pushup", "bicep_curl", "lunge",
                                      "shoulder_press"])
def test_forced_pipeline_counts_synthetic_reps(exercise):
    lib = ExerciseLibrary.from_dir(CONFIGS)
    clip = generate_clip(exercise, reps=6, fault=GOOD, seed=11)
    pipe = MonitorPipeline(lib, forced_exercise=exercise)
    pipe.replay(clip)
    assert abs(pipe.rep_totals().get(exercise, 0) - 6) <= 1


@pytest.mark.parametrize("exercise", ["squat", "pushup", "bicep_curl", "lunge",
                                      "shoulder_press"])
def test_good_form_clips_raise_no_errors(exercise):
    """Guards against threshold drift that would nag a user doing it right."""
    lib = ExerciseLibrary.from_dir(CONFIGS)
    pipe = MonitorPipeline(lib, forced_exercise=exercise)
    pipe.replay(generate_clip(exercise, reps=6, fault=GOOD, seed=23))
    errors = {c for c, n in pipe.detector.summary().items() if n}
    assert not errors, f"good {exercise} produced {errors}"


@pytest.mark.parametrize("exercise,fault_name,code", [
    ("squat", "shallow", "INSUFFICIENT_DEPTH"),
    ("squat", "valgus", "KNEE_ALIGNMENT_POOR"),
    ("squat", "lean", "EXCESSIVE_TORSO_LEAN"),
    ("squat", "fast", "TOO_FAST"),
    ("pushup", "sag", "HIP_SAG"),
    ("pushup", "flare", "ELBOW_FLARE"),
    ("bicep_curl", "swing", "SHOULDER_SWING"),
    ("shoulder_press", "no_lockout", "INCOMPLETE_LOCKOUT"),
    ("shoulder_press", "asymmetric", "ASYMMETRIC_MOVEMENT"),
])
def test_intentional_faults_are_detected(exercise, fault_name, code):
    lib = ExerciseLibrary.from_dir(CONFIGS)
    fault = next(f for f in FAULTS[exercise] if f.name == fault_name)
    pipe = MonitorPipeline(lib, forced_exercise=exercise)
    pipe.replay(generate_clip(exercise, reps=6, fault=fault, seed=31))
    assert code in pipe.detector.summary()


@pytest.mark.parametrize("exercise", ["squat", "pushup", "bicep_curl", "lunge",
                                      "shoulder_press"])
def test_config_matcher_identifies_exercises_without_training_data(exercise):
    """Recognition from the configs alone - no classifier, no recordings.

    This is what makes a newly added exercise work immediately, and what covers
    camera framings the trained model has never seen.
    """
    lib = ExerciseLibrary.from_dir(CONFIGS)
    pipe = MonitorPipeline(lib, bundle=None)          # no model at all
    states = pipe.replay(generate_clip(exercise, reps=8, fault=GOOD, seed=99))
    labels = Counter(s.exercise or "generic" for s in states)
    assert labels.most_common(1)[0][0] == exercise
    assert abs(pipe.rep_totals().get(exercise, 0) - 8) <= 1


@pytest.mark.parametrize("movement", ["toe_touch", "star_jump", "side_bend",
                                      "arm_swing", "idle"])
def test_config_matcher_refuses_untrained_movements(movement):
    """A straight-arm raise must not be called a shoulder press."""
    lib = ExerciseLibrary.from_dir(CONFIGS)
    pipe = MonitorPipeline(lib, bundle=None)
    states = pipe.replay(generate_clip(movement, reps=8, fault=GOOD, seed=99))
    labels = Counter(s.exercise or "generic" for s in states)
    assert labels.most_common(1)[0][0] == "generic"


def test_matcher_label_is_stable():
    """Label chatter strands half-finished reps in the wrong counter."""
    lib = ExerciseLibrary.from_dir(CONFIGS)
    pipe = MonitorPipeline(lib, bundle=None)
    states = pipe.replay(generate_clip("shoulder_press", reps=8, fault=GOOD, seed=99))
    switches = sum(1 for a, b in zip(states, states[1:]) if a.exercise != b.exercise)
    assert switches <= 3, f"label changed {switches} times in one set"


def test_signature_gate_stat_is_explicit():
    """`over: trough` is what separates a press from a straight-arm raise."""
    from src.exercise_config import ConfigError

    lib = ExerciseLibrary.from_dir(CONFIGS)
    press = lib["shoulder_press"]
    elbow = next(g for g in press.signature if g.metric == "elbow_mean")
    assert elbow.over == "trough"
    with pytest.raises(ConfigError, match="invalid 'over'"):
        parse_exercise_config({
            "name": "x",
            "signature": [{"metric": "knee_mean", "max": 1, "over": "average"}],
        })


def test_label_source_is_reported():
    """A rule-based guess must never be presented as a trained recognition."""
    lib = ExerciseLibrary.from_dir(CONFIGS)
    pipe = MonitorPipeline(lib, bundle=None)
    states = pipe.replay(generate_clip("squat", reps=6, fault=GOOD, seed=7))
    sources = {s.label_source for s in states}
    assert sources <= {"generic", "matched", "ambiguous"}
    assert "matched" in sources

    forced = MonitorPipeline(lib, forced_exercise="squat")
    fstates = forced.replay(generate_clip("squat", reps=3, fault=GOOD, seed=7))
    assert {s.label_source for s in fstates} == {"forced"}


def test_generic_mode_tracks_an_unknown_movement():
    """No classifier, unknown movement: still tracks joints and counts cycles."""
    lib = ExerciseLibrary.from_dir(CONFIGS)
    pipe = MonitorPipeline(lib)          # no bundle -> generic for everything
    clip = generate_clip("star_jump", reps=8, fault=GOOD, seed=41)
    states = pipe.replay(clip)
    assert all(s.generic for s in states)
    assert pipe.generic_counter.count >= 6
    last = states[-1]
    assert not math.isnan(last.metrics["knee_mean"])
    assert not math.isnan(last.metrics["elbow_mean"])


def test_session_roundtrip(tmp_path):
    from src.session import Session

    clip = generate_clip("squat", reps=3, fault=GOOD, seed=5)
    path = clip.save(tmp_path / "s.npz")
    back = Session.load(path)
    assert back.label == "squat"
    assert back.expected_reps == 3
    assert back.n_frames == clip.n_frames
    assert np.allclose(back.metrics, clip.metrics, equal_nan=True)
