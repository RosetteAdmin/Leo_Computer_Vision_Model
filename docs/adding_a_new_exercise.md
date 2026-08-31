# Adding a new exercise

One YAML file in `configs/`. No code changes to the pipeline.

What you get immediately: rep counting, angle-mistake detection, per-rep verdicts,
the live angle table, coaching text. What needs training data: the automatic
on-screen *name*. Until you record clips and retrain, run with
`--exercise <name>`.

---

## 1. Pick the rep signal

Choose the metric with the largest, cleanest swing between rest and the working
position. Any metric the angle engine emits is valid — run this to see them all
for a live pose:

```bash
python -m src.main --exercise squat        # press "a" to toggle the angle table
```

Or read the values off a recorded clip:

```python
from src.session import Session
s = Session.load("data/raw_sessions/my_clip.npz")
series = s.metric_series("hip_mean")
print(series.min(), series.max())
```

Typical choices:

| exercise family | primary metric |
|---|---|
| squat, lunge, step-up, glute bridge | `knee_mean` or `hip_mean` |
| push-up, dip, bench, curl, row | `elbow_mean` |
| overhead press, lateral raise, front raise | `shoulder_mean` |
| plank hold (no reps) | omit `rep_counter` entirely |
| deadlift, hip hinge, good morning | `hip_mean` |

The signal may increase *or* decrease toward the working position — the state
machine infers the direction from the sign of `far_threshold - near_threshold`.

## 2. Measure the two thresholds

Record one clip, then read off the resting and working values:

```bash
python -m src.main --record glute_bridge --exercise squat --expected-reps 10
```

```python
import numpy as np
from src.session import Session
s = Session.load("data/raw_sessions/glute_bridge_good_....npz")
v = s.metric_series("hip_mean")
print("rest  ~", np.percentile(v, 90))
print("worked ~", np.percentile(v, 5))
```

- `near_threshold` — the resting value, pulled ~10-15 units toward the middle.
- `far_threshold` — the value that means "you got there", ~10-15 units short of
  the best rep.
- `hysteresis` — roughly 7% of the near-to-far span. Too small double-counts on
  jitter; too large drops shallow reps.

## 3. Write the config

```yaml
name: glute_bridge                 # MUST match the filename stem
display_name: Glute Bridge
description: >
  Supine hip extension. Rep signal is the hip angle, ~95 deg down and
  ~170 deg at full extension.

# Extra metrics shown in the live angle table for this exercise.
tracked_metrics: [hip_mean, knee_mean, torso_lean, hip_diff]

# Optional. Suppresses range-of-motion verdicts (and raises
# SUBOPTIMAL_CAMERA_VIEW) when the camera cannot support the measurement.
view:
  - metric: view_frontality
    max: 0.80

# Optional. Identifies the movement for the training-free ConfigMatcher, so the
# exercise is recognised before you have recorded any data for it. Needed when
# another exercise shares this one's primary joint. `over` picks which end of the
# recent motion the condition is about: peak (default), trough, or median.
signature:
  - metric: torso_lean      # lying down, unlike a standing hip hinge
    min: 60
  - metric: knee_mean       # knees stay bent throughout
    max: 120
    over: peak

rep_counter:
  primary_metric: hip_mean
  near_threshold: 110       # value at rest
  far_threshold: 158        # value in the worked position
  hysteresis: 7
  min_rep_seconds: 0.27     # SECONDS, never frames - see the note below
  max_rep_seconds: 13.0     # abandon a rep that never came back
  count_ratio: 0.45         # fraction of the span needed to count at all

# --- per-frame angle mistakes ---
angle_checks:
  - code: KNEE_ALIGNMENT_POOR
    metric: knee_ankle_ratio
    phases: [moving_away, extreme, returning]
    min: 0.80                     # violated when the value drops below this
    min_frames: 5                 # consecutive frames before it fires
    severity: error               # info | warning | error
    highlight: [left_knee, right_knee]    # joints drawn in red
    gates:                        # check is skipped unless all gates pass
      - metric: view_frontality
        min: 0.45

  - code: EXCESSIVE_TORSO_LEAN
    metric: torso_lean
    phases: [extreme]
    max: 25                       # violated when the value exceeds this
    min_frames: 5
    severity: warning

# --- per-rep mistakes ---
rep_checks:
  rom:
    metric: hip_mean
    target_extreme: 152      # the rep must reach at least this far
    min_range: 35            # minimum swing within the rep
    code: PARTIAL_REP
  tempo:
    max_speed: 110           # deg/s of the ROM metric - see calibration below
    min_seconds: 0.25        # hard floor for physically impossible reps
    max_seconds: 12.0
    relative_max_speed_ratio: 1.9   # also flag vs the user's own median speed
  symmetry:
    pairs: [[hip_l, hip_r], [knee_l, knee_r]]
    max_diff: 15             # degrees at the extreme of the rep
    code: ASYMMETRIC_MOVEMENT
  consistency:
    metric: hip_mean
    min_correlation: 0.82    # rep-shape correlation vs the user's average
    min_reps: 3
```

Then verify it loads:

```bash
python -c "from src.exercise_config import load_exercise_config as L; print(L('configs/glute_bridge.yaml'))"
python -m src.main --exercise glute_bridge
```

The loader validates every metric name against the angle engine's catalogue and
raises `ConfigError` on a typo. It will not silently ignore a broken rule.

---

## Field reference

### `rep_counter`

| key | meaning |
|---|---|
| `primary_metric` | any metric from `angle_engine.METRIC_NAMES` |
| `near_threshold` | value at rest / ready |
| `far_threshold` | value in the worked position (may be above or below `near`) |
| `hysteresis` | margin required to cross a threshold |
| `min_rep_seconds` | shorter excursions are jitter, not reps |
| `max_rep_seconds` | abandon a rep that never returns |
| `count_ratio` | fraction of the near->far span needed to count at all (default 0.45). Applies twice: the peak must get this far into the span, **and** the movement must travel this far from the resting position |

> **Use seconds, not frames.** `min_rep_frames` / `max_rep_frames` are still
> accepted and converted at 30 FPS, but they are not portable. A webcam in dim
> light drops to 11 FPS, where "8 frames" silently becomes 0.73 s and rejects
> genuine repetitions. Every duration in the pipeline is wall-clock for this
> reason, including the smoothing window.

### `signature[]`

Used only by `src/matcher.py` for training-free recognition. Judged over a window
of recent motion, not a single frame - sampled instantaneously a shoulder press
looks exactly like a curl during the part of the rep where the arms are down.

| key | meaning |
|---|---|
| `metric` | any metric from `angle_engine.METRIC_NAMES` |
| `min` / `max` | the bound the statistic must satisfy |
| `over` | `peak` (default), `trough`, or `median` of the recent window |

Choose `over` deliberately: "the feet never got wide" is about the **peak** of
stance width, while "the elbow bent at some point" is about the **trough** of the
elbow angle. A condition whose metric is unmeasurable (hips out of frame, say)
scores as *partial* rather than pass or fail, so the match is reported with lower
confidence instead of being asserted or thrown away.

### `angle_checks[]`

| key | meaning |
|---|---|
| `code` | structured error code; add text for it in `src/feedback.py` |
| `metric` | any engine metric |
| `phases` | `any`, `ready`, `moving_away`, `extreme`, `returning`, `moving` |
| `min` / `max` | at least one required; violated outside the range |
| `min_frames` | consecutive violating frames before it fires (default 3) |
| `severity` | `info`, `warning`, `error` (drives the overlay colour) |
| `highlight` | landmark names drawn in red, from `landmarks.LANDMARK_NAMES` |
| `gates[]` | `{metric, min, max}` preconditions; the check is skipped unless all pass |

`phases` matters more than it looks. `ELBOW_FLARE` on a push-up is only meaningful
at the bottom: at the top the upper arm is perpendicular to the torso *by
definition*, so an `any`-phase rule would fire on every good rep.

### `rep_checks`

| block | keys |
|---|---|
| `rom` | `metric`, `target_extreme`, `min_range`, `code` |
| `tempo` | `max_speed` (deg/s), `min_seconds`, `max_seconds`, `relative_max_speed_ratio`, `fast_code`, `slow_code` |
| `symmetry` | `pairs` (list of two metric names), `max_diff`, `code` |
| `consistency` | `metric`, `min_correlation`, `min_reps`, `code` |

Omit any block you do not want. Omit `symmetry` entirely for unilateral movements
— `lunge.yaml` does exactly that, because the left and right knee angles are
*supposed* to differ.

---

## Calibrating thresholds from data

Guessing thresholds produces a system that nags people with good form. Measure
instead. Record good clips *and* clips where you exaggerate the specific fault,
then compare.

```python
import numpy as np
from collections import defaultdict
from src.exercise_config import ExerciseLibrary
from src.pipeline import MonitorPipeline
from src.session import load_sessions

lib = ExerciseLibrary.from_dir("configs")
name = "glute_bridge"

rows = defaultdict(list)
for s in load_sessions("data/test_faults"):
    if s.label != name:
        continue
    pipe = MonitorPipeline(lib, forced_exercise=name)
    pipe.replay(s)
    for r in pipe.counter_for(name).reps:
        rows[s.fault].append((r.extreme_value, r.rom, r.rom / r.duration))

for fault, v in sorted(rows.items()):
    a = np.array(v)
    print(f"{fault:<12} extreme {a[:,0].min():6.0f}-{a[:,0].max():6.0f}  "
          f"rom {a[:,1].min():5.0f}-{a[:,1].max():5.0f}  "
          f"speed {a[:,2].min():5.0f}-{a[:,2].max():5.0f}")
```

Set each threshold **between** the good and faulty ranges, nearer the faulty side
so borderline-good reps are not flagged. This is exactly how the shipped
`max_speed` values were derived; the numbers are recorded in the config comments
and in `docs/architecture.md`.

For per-frame rules, plot the metric across both clip types:

```python
s_good = load_sessions("data/raw_sessions")[0]
s_bad = load_sessions("data/test_faults")[0]
print(np.nanpercentile(s_good.metric_series("torso_lean"), [50, 95, 100]))
print(np.nanpercentile(s_bad.metric_series("torso_lean"),  [50, 95, 100]))
```

---

## Teaching the classifier the new exercise

Only needed for automatic recognition. Rep counting and mistake detection already
work with `--exercise`.

```bash
# 6-10 clips, 20-30 reps each. Vary people, distance, camera angle, lighting,
# depth and tempo. Include a few sloppy-but-still-correct-exercise clips.
python -m src.main --record glute_bridge --exercise glute_bridge \
    --subject alice --expected-reps 20

python -m src.train --data data/raw_sessions
python -m src.evaluate
```

Three things that will bite you:

1. **Include imperfect reps, labelled with the correct exercise name.** Identity
   is independent of form quality. Train only on textbook reps and the novelty
   gate rejects sloppy ones as "unknown", switching off feedback for the people
   who need it most.
2. **Record a `standing_still` / rest class and give it no config.** The pipeline
   maps any label without a config to generic mode. Without a rest class, idle
   frames must be assigned to some exercise.
3. **Check the new exercise is separable.** If the confusion matrix in
   `docs/metrics_report.md` mixes it with a neighbour, the two probably share a
   primary joint; add a distinguishing metric to `CLASSIFIER_METRICS` in
   `angle_engine.py` or record more varied clips.

## Adding a new error code

1. Use the code in a config's `angle_checks[].code` or a `rep_checks` `code`.
2. Add an entry to `MESSAGES` and `SHORT_LABELS` in `src/feedback.py`.

Step 2 is optional — unknown codes fall back to a readable auto-generated string —
but a written cue is worth far more to the user than `HIP_HINGE_TOO_EARLY`.

## Exercises with no repetitions

Isometric holds (plank, wall sit, hollow hold) work: omit `rep_counter` and
`rep_checks`, keep `angle_checks` with `phases: [any]`. The pipeline then runs the
generic cycle counter (which correctly finds nothing to count) while the angle
rules police the hold.
