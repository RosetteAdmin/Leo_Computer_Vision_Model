# Architecture

## Design principle

The tracking layer does not know what exercise is happening; only the
interpretation layer does, and that layer is configuration, not code.

That single split is what makes "monitor every movement" achievable rather than
marketing. Landmarks and ~50 whole-body metrics are computed on every frame
unconditionally. A known exercise adds a rep signal and a set of threshold rules
on top. An unknown movement loses those rules but keeps everything else.

```
                              +-----------------------------+
  frame  --> MediaPipe Pose --> Universal Angle Engine       |  exercise-agnostic
                              | 33 landmarks -> ~50 metrics  |  always runs
                              +--------------+--------------+
                                             |
                        +--------------------+--------------------+
                        |                                         |
              Exercise classifier                          (no confident match)
              + config from YAML                                  |
                        |                                         |
        config-driven rep counter                    generic cycle counter
        + angle rules + per-rep verdicts             + anatomical/asymmetry/jerk
                        |                                         |
                        +--------------------+--------------------+
                                             |
                                    error codes -> text -> overlay
```

## Components

| Module | Responsibility | Key decision |
|---|---|---|
| `pose_extraction.py` | MediaPipe wrapper | `static_image_mode=False` so MediaPipe tracks between frames instead of re-detecting; this is the single biggest reason it is real-time on CPU. MediaPipe is imported lazily so every other module works on recorded data without it. |
| `angle_engine.py` | All whole-body metrics | Pure function, no state. Emits `NaN` rather than a plausible-looking number when something is unmeasurable. |
| `features.py` | Classifier features | 57 per-frame channels reduced to 5 statistics over a 45-frame window = 285 dims. |
| `exercise_config.py` | YAML loading + validation | Validates every metric name against the engine's catalogue and raises. A typo must not become a silently dead rule. |
| `classifier.py` | Identity + rejection | Two rejection gates plus a rolling vote. |
| `rep_counter.py` | Cycle detection | One state machine, direction-agnostic, config-driven or self-calibrating. |
| `mistake_detector.py` | Form rules | Per-frame streaks, per-rep verdicts, generic anomalies. Structured codes only. |
| `pipeline.py` | Orchestration | Used by the live loop **and** by `evaluate.py`, so the metrics report measures shipped behaviour. |
| `overlay.py` | Rendering | Completely separate from analysis, so the pipeline runs headless. |

## Measurement decisions that were not obvious

These are the places where the naive implementation is wrong. Each is enforced by
a test in `tests/test_pipeline.py`.

### Aspect correction comes first

MediaPipe normalises `x` by width and `y` by height independently. On a 16:9
frame a limb at a true 45 degrees measures about 30. Every consumer of landmark
geometry multiplies `x` by the aspect ratio first. Skipping this puts a
view-dependent bias into every angle.

### Torso length as the unit of distance

Every distance-style metric is divided by the mid-hip to mid-shoulder distance, so
a tall person 3 m away is comparable to a short person at 1.5 m. Feature vectors
are additionally hip-centred, removing position in frame.

### Knee valgus needs a ratio, not an offset

The obvious valgus metric — horizontal offset of the knee from the ankle — does
not work. During a squat both knees travel forward, and in any oblique view that
sagittal travel projects onto the same horizontal axis as the frontal-plane
narrowing you are trying to measure. It swamps the signal: on the bench, a
25 cm knee shift would have been needed to overcome it.

`knee_ankle_ratio` = knee track width / ankle track width. Forward travel is
common to both knees so it cancels, and the ratio is invariant to camera azimuth
because both spans shrink by the same cosine. Below ~0.80 means the knees have
genuinely narrowed. It becomes noise-dominated in a pure side view, hence the
`view_frontality` gate on the check.

### Hip sag needs a gravity reference

`body_align` (the shoulder-hip-ankle angle) is unsigned, so it reads the same for
a sagging push-up and a piked one — and the coaching cues are opposite. Worse, a
`body_align > 180` rule can never fire, because an interior angle is bounded at
180. That bug shipped in the first draft of `pushup.yaml` and was caught by
walking the synthetic bench values by hand.

`hip_drop` measures the hips' vertical offset from the shoulder-ankle line,
signed by the image's `y` axis. "Below" is defined by gravity, not by which way
the person faces, so sag is positive and pike is negative. It is only defined for
a roughly horizontal body and returns `NaN` otherwise — which conveniently means
both push-up checks self-disable if the classifier mislabels a standing movement.

### Tempo must be angular speed, not duration

A rep's duration is measured between the counter's two hysteresis crossings, so a
shallow rep covers less angle, crosses sooner, and *measures as fast* even when the
person is moving slowly. The first implementation used duration and flagged every
deliberately-shallow clip as `TOO_FAST` — precision 0.65, and the coaching advice
was the opposite of what the user needed ("slow down" for someone moving too
slowly).

`TempoCheck` now uses `rom / duration` in degrees per second. On the bench the
separation is clean and the confound is gone:

| exercise | controlled reps (deg/s) | rushed reps (deg/s) | threshold |
|---|---|---|---|
| squat | 73–96 | 152–206 | 124 |
| push-up | 106–131 | 198–245 | 165 |
| bicep curl | 178–216 | 330–403 | 273 |
| lunge | 47–76 | 116–170 | 96 |
| shoulder press | 118–163 | 219–306 | 191 |

Precision went to 1.00. Shallow clips now measure *slower* than good ones, which
is correct.

### The camera view is a first-class input

A single 2D camera cannot measure sagittal flexion from directly in front: at 0
degrees azimuth a deep squat's knee angle projects to nearly 180. Measured on the
bench, an 80-degree true knee angle reads 107 at 25 degrees azimuth, 91 at 45, and
80 at 82.

Rather than silently reporting wrong depth, exercise configs carry a `view` gate.
When it fails the pipeline raises `SUBOPTIMAL_CAMERA_VIEW` and suppresses
range-of-motion verdicts while leaving counting and other checks running. A
~45-degree view is the sweet spot: oblique enough for sagittal angles, frontal
enough for the valgus ratio.

### Direction independence in the rep counter

A squat's knee angle *decreases* toward the working position; a shoulder press's
abduction *increases*. The state machine works on `u = direction * value`, where
`direction` is the sign of `far_threshold - near_threshold`, so `u` always grows
toward the working position and one set of comparisons handles both. Without this
you end up with two nearly-identical counters and a bug in one of them.

### Shallow reps must be counted, not dropped

If a rep only counts when it reaches `far_threshold`, someone doing ten shallow
squats sees a rep count of zero and never learns why. `count_ratio` (default 0.45)
lets a rep that covers most of the way count *and* be flagged `PARTIAL_REP`. The
`reached_target_zone` flag records which happened.

## Classifier and the "unknown" path

Windowed statistics + Random Forest, not a sequence model. The window's
mean/min/max capture posture and its std/drift capture motion, which is most of
what a recurrent model would learn from this input. It trains in about a second
and predicts in 3.9 ms, which matters because you retrain every time you record
more data.

`n_jobs=1` on the forest is deliberate. Inference predicts one window at a time,
and joblib's thread dispatch costs more than the trees do: measured 29 ms/window
at `n_estimators=300, n_jobs=-1` versus 3.9 ms at `n_estimators=200, n_jobs=1`.
That difference alone is the gap between missing and meeting the FPS target.

Rejecting the unknown is harder than classifying the known. A forest asked about a
movement it has never seen returns one of its classes, often confidently. Three
mechanisms:

1. **Confidence gate** — `max(predict_proba) < 0.65` -> unknown.
2. **Novelty gate** — an `IsolationForest` on the training features rejects
   out-of-distribution windows even when the forest is confident.
3. **Rolling vote** — a label needs a strict majority of the last 7 decisions
   before it is displayed, so the on-screen label does not flicker.

The novelty gate has a sharp edge that took measurement to find: **bad form is
legitimately out-of-distribution** relative to well-executed training reps. With a
training set of textbook reps only, the gate rejected sagging push-ups and leaning
lunges, dropping end-to-end rep counting on those clips to zero. Two fixes, both
kept:

- `novelty_override_confidence` (0.85): a very confident prediction survives
  novelty rejection.
- Training data must span the form distribution — varied depth, varied tempo,
  posture jitter, and clips of deliberately poor form labelled with the correct
  exercise. Exercise identity is independent of form quality.

There is a real trade-off here, and it is worth being explicit about the knob.
Widening the training distribution improves robustness to bad form but weakens
unknown-rejection. Measured on the bench:

| training data | unknown rate on untrained movements | end-to-end rep MAE |
|---|---|---|
| textbook reps only | 0.84 | 2.81 |
| + depth/tempo variation | 0.78 | 1.05 |
| + 16-degree posture jitter | 0.40 | 0.92 |
| + explicit `idle` class, 10-degree jitter | 1.00 | 1.01 |
| + bad-form clips with correct labels | 1.00 | 0.17 |

The explicit rest class is the important row. Without it, a person standing
motionless must be assigned to *some* exercise, and the rep counter starts hunting
for reps in landmark noise. `idle` is trained as a label but deliberately has no
`configs/idle.yaml`, so the pipeline maps it to generic mode: joints tracked,
nothing counted, nothing judged.

## Training-free recognition (`matcher.py`)

The trained forest needs labelled recordings of the framing it will see. That is
the right long-term answer but it fails in two ordinary situations: a newly added
exercise with no data yet, and a camera showing only part of the body, where most
of its features are `NaN`. A laptop webcam at desk distance frames head, shoulders
and arms — every leg feature missing.

`ConfigMatcher` handles both using information the configs already carry. It asks
which exercise's rep signal the recent motion actually traces out, scoring three
factors and multiplying them:

| factor | question |
|---|---|
| `coverage` | how much of the configured `near -> far` span the observed range covers |
| `fit` | how much of the observed motion that span explains |
| `signature` | do the config's postural conditions hold? |

`fit` matters because `coverage` alone prefers whichever exercise has the widest
configured span. A curl swings the elbow ~125 degrees; scored against the push-up's
narrower 50-degree window most of that motion is unexplained, so the curl wins.

`signature` carries the discrimination between exercises that share a joint — a
curl and a push-up both bend the elbow ~90 degrees, and only torso orientation
separates them. Two details were found by measurement, not design:

- **Signatures must be judged over the movement, not the current frame.** Sampled
  instantaneously, a press looks exactly like a curl while the arms are down and a
  lunge looks like a squat while the feet are still together. The label chattered
  31 times in one set and neither counter ever completed a rep.
- **Which end of the motion matters differs per condition**, so `over` is stated
  explicitly in the config. Without an elbow condition read at its *trough*, any
  straight-arm lateral raise matched `shoulder_press`.

Measured with no classifier loaded at all: 5/5 shipped exercises identified from
synthetic clips, 5/5 untrained movements correctly left unrecognised.

The result is reported honestly rather than blended into the model's confidence.
`FrameState.label_source` distinguishes `forced`, `model`, `matched` and
`ambiguous`, and the HUD prints which one it was — a rule-based guess and a trained
recognition should not look identical on screen. When the distinguishing posture is
unmeasurable the match is marked ambiguous and the rivals are named, instead of
picking one and hiding the doubt.

## Generic mode

When no exercise is confidently matched:

- The angle engine runs unchanged; all metrics are computed and displayed.
- `GenericRepCounter` watches ten candidate signals, scores each by
  `amplitude x periodicity` (normalised autocorrelation peak over lags 8..N/2),
  derives near/far thresholds from that signal's own rolling 3rd/97th percentiles,
  and feeds the identical state machine.
- Only exercise-agnostic checks run: anatomically implausible angles, left/right
  difference above 35 degrees, angular speed above 800 deg/s.

The periodicity requirement is not decoration. Amplitude alone counts landmark
noise on a rescaled distance metric, and a slow drift (someone walking out of
frame) has enormous amplitude with no cycles at all. Both produced phantom reps
before the periodicity floor was added; `test_generic_counter_stays_quiet_when_nothing_moves`
locks the behaviour down.

## Robustness details

- **Smoothing**: causal moving average over every metric, NaN-aware, with a
  window measured in **seconds** (0.15 s default) rather than frames. A fixed
  5-frame average is 0.17 s at 30 FPS but 0.45 s on a webcam limited to 11 FPS in
  dim light - long enough to flatten the bottom of a squat so the counter never
  sees the depth. Every duration in the pipeline is wall-clock for the same
  reason, including `min_rep_seconds`.
- **Hysteresis**: every threshold crossing needs a margin, so jitter at a
  threshold cannot double-count.
- **Travel from rest**: a repetition must both reach into the working zone *and*
  travel `count_ratio` of the span from where the body was resting, the rest level
  being the median of the idle samples over the preceding 0.6 s. Depth alone is
  not a repetition: a live session held the elbow near 92 deg, dipped to 72 and
  was credited with a rep for 20 deg of movement, because 72 deg looks deep
  against a configured 150 deg start. Partial reps are unaffected - they are
  shallow at the *top* but still travel from rest, so they count and get flagged.
- **Visibility discipline**: BlazePose reports a *guessed* position for occluded
  joints, and believing it made every left/right symmetry check fire on a
  symmetric squat filmed at 45 degrees. Low-visibility landmarks yield `NaN`, and
  left/right comparisons demand higher confidence than a single-sided reading.
- **Streak requirements**: an angle rule must be violated for `min_frames`
  consecutive frames before it fires, and a raised warning is held for ~20 frames
  so it is readable.
- **NaN discipline**: unmeasurable is `NaN`, checks skip `NaN`, the rep counter
  holds its state through dropped frames rather than miscounting.
- **Per-exercise counters**: switching exercise and back preserves each count.
- **Gates**: any check can be made conditional on another metric, which is how
  view-dependent rules avoid firing when they cannot be trusted.

## Performance

Measured on an Intel Core i7-12xxx class laptop CPU, 640x480, no GPU:

| stage | mean |
|---|---|
| MediaPipe Pose (`model_complexity=1`) | 16.0 ms |
| Angle engine + features + classifier + counter + checks | 3.1 ms (median 0.8 ms) |
| **total** | **~19 ms -> ~52 FPS** |

The analysis layer is ~16% of the budget; pose estimation dominates, which is why
the embedded notes focus on it. The p95 of the analysis layer (8.2 ms) is higher
than the median because the classifier only runs every third frame.

## Deliberate non-choices

- **No 3D-CNN or graph network on raw video.** Too slow for CPU real-time and far
  too data-hungry for a dataset you record yourself.
- **No LSTM/BiLSTM yet.** The brief allows escalating to one. It is only worth it
  if windowed features plateau on real data; they capture posture and motion well
  enough that the sequence model is unlikely to be the bottleneck first.
- **No `z` in form logic.** Recorded for future use, but monocular depth is not
  stable enough to base a safety judgement on.
- **No per-joint classifier ensemble.** Worth trying (see the PoseLab reference)
  if a single global classifier underperforms on real data.

## Reference implementations consulted

- `chrisprasanna/Exercise_Recognition_AI` — BlazePose + joint angles + LSTM
  classification with heuristic rep counting.
- `RiccardoRiccio/Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting`
  — the known-versus-manual fallback pattern that generic mode generalises.
- `kochlisGit/Deep-Trainer` — precedent for warning on bad pose and rushed reps.
- `JRKagumba/2D-video-based-exercise-classification` — per-joint classifiers.
- GymCam (research) — multi-exercise recognition and rep counting in
  unconstrained settings; useful benchmark for the robustness goals.
