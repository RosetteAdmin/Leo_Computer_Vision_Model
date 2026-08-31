# Engineering Review Pack — Exercise Form Monitor

For the Leosphere Tech engineering team. This document is written to be read
before the code: it explains what the prototype does, how to run it in about five
minutes, exactly which claims are measured and which are not, and what we believe
must change to meet production standards on the AuraFit device.

Nothing in here is marketing. Where a number is an estimate it says so.

---

## 1. What this is

A real-time exercise form monitor: webcam in, pose landmarks out, then joint
angles, exercise recognition, repetition counting and form-mistake feedback —
all on CPU, fully offline. There is no network call anywhere in the runtime.

In AuraFit terms this is a prototype of the **posture-correction subsystem**: the
part that watches the user, decides whether the movement is correct, and says why
it is not. It is intended to run on the Edge-AI HMI board.

What it does **not** yet do, and would need for the shipping product, is covered
in §6.3 — most importantly it has no output path to drive resistance hardware.

### Design principle worth knowing before you read any code

Form rules live in YAML, not in Python. One file per exercise under `configs/`
declares the rep signal, its thresholds, and every mistake check. Adding an
exercise is a config change, not a code change, and `src/matcher.py` can then
recognise that exercise **without retraining any model**. For a catalogue of
physiotherapy routines this is the property that matters most.

```
configs/squat.yaml        <- thresholds, checks, tempo limits, symmetry pairs
configs/bicep_curl.yaml
configs/lunge.yaml  configs/pushup.yaml  configs/shoulder_press.yaml
```

---

## 2. Quick start

Verified on Windows 11, **Python 3.10.11**, CPU only, no GPU.

```bash
# 1. Create an isolated environment (recommended - see the caveat below)
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

# 2. Runtime + training dependencies
pip install -r requirements.txt

# 3. Test dependency - NOT in requirements.txt, see issue S-4 in section 5
pip install pytest

# 4. Confirm the install, the camera and the model in one step
python -m tools.check_camera
```

`check_camera` verifies dependencies, configs, the classifier, the camera, and
then measures real capture rate and pose-detection quality for 90 frames. Start
here; it catches most setup problems before they look like application bugs.

### Running it

```bash
python -m src.main                      # webcam, recognises the exercise itself
python -m src.main --exercise squat     # force one exercise, skip recognition
python -m src.main --display-width 1500 # larger window (capture stays small)
python -m src.main --source clip.mp4    # analyse a video file
python -m src.main --replay data/test_faults/squat_shallow_00.npz
python -m src.main --headless           # no window, for a kiosk or a board
```

Keys while running: `q` quit, `r` reset counters, `a` toggle the joint-angle
table, `m` mirror, `f` fullscreen, `p` pause, `s` save the recording.

If the camera does not open, try `--source 1` or `--source 2`. On Windows the
DirectShow backend is selected automatically to avoid a multi-second MSMF stall.

### Recording data and retraining

```bash
python -m src.main --record squat --fault good --expected-reps 10
python -m src.train                     # retrain on everything in data/raw_sessions
python -m src.evaluate                  # regenerate docs/metrics_report.md
```

### Diagnostics

```bash
python -m tools.check_camera            # dependencies, camera, framing, FPS
python -m tools.check_capture_rate      # true capture FPS and frame brightness
python -m tools.inspect_reps <session.npz> --exercise bicep_curl
```

`inspect_reps` replays a recording through the real pipeline and prints the rep
signal beside every decision the counter made — when it armed, where it peaked,
what the other joints were doing. It is how the counting bug in §5 was found and
confirmed rather than guessed at.

---

## 3. Verifying our claims yourself

Please do not take the numbers in §4 on trust. These four commands regenerate all
of them from scratch:

```bash
python -m pytest tests -q               # 81 tests, ~17 s, no hardware needed
python -m src.evaluate                  # rebuilds docs/metrics_report.md
python -m src.main --benchmark 300      # sustained FPS on your hardware
python -m tools.check_camera            # your camera, your lighting
```

`src/pipeline.py` is deliberately the same code path live and during evaluation,
so a replayed recording produces exactly what the live app produced. That is what
makes `inspect_reps` trustworthy as a debugging tool.

---

## 4. Status: what is measured, and what is not

### Read this first

**All accuracy figures come from synthetic data. The performance figures are
real.** The classifier is trained on a synthetic pose generator
(`src/synthetic.py`), not on recorded humans. Real-world accuracy will be lower
and we do not currently know by how much. Treat the accuracy table as evidence
that the pipeline is wired correctly, not as a field result.

### Performance — measured, real

| measurement | value | conditions |
|---|---|---|
| Pose estimation (BlazePose full) | 16.0 ms | i7-12xxx class, 640x480, no GPU |
| Analysis layer (angles, features, classifier, counter, checks) | 3.1 ms mean, 0.8 ms median | same |
| Theoretical total | ~19 ms, ~52 FPS | same |
| **Observed end to end, live webcam** | **15.6 FPS** | laptop webcam, indoor light |
| Observed per-frame pose+analysis, live | 25.1 ms mean, 31.1 ms p95 | same session |

The gap between 52 FPS capability and 15.6 FPS observed is the **camera**, not the
pipeline — the webcam does not deliver frames faster in this lighting. Worth
knowing before anyone optimises the wrong layer.

Pose estimation is ~84% of the compute budget. Optimisation effort belongs there.

### Accuracy — synthetic data only

| metric | value |
|---|---|
| Exercise classification accuracy | 0.9962, macro-F1 0.9968 |
| Rejection of untrained movements ("unknown" rate) | 1.000 |
| Rep counting, exercise given | MAE 0.05, exact 96.4%, within ±1 98.8% |
| Rep counting, end to end | MAE 0.29, exact 72.6%, within ±1 98.8% |
| Form-mistake detection | macro precision 0.993, recall 1.000 over 11 codes |

The first five rows are regenerated by `python -m src.evaluate` into
`docs/metrics_report.md`. Training-free recognition is covered separately by the
test suite rather than that report — `matcher.py` with no model loaded identifies
5/5 shipped exercises and refuses 5/5 untrained movements
(`tests/test_pipeline.py::test_config_matcher_identifies_exercises_without_training_data`
and `::test_config_matcher_refuses_untrained_movements`).

### Verified on one real recording

A 70-second live bicep-curl session (`data/real_sessions/`) was replayed through
the pipeline. Auto-recognition held the correct label for 84% of frames (79% from
the trained model, 5% from the config matcher) and changed its mind exactly once.
Rep counts were identical whether the exercise was auto-detected or forced.

That is one session, one subject, one exercise. It is not a validation set.

---

## 5. Known issues and open items

Numbered so they can be referenced in review comments.

### Correctness

**C-1 — Residual rep-counting regression. Open.**
A bug was fixed where holding the hands part-way up and jiggling scored
repetitions: the counter checked only whether the joint angle got *deep*, never
whether it had *travelled*. On the real recording it credited a rep for 20° of
elbow movement, because 72° looks deep against a configured 150° start. The fix
requires a rep that falls short of the working position to also travel a minimum
distance from the resting position.

The fix is correct on real data and has 6 regression tests. **However**, synthetic
end-to-end exact-match dropped from 82.1% to 72.6% (MAE 0.19 → 0.29). Within ±1
rep is unchanged at 98.8%, so it is losing one rep on some clips — we believe the
first rep of a set, where the classifier needs a window before it names the
exercise and the counter therefore starts mid-movement with no rest baseline to
measure against. **Not yet resolved.** See `src/rep_counter.py::_complete`.

**C-2 — Anatomically impossible joint angles at close range.**
At the current camera distance the elbow reads 0.6–6° at full curl. A human elbow
cannot flex below ~30°; this is 2D foreshortening as the forearm points at the
camera. It inflates measured range of motion and makes the ROM target trivially
easy to satisfy. Mitigation today is a framing warning on screen; a real fix needs
either depth (`z`) or a calibrated camera model.

**C-3 — 15–20% of frames have no usable pose** at the framing used in testing,
because hips and legs leave the frame. Metrics that depend on missing joints are
`NaN` by design and the checks skip them, so this degrades gracefully — but squats
and lunges cannot be detected at all without legs in shot.

**C-4 — Camera angle is a first-class constraint.** Squat depth and knee valgus
need roughly a 45° view. A head-on mirror-mounted camera raises
`SUBOPTIMAL_CAMERA_VIEW` on every squat. This is surfaced deliberately rather than
silently producing wrong verdicts, but it constrains industrial design.

### Engineering standards

**S-1 — Not under version control.** There is no `.git` directory. This is the
first thing to fix before any team touches it.

**S-2 — No packaging or tooling.** All absent:
`pyproject.toml`, `setup.py`, `ruff`/`flake8`, `mypy`, `pre-commit`, `LICENSE`,
`Dockerfile`, `Makefile`, CI config. The project is a collection of scripts, not
an installable component.

**S-3 — No logging.** 44 `print()` calls in `src/`, zero uses of `logging`. An
embedded device needs levelled, structured, redirectable logs.

**S-4 — Dependency drift.** `requirements.txt` pins versions that are not the ones
installed and measured:

| package | pinned | actually installed |
|---|---|---|
| opencv-python | 4.11.0.86 | **4.13.0.92** |
| scipy | 1.14.1 | **1.15.3** |
| PyYAML | 6.0.2 | 6.0.3 |
| pytest | *commented out* | 9.1.1 |

Every measurement in §4 was taken against the installed set, not the pinned set.
`pytest` being commented out also means a clean `pip install -r requirements.txt`
cannot run the test suite. Recommended fix: pin what is actually used, and split
runtime / training / dev into separate requirement sets or a `pyproject.toml`
with extras.

**S-5 — No test coverage measurement.** 81 tests pass, but nothing reports what
fraction of `src/` they exercise.

---

## 6. Optimisation plan for production standards

Ordered within each area by return on effort. Estimates are flagged as estimates.

### 6.1 Engineering standards — cheapest, most visible

| # | change | why | effort |
|---|---|---|---|
| 1 | `git init`, commit, branch policy | S-1. Nothing else is safe without it | minutes |
| 2 | `pyproject.toml` with pinned runtime + `[dev]`/`[train]` extras | S-2, S-4. Makes it an installable component | ~2 h |
| 3 | Replace `print()` with `logging`, one logger per module | S-3. Required for a headless device | ~3 h |
| 4 | `ruff` + `mypy` config, then fix what they report | Consistency; catches real NaN/None bugs | ~4 h |
| 5 | CI running `ruff`, `mypy`, `pytest` on push | Stops regressions like C-1 reaching review | ~2 h |
| 6 | `pytest --cov`, publish the number | S-5 | ~1 h |
| 7 | `Dockerfile` for reproducible evaluation | Makes §4 reproducible on any machine | ~2 h |

### 6.2 Resource optimisation — for the Edge-AI board

Two findings in the current code, both verifiable:

**R-1 — The display path allocates a new full-size buffer every frame.**
`src/main.py::_for_display` calls `cv2.resize` into a fresh array each frame. At
`--display-width 1500` that is 1500×1125×3 ≈ **5.06 MB per frame**, roughly
**79 MB/s** of allocation and copying at the observed 15.6 FPS — before any
analysis runs. On a device with a 20-inch touchscreen this path is permanent, not
a debugging convenience. Fix: resize into a preallocated buffer, reused per frame.

**R-2 — `src/overlay.py::_panel` allocates per panel, per frame** via
`np.full_like(roi, DARK)`, several times a frame. Fix: blend with a scalar instead
of materialising a filled array.

**R-3 — There is no memory measurement anywhere in the project.** Timing is
well covered; peak RSS is not. For a fixed-RAM board that is the number that
decides whether it ships. Recommend a `--profile` mode reporting peak RSS
alongside per-stage timing.

Then the pose model, which is 84% of the budget. `docs/embedded_deployment_notes.md`
covers this in depth; the headline options, **all estimates, none measured on
target hardware**:

| lever | expected effect | already available? |
|---|---|---|
| `--model-complexity 0` (BlazePose Lite) | ~2x frame rate, some landmark accuracy loss | yes, a flag |
| Capture at 480x360 or 320x240 | Saves colour conversion and memory bandwidth | yes, `--width`/`--height` |
| Cap frame rate at ~15 FPS | Large power and thermal saving; 10–12 FPS is enough to resolve a fast rep | no |
| `--classify-every 5` | Still 3 decisions/second at 15 FPS | yes, a flag |
| INT8 quantisation of the landmark model | est. 2–3x on ARM; **thresholds must be re-derived**, quantisation noise behaves like wider hysteresis | no |
| NPU / Coral / TensorRT delegate | Board-specific; the only route to 30 FPS+ on ARM | no |
| Reduce forest 200 → 100 trees | ~halves classifier cost, small accuracy cost | config |
| Motion gate: wake pose only when the frame changes | Near-zero idle draw, matters more than peak FPS for an always-on device | no |

Important caveat for anyone changing the frame rate or the model: **re-run
`python -m src.evaluate` afterwards.** Noisier landmarks change threshold
behaviour. The durations in the pipeline are wall-clock (seconds, not frames)
specifically so a frame-rate change does not silently alter rep counting, but the
*thresholds* still need revalidating.

### 6.3 Scalability — the architectural gaps

These are the ones that need a design decision, not just effort.

| gap | current state | what AuraFit needs |
|---|---|---|
| **Hardware output loop** | The pipeline produces pixels and text only | Auto-adjusting resistance needs rep/form state to drive the motor in real time. This is the largest gap: the headline safety feature is a closed loop and there is no output interface at all |
| **Multi-user** | MediaPipe Pose tracks exactly one person | A shared device needs person detection plus tracking to pick and hold a subject |
| **Persistence / user model** | Sessions are `.npz` blobs on disk; no users, no history, no database | Progress tracking and the Coach Dashboard need per-user history. `Rep` records (ROM, duration, curve shape) are the natural unit to store |
| **API boundary** | It is a CLI application; `main.py` owns the loop | A touchscreen UI, a dashboard and a resistance controller cannot all consume a CLI. The pipeline should be a library or service with a stable API |
| **Sensor fusion** | Vision only | No abstraction for the smart floor or resistance/force sensors. Force and velocity metrics would be far more reliable from the hardware than inferred from video |
| **Exercise catalogue** | 5 exercises; thresholds hand-calibrated per exercise | Config-driven design already scales well and `matcher.py` needs no retraining, but calibration is manual. A calibration tool would be needed for a large physio catalogue |

The recommended sequence is: extract a stable pipeline API first (it unblocks the
UI, the dashboard and the hardware loop simultaneously), then the output interface
for resistance, then persistence, then multi-user.

---

## 7. Where to look in the code

Suggested reading order for a reviewer:

```
src/pipeline.py          START HERE. frame in -> FrameState out.
                         The same code runs live and in evaluation.
configs/squat.yaml       What a form specification looks like
src/angle_engine.py      The whole-body metric engine (all joints, always)
src/rep_counter.py       Hysteresis state machine; see C-1
src/matcher.py           Training-free recognition from configs
src/mistake_detector.py  Per-frame, per-rep and generic anomaly checks
src/overlay.py           Rendering only; no analysis. See R-2
src/main.py              The application loop. See R-1
tests/test_pipeline.py   81 tests; the counting bug cases are named for it
docs/architecture.md     Design decisions, and the measurements behind them
docs/embedded_deployment_notes.md   Port planning; explicitly unverified
docs/adding_a_new_exercise.md       Worked example + calibration recipe
docs/metrics_report.md   Generated by src/evaluate.py
```

`docs/architecture.md` is the most useful document for understanding *why* the
design is as it is — including several decisions that were changed because
measurement contradicted the original intuition.

---

## 8. Questions we would like reviewed

1. **Target hardware.** Which board, how much RAM, is there an NPU or GPU? Every
   figure in §6.2 is an estimate until we measure on the real thing.
2. **The resistance control loop.** What interface will the pipeline write to, and
   what latency budget does it have? This determines the API in §6.3.
3. **Is 2D pose sufficient?** C-2 shows joint angles are unreliable when a limb
   points at the camera. Is a depth camera or a second camera an option?
4. **Camera placement.** C-4 constrains it to roughly 45° off-axis for squats.
   How does that interact with the device's industrial design?
5. **Accuracy validation.** We can report synthetic accuracy only. What real
   dataset, subject count and acceptance criteria would you want before this is
   considered validated?
