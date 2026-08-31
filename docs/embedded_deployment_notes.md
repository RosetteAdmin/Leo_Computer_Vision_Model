# Embedded deployment notes

Forward-looking only — no code changes required, and nothing here has been
measured on target hardware. The numbers below are estimates derived from the
laptop measurements plus published throughput for these devices, and they are
labelled as such. Treat them as planning figures, not results.

## Measured baseline (laptop, for scaling)

Intel Core i7-12xxx class CPU, 640x480, `model_complexity=1`, no GPU:

| stage | mean |
|---|---|
| MediaPipe Pose | 16.0 ms |
| Analysis (angles, features, classifier, counter, checks) | 3.1 ms |
| **total** | **~19 ms -> ~52 FPS** |

Two facts drive every decision below:

1. **Pose estimation is 84% of the budget.** Optimisation effort belongs there.
2. **The analysis layer is already trivial** — 3.1 ms, mostly the Random Forest,
   and it only runs every third frame. It will survive a 5-10x slower CPU without
   changes.

## Estimated throughput on target hardware

Scaled from the laptop baseline using published relative CPU throughput for these
boards. **Unverified.**

| device | pose model | est. pose | est. analysis | est. total FPS |
|---|---|---|---|---|
| Raspberry Pi 5 (Cortex-A76 2.4 GHz, 4 core) | BlazePose Lite, 256x256 | 45-65 ms | 8-12 ms | 13-18 |
| Raspberry Pi 4B (Cortex-A72 1.5 GHz) | BlazePose Lite, 256x256 | 90-130 ms | 15-25 ms | 6-9 |
| Pi 4B + Coral USB TPU | Lite, quantised, delegated | 12-20 ms | 15-25 ms | 22-30 |
| Jetson Nano (CUDA, TensorRT) | BlazePose Full | 12-20 ms | 10-15 ms | 30-40 |
| Jetson Orin Nano | BlazePose Full/Heavy | 5-10 ms | 5-8 ms | 60+ |
| Rockchip RK3588 (+ NPU) | Lite via RKNN | 15-25 ms | 8-12 ms | 30-40 |

A Pi 4B without an accelerator will not hit 15 FPS with this pipeline. A Pi 5 is
borderline. Anything with an NPU or CUDA has plenty of headroom.

## Conversion path

### Pose estimator

MediaPipe already ships BlazePose as TFLite, so there is nothing to convert — the
work is in *how* it runs.

- **`model_complexity=0` (BlazePose Lite)** is the first and largest win. Already
  exposed: `python -m src.main --model-complexity 0`. Expect roughly 2x the frame
  rate with modest landmark accuracy loss. **Re-run `python -m src.evaluate` after
  switching**, because noisier landmarks change the threshold behaviour.
- **XNNPACK delegate** is enabled by default in MediaPipe's TFLite build and gives
  a large ARM speedup. Confirm it is active (MediaPipe logs
  `Created TensorFlow Lite XNNPACK delegate for CPU`).
- **INT8 quantisation** of the landmark model: ~2-3x on ARM, some landmark
  jitter. If you quantise, re-derive the config thresholds — quantisation noise
  behaves like a wider hysteresis requirement.
- **Coral Edge TPU** needs full INT8 quantisation and per-op delegate support.
  Verify the whole graph maps to the TPU; a partial mapping can be slower than CPU
  because of transfer overhead.
- **Jetson**: convert to ONNX then TensorRT FP16. Alternatively use NVIDIA's own
  body-pose models, which are tuned for the platform.
- **RK3588**: convert via the RKNN toolkit; INT8 recommended.

### Classifier

Three options, in increasing effort:

1. **Ship the `.pkl` as-is.** Needs scikit-learn on the device (~50 MB with
   dependencies). Fine on a Pi, unnecessary weight for a microcontroller-class
   target.
2. **Export the forest to ONNX** via `skl2onnx` and run it with ONNX Runtime.
   Removes the scikit-learn dependency; ONNX Runtime is well supported on ARM.
3. **Emit the forest as C.** 200 trees of depth ~10 is a few hundred KB of nested
   `if` statements. Eliminates every ML runtime dependency and makes inference
   essentially free. Worth it only for a bare-metal target.

If a neural classifier is ever substituted for the forest, `float32` TFLite with
XNNPACK is the natural choice; at 285 input features it is far below the pose
model's cost either way.

### Analysis layer

`angle_engine`, `rep_counter`, `mistake_detector` and `feedback` are pure NumPy
and Python with no model dependencies. On any Linux SBC they run unmodified. For a
microcontroller they port to plain C almost mechanically — the arithmetic is
`arccos` of normalised dot products, comparisons against thresholds, and a small
state machine. The YAML configs would become generated C structs or be parsed
from flash.

## Recommended changes before an embedded build

Ordered by return on effort:

1. **Drop resolution to 480x360 or 320x240.** BlazePose downsamples internally
   anyway; capturing smaller saves colour conversion and memory bandwidth for
   almost no accuracy cost.
2. **Switch to `model_complexity=0`.** Already a flag.
3. **Cap the frame rate deliberately** (for example 15 FPS) instead of running
   flat out. Rep counting needs perhaps 10-12 FPS to resolve a fast rep; anything
   above that is spent on smoothness. A cap cuts power and heat substantially.
   Note that `min_rep_frames` and the smoothing window are expressed in *frames*
   and must be rescaled if the frame rate changes materially.
4. **Increase `--classify-every`.** At 15 FPS, classifying every 5th frame is
   still 3 decisions per second — far faster than anyone changes exercise.
5. **Reduce the forest.** 100 trees instead of 200 roughly halves classifier cost
   at a small accuracy cost; measure with `python -m src.evaluate`.
6. **Run headless where possible.** Compositing the overlay is not free on a weak
   GPU-less SBC. `--headless` already exists; a kiosk build would render a much
   simpler HUD, or drive LEDs and audio instead of a screen.
7. **Consider frame skipping for pose, not for analysis.** Running pose at 15 FPS
   while the angle engine interpolates is possible but changes the effective
   smoothing; measure before adopting.

## Power and thermal

- Pose estimation saturates all available cores. On a Pi 4/5 in a closed
  enclosure, expect thermal throttling within minutes without a heatsink and fan;
  a throttled Pi loses 20-30% of its clock, which is the difference between usable
  and not.
- Sustained draw: Pi 4B ~5-6 W, Pi 5 ~7-9 W, Jetson Nano ~7-10 W in 10 W mode.
  Add ~1-2 W for a USB camera and Coral accelerator each.
- For a battery-powered or always-on device, add a motion gate: run a cheap frame
  difference and only wake the pose model when something moves. Idle draw drops to
  near zero, which matters far more than peak throughput for a smart-mirror or
  kiosk use case.
- A camera with hardware MJPEG/H.264 output offloads capture from the CPU. Verify
  the decode path is hardware-accelerated, or the win is cancelled.

## Smart-mirror / kiosk considerations

- **Camera placement**: the pipeline needs a ~45-degree view for depth *and*
  valgus. A mirror-mounted camera facing the user head-on will trigger
  `SUBOPTIMAL_CAMERA_VIEW` on every squat. Either mount the camera off-axis, or
  add a second camera and fuse — the config `view` gates make the limitation
  explicit rather than silent, which is the point.
- **Startup**: MediaPipe graph construction takes 300-600 ms on a laptop and
  proportionally longer on ARM. Warm the extractor at boot, not on first use.
- **Multi-user**: MediaPipe Pose tracks one person. A shared device needs a person
  detector plus tracking to pick and hold a subject, which is a substantially
  larger change.
- **Persistence**: `SessionRecorder` already writes compressed `.npz`. A kiosk
  would want per-user history for progress tracking; the per-rep records
  (`Rep.rom`, `Rep.duration`, curve shape) are the natural unit to store.

## Validation checklist for a port

Do not trust a port until all of these pass on the target:

- [ ] `python -m pytest tests -q` — 56 tests, no hardware needed.
- [ ] `python -m src.evaluate` — regenerate the metrics report **on the device**,
      with the device's pose model. Landmark noise differs, so thresholds may need
      retuning.
- [ ] `python -m src.main --benchmark 300` — real sustained FPS, after the device
      has been running long enough to reach its thermal steady state.
- [ ] Rep count against a manual count on live footage, several people.
- [ ] Confirm no network traffic during inference (`tcpdump`, or run with
      networking disabled — the pipeline makes no outbound calls by design).
