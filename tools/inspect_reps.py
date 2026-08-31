"""Print the rep signal of a recorded session next to the counter's decisions.

    python -m tools.inspect_reps data/real_sessions/auto_good_...npz --exercise bicep_curl

Use this when a live count disagrees with what you actually did. It replays the
recording through the real pipeline and shows, per rep, when the state machine
armed, where it peaked, and what the other joints were doing at that moment - so
a false count can be attributed instead of guessed at.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.exercise_config import ExerciseLibrary
from src.pipeline import MonitorPipeline
from src.session import Session


def sparkline(values: np.ndarray, lo: float, hi: float) -> str:
    blocks = " .:-=+*#%@"
    out = []
    for v in values:
        if not np.isfinite(v):
            out.append("?")
            continue
        t = (v - lo) / max(hi - lo, 1e-6)
        out.append(blocks[int(np.clip(t, 0, 1) * (len(blocks) - 1))])
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--exercise", default=None, help="force this config")
    ap.add_argument("--model", default="models/exercise_classifier.pkl")
    ap.add_argument("--no-classifier", action="store_true",
                    help="matcher only, no trained model")
    ap.add_argument("--metric", default=None, help="signal to display")
    ap.add_argument("--every", type=int, default=1, help="downsample the trace")
    args = ap.parse_args()

    lib = ExerciseLibrary.from_dir(args.configs)
    session = Session.load(args.session)

    # Load the classifier unless told not to, so a replay reflects what the live
    # app actually does in auto mode. Without it this tool measured the matcher
    # alone and quietly disagreed with the behaviour it was meant to explain.
    bundle = None
    if not args.no_classifier and args.exercise is None:
        try:
            from src.classifier import ClassifierBundle
            bundle = ClassifierBundle.load(args.model)
        except Exception as exc:                       # noqa: BLE001
            print(f"no classifier ({exc}); matcher only")

    pipe = MonitorPipeline(lib, bundle=bundle, forced_exercise=args.exercise)
    states = pipe.replay(session)

    if args.exercise is None:
        from collections import Counter
        labels = Counter((s.exercise or "none", s.label_source or "-") for s in states)
        print("auto-detection over the recording:")
        for (name, src), n in labels.most_common():
            print(f"  {name:<16} via {src:<10} {n:5d} frames "
                  f"({100.0 * n / len(states):5.1f}%)")
        switches = sum(1 for a, b in zip(states, states[1:]) if a.exercise != b.exercise)
        print(f"  label changed {switches} times")

    fps = session.n_frames / max(float(session.timestamps[-1]), 1e-6)
    print(f"{Path(args.session).name}: {session.n_frames} frames, "
          f"{session.timestamps[-1]:.1f}s, ~{fps:.1f} FPS, label={session.label}")
    print(f"totals: {pipe.rep_totals()}")

    metric = args.metric or next((s.primary_metric for s in states if s.primary_metric), None)
    if metric is None:
        print("no rep signal was selected")
        return 0

    vals = np.array([s.metrics.get(metric, np.nan) for s in states], dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        print(f"{metric} is never measurable in this recording")
        return 0
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    print(f"\nsignal: {metric}  range {lo:.0f}..{hi:.0f}")

    cfg = lib[args.exercise] if args.exercise else None
    if cfg and cfg.rep_counter:
        rc = cfg.rep_counter
        print(f"config: near={rc.near_threshold} far={rc.far_threshold} "
              f"hyst={rc.hysteresis} count_ratio={rc.count_ratio} "
              f"min_rep={rc.min_rep_seconds}s")

    print("\ntrace (one char per frame, '?' = unmeasurable):")
    step = max(1, args.every)
    for start in range(0, len(vals), 120 * step):
        chunk = vals[start:start + 120 * step:step]
        t0 = float(session.timestamps[start])
        print(f"  {t0:6.1f}s |{sparkline(chunk, lo, hi)}|")

    print("\nreps the counter accepted:")
    reps = [s.new_rep for s in states if s.new_rep is not None]
    if not reps:
        print("  none")
    for r in reps:
        peak_shoulder = r.metrics_at_extreme.get("shoulder_mean", float("nan"))
        peak_torso = r.metrics_at_extreme.get("torso_lean", float("nan"))
        print(f"  #{r.index:<3} t={r.start_time:6.2f}..{r.end_time:6.2f}s "
              f"({r.duration:4.2f}s)  start={r.start_value:6.1f} "
              f"peak={r.extreme_value:6.1f}  ROM={r.rom:5.1f}  "
              f"full={'Y' if r.reached_target_zone else 'n'}  "
              f"shoulder@peak={peak_shoulder:6.1f} torso@peak={peak_torso:6.1f}")

    print("\nper-frame availability of the metrics a curl depends on:")
    for name in ("elbow_mean", "shoulder_mean", "torso_lean", "elbow_l", "elbow_r"):
        col = np.array([s.metrics.get(name, np.nan) for s in states], dtype=np.float64)
        pct = 100.0 * np.isfinite(col).mean()
        rng = (f"{np.nanmin(col):6.1f}..{np.nanmax(col):6.1f}"
               if np.isfinite(col).any() else "   n/a")
        print(f"  {name:<16} measurable {pct:5.1f}%   range {rng}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
