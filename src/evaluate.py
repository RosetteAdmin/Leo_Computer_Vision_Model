"""Stage 8 - evaluation. Produces ``docs/metrics_report.md``.

    python -m src.evaluate

Everything is measured by replaying recorded sessions through the *same*
:class:`~src.pipeline.MonitorPipeline` that runs live, so the report cannot drift
away from shipped behaviour.

Measured
--------
1. Exercise classification: accuracy / macro-F1 / confusion matrix (from
   training's held-out split) plus the unknown-rejection rate on movements that
   were never trained.
2. Rep counting: mean absolute error against per-clip ground truth, both with
   the exercise forced (isolating the counter) and end-to-end through the
   classifier.
3. Mistake detection: per-code precision and recall.
4. Latency: per-frame cost of the analysis pipeline, plus a separate MediaPipe
   pose-estimation benchmark, and the resulting end-to-end FPS estimate.

Metric definitions for mistake detection
----------------------------------------
For each error code, over the fault test set:

* **positives** - clips whose ``expected_codes`` contain that code,
* **negatives** - clips recorded with good form only.

Other faults' clips are excluded from a code's negatives, because a deliberately
bad rep frequently exhibits more than one genuine fault (a very asymmetric squat
really is shallow on one side), and counting those as false positives would
punish correct detections. This is stated explicitly because it materially
affects the precision figure.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .classifier import ClassifierBundle
from .exercise_config import ExerciseLibrary
from .pipeline import MonitorPipeline
from .session import Session, load_sessions


# --- helpers -----------------------------------------------------------------
@dataclass
class RepResult:
    clip: str
    exercise: str
    fault: str
    expected: int
    counted: int
    label_purity: float = 0.0

    @property
    def error(self) -> int:
        return abs(self.counted - self.expected)


@dataclass
class CodeStats:
    tp: int = 0
    fn: int = 0
    fp: int = 0
    n_pos_clips: int = 0
    n_neg_clips: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not (p == p and r == r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)


def _fmt(v: float, nd: int = 3) -> str:
    return "n/a" if v != v else f"{v:.{nd}f}"


def _new_pipeline(library: ExerciseLibrary, bundle: ClassifierBundle | None,
                  forced: str | None) -> MonitorPipeline:
    return MonitorPipeline(library, bundle=bundle, forced_exercise=forced)


# --- 2. rep counting ----------------------------------------------------------
def eval_reps(
    sessions: Sequence[Session],
    library: ExerciseLibrary,
    bundle: ClassifierBundle | None,
    forced: bool,
) -> list[RepResult]:
    out: list[RepResult] = []
    for s in sessions:
        if s.expected_reps is None or s.label not in library:
            continue
        pipe = _new_pipeline(library, None if forced else bundle, s.label if forced else None)
        states = pipe.replay(s)
        counted = int(pipe.rep_totals().get(s.label, 0))
        labels = Counter(st.exercise for st in states if st.detected)
        purity = labels.get(s.label, 0) / max(1, sum(labels.values()))
        out.append(RepResult(
            clip=s.path.name if s.path else "?",
            exercise=s.label, fault=s.fault,
            expected=s.expected_reps, counted=counted, label_purity=purity,
        ))
    return out


# --- 3. mistake detection -----------------------------------------------------
def eval_mistakes(
    sessions: Sequence[Session],
    library: ExerciseLibrary,
    bundle: ClassifierBundle | None,
    forced: bool = True,
) -> tuple[dict[str, CodeStats], list[dict]]:
    """Per-code precision/recall, plus a per-clip trace for the appendix."""
    stats: dict[str, CodeStats] = defaultdict(CodeStats)
    per_clip: list[dict] = []
    all_codes: set[str] = set()

    detections: list[tuple[Session, set[str]]] = []
    for s in sessions:
        if s.label not in library:
            continue
        pipe = _new_pipeline(library, None if forced else bundle, s.label if forced else None)
        pipe.replay(s)
        raised = {m.code for m in pipe.detector.history}
        detections.append((s, raised))
        all_codes |= set(s.expected_codes)
        per_clip.append({
            "clip": s.path.name if s.path else "?",
            "exercise": s.label,
            "fault": s.fault,
            "expected": list(s.expected_codes),
            "raised": sorted(raised),
            "reps": pipe.rep_totals(),
        })

    for code in sorted(all_codes):
        st = stats[code]
        for s, raised in detections:
            if code in s.expected_codes:
                st.n_pos_clips += 1
                if code in raised:
                    st.tp += 1
                else:
                    st.fn += 1
            elif s.fault == "good":
                st.n_neg_clips += 1
                if code in raised:
                    st.fp += 1
    return dict(stats), per_clip


# --- 1b. unknown rejection ----------------------------------------------------
def eval_unknown(
    sessions: Sequence[Session],
    library: ExerciseLibrary,
    bundle: ClassifierBundle,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for s in sessions:
        pipe = _new_pipeline(library, bundle, None)
        states = pipe.replay(s)
        # ignore the warm-up frames before the first window is full
        useful = [st for st in states if st.detected][bundle.window:]
        if not useful:
            continue
        unknown = sum(1 for st in useful if st.exercise is None)
        row = out.setdefault(s.label, {"frames": 0.0, "unknown": 0.0, "clips": 0.0,
                                       "generic_reps": 0.0})
        row["frames"] += len(useful)
        row["unknown"] += unknown
        row["clips"] += 1
        row["generic_reps"] += pipe.generic_counter.count
    for row in out.values():
        row["unknown_rate"] = row["unknown"] / max(1.0, row["frames"])
    return out


# --- 4. latency ---------------------------------------------------------------
def eval_latency(
    sessions: Sequence[Session],
    library: ExerciseLibrary,
    bundle: ClassifierBundle | None,
    max_frames: int = 4000,
) -> dict[str, float]:
    """Per-frame cost of everything after pose estimation."""
    pipe = _new_pipeline(library, bundle, None)
    per_frame: list[float] = []
    n = 0
    for s in sessions:
        for i in range(s.n_frames):
            t0 = time.perf_counter()
            pipe.process_landmarks(s.landmarks[i], s.image_size, i, float(s.timestamps[i]))
            per_frame.append((time.perf_counter() - t0) * 1000.0)
            n += 1
            if n >= max_frames:
                break
        if n >= max_frames:
            break
    arr = np.asarray(per_frame)
    return {
        "frames": float(n),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }


def benchmark_mediapipe(frames: int = 90, size: tuple[int, int] = (640, 480),
                        model_complexity: int = 1) -> dict[str, float] | None:
    """Time MediaPipe Pose on this CPU. Returns ``None`` if unavailable.

    Runs on synthetic frames with no person in them, which is the *worst* case:
    MediaPipe re-runs its person detector every frame instead of falling back to
    cheap frame-to-frame tracking. Treat the number as a conservative bound.
    """
    try:
        from .pose_extraction import PoseExtractor
    except Exception:  # pragma: no cover
        return None
    try:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
        with PoseExtractor(model_complexity=model_complexity) as ex:
            ex.process(img, 0.0)   # warm-up (graph construction)
            times = []
            for i in range(frames):
                t0 = time.perf_counter()
                ex.process(img, float(i))
                times.append((time.perf_counter() - t0) * 1000.0)
    except Exception as exc:  # pragma: no cover - mediapipe/env problems
        return {"error": str(exc)}  # type: ignore[return-value]
    arr = np.asarray(times)
    return {
        "frames": float(frames),
        "width": float(size[0]),
        "height": float(size[1]),
        "model_complexity": float(model_complexity),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


# --- report -------------------------------------------------------------------
def render_report(
    bundle: ClassifierBundle | None,
    rep_forced: list[RepResult],
    rep_e2e: list[RepResult],
    mistakes: dict[str, CodeStats],
    per_clip: list[dict],
    unknown: dict[str, dict[str, float]],
    latency: dict[str, float],
    mp_bench: dict[str, float] | None,
    data_dirs: dict[str, Path],
    synthetic: bool,
) -> str:
    L: list[str] = []
    add = L.append

    add("# Metrics report")
    add("")
    add(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    add(f"Machine: {platform.processor() or platform.machine()} / "
        f"{platform.system()} {platform.release()} / Python {platform.python_version()}")
    add("")

    if synthetic:
        add("> ## Read this first")
        add(">")
        add("> **Every clip behind these numbers is synthetic**, generated by")
        add("> `src/synthetic.py` from a 2D forward-kinematic stick figure. The")
        add("> figures below therefore measure whether the *logic* works - angle")
        add("> maths, rep state machine, threshold rules, unknown-rejection,")
        add("> latency - and say nothing about real-world accuracy. In")
        add("> particular the synthetic classes are far more separable than real")
        add("> recordings, so the classification accuracy is an upper bound and")
        add("> should be read as \"no bug\", not as \"99% accurate\".")
        add(">")
        add("> They also exclude the two largest real-world error sources:")
        add("> BlazePose landmark error under occlusion/motion blur, and genuine")
        add("> out-of-plane movement.")
        add(">")
        add("> To get real numbers: record clips with")
        add("> `python -m src.main --record squat --exercise squat`, annotate")
        add("> `expected_reps` / `expected_codes`, retrain, and re-run this script.")
        add("")

    # --- 1. classification ---
    add("## 1. Exercise classification")
    add("")
    if bundle is None:
        add("No classifier bundle found - skipped.")
    else:
        md = bundle.metadata
        add(f"* Model: `{bundle.model_type}`, {len(bundle.feature_names)} features, "
            f"window {bundle.window} frames (stride {bundle.stride})")
        add(f"* Held-out split: grouped by `{md.get('group_by')}` "
            f"({md.get('n_train_windows')} train / {md.get('n_test_windows')} test windows)")
        add(f"* **Accuracy {md.get('test_accuracy')}, macro-F1 {md.get('test_macro_f1')}**")
        add(f"* Unknown gate: max-probability < {bundle.min_confidence} "
            f"OR IsolationForest novelty rejection")
        add("")
        labels = list(bundle.labels)
        cm = md.get("confusion_matrix")
        if cm:
            add("Confusion matrix (rows = truth, columns = predicted):")
            add("")
            add("| |" + "|".join(labels) + "|")
            add("|---|" + "|".join("---" for _ in labels) + "|")
            for i, l in enumerate(labels):
                add(f"|**{l}**|" + "|".join(str(v) for v in cm[i]) + "|")
            add("")
        rep = md.get("class_report") or {}
        if rep:
            add("Per-class precision / recall / F1:")
            add("")
            add("|exercise|precision|recall|f1|support|")
            add("|---|---|---|---|---|")
            for l in labels:
                r = rep.get(l, {})
                add(f"|{l}|{r.get('precision', float('nan')):.3f}|"
                    f"{r.get('recall', float('nan')):.3f}|{r.get('f1-score', float('nan')):.3f}|"
                    f"{int(r.get('support', 0))}|")
            add("")

    # --- 1b. unknown rejection ---
    add("### 1b. Fallback to generic mode on untrained movements")
    add("")
    if not unknown:
        add("No out-of-set clips found - skipped.")
    else:
        add("Fraction of frames labelled `unknown` (higher is better: these")
        add("movements have no config and must **not** be forced into a known class).")
        add("")
        add("|movement|clips|frames|unknown rate|generic reps counted|")
        add("|---|---|---|---|---|")
        for name, row in sorted(unknown.items()):
            add(f"|{name}|{int(row['clips'])}|{int(row['frames'])}|"
                f"{row['unknown_rate']:.3f}|{int(row['generic_reps'])}|")
        add("")
        overall = sum(r["unknown"] for r in unknown.values()) / max(
            1.0, sum(r["frames"] for r in unknown.values()))
        add(f"**Overall unknown rate on untrained movements: {overall:.3f}**")
        add("")

    # --- 2. rep counting ---
    add("## 2. Repetition counting")
    add("")
    for title, results in (("Exercise forced (counter in isolation)", rep_forced),
                           ("End-to-end (classifier chooses the exercise)", rep_e2e)):
        add(f"### {title}")
        add("")
        if not results:
            add("No clips with `expected_reps` - skipped.")
            add("")
            continue
        errs = [r.error for r in results]
        mae = statistics.fmean(errs)
        within1 = sum(1 for e in errs if e <= 1) / len(errs)
        exact = sum(1 for e in errs if e == 0) / len(errs)
        add(f"* clips: {len(results)}")
        add(f"* **MAE {mae:.2f} reps**, exact {exact:.1%}, within +/-1 {within1:.1%}")
        add("")
        by_ex: dict[str, list[RepResult]] = defaultdict(list)
        for r in results:
            by_ex[r.exercise].append(r)
        add("|exercise|clips|expected|counted|MAE|within +/-1|")
        add("|---|---|---|---|---|---|")
        for ex, rows in sorted(by_ex.items()):
            e = [r.error for r in rows]
            add(f"|{ex}|{len(rows)}|{sum(r.expected for r in rows)}|"
                f"{sum(r.counted for r in rows)}|{statistics.fmean(e):.2f}|"
                f"{sum(1 for x in e if x <= 1) / len(e):.0%}|")
        add("")
        worst = sorted(results, key=lambda r: -r.error)[:6]
        if worst and worst[0].error > 0:
            add("Worst clips:")
            add("")
            add("|clip|fault|expected|counted|")
            add("|---|---|---|---|")
            for r in worst:
                if r.error == 0:
                    continue
                add(f"|{r.clip}|{r.fault}|{r.expected}|{r.counted}|")
            add("")

    # --- 3. mistake detection ---
    add("## 3. Mistake detection")
    add("")
    add("Clip-level detection. Positives = clips whose fault is expected to")
    add("trigger the code; negatives = good-form clips only (see the module")
    add("docstring for why other faults are excluded from the negatives).")
    add("")
    if not mistakes:
        add("No labelled fault clips found - skipped.")
    else:
        add("|error code|pos clips|neg clips|TP|FN|FP|precision|recall|F1|")
        add("|---|---|---|---|---|---|---|---|---|")
        for code, st in sorted(mistakes.items()):
            add(f"|`{code}`|{st.n_pos_clips}|{st.n_neg_clips}|{st.tp}|{st.fn}|{st.fp}|"
                f"{_fmt(st.precision)}|{_fmt(st.recall)}|{_fmt(st.f1)}|")
        add("")
        ps = [s.precision for s in mistakes.values() if s.precision == s.precision]
        rs = [s.recall for s in mistakes.values() if s.recall == s.recall]
        if ps and rs:
            add(f"**Macro precision {statistics.fmean(ps):.3f}, "
                f"macro recall {statistics.fmean(rs):.3f}** over "
                f"{len(mistakes)} codes.")
            add("")

    # --- 4. latency ---
    add("## 4. Latency and throughput")
    add("")
    add("### Analysis pipeline (angle engine + features + classifier + counter + checks)")
    add("")
    add(f"* frames timed: {int(latency['frames'])}")
    add(f"* mean **{latency['mean_ms']:.2f} ms**, median {latency['median_ms']:.2f} ms, "
        f"p95 {latency['p95_ms']:.2f} ms, max {latency['max_ms']:.2f} ms")
    add("")
    add("### MediaPipe Pose (the dominant cost)")
    add("")
    if not mp_bench:
        add("MediaPipe unavailable - not measured.")
    elif "error" in mp_bench:
        add(f"Benchmark failed: `{mp_bench['error']}`")
    else:
        add(f"* {int(mp_bench['width'])}x{int(mp_bench['height'])}, "
            f"model_complexity={int(mp_bench['model_complexity'])}, "
            f"{int(mp_bench['frames'])} frames")
        add(f"* mean **{mp_bench['mean_ms']:.1f} ms**, median {mp_bench['median_ms']:.1f} ms, "
            f"p95 {mp_bench['p95_ms']:.1f} ms")
        add("* Worst case: no person in frame, so the person detector runs every")
        add("  frame instead of tracking. A real subject is typically faster.")
        add("")
        total = mp_bench["mean_ms"] + latency["mean_ms"]
        add(f"### Estimated end-to-end")
        add("")
        add(f"* {total:.1f} ms/frame -> **~{1000.0 / total:.1f} FPS** "
            f"(pose {mp_bench['mean_ms']:.1f} ms + analysis {latency['mean_ms']:.2f} ms)")
        add("* Excludes webcam capture and window compositing; measure the real")
        add("  figure with `python -m src.main --benchmark`.")
        add(f"* Target from the brief was >=15 FPS: "
            f"{'MET' if 1000.0 / total >= 15 else 'NOT MET'} on this machine.")
    add("")

    # --- appendix ---
    add("## Appendix: per-clip mistake-detection trace")
    add("")
    add("|clip|fault|expected codes|codes raised|reps|")
    add("|---|---|---|---|---|")
    for row in per_clip:
        exp = ", ".join(f"`{c}`" for c in row["expected"]) or "-"
        got = ", ".join(f"`{c}`" for c in row["raised"]) or "-"
        reps = ", ".join(f"{k}={v}" for k, v in row["reps"].items()) or "0"
        add(f"|{row['clip']}|{row['fault']}|{exp}|{got}|{reps}|")
    add("")
    add("## Data sources")
    add("")
    for k, v in data_dirs.items():
        add(f"* {k}: `{v}`")
    add("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate the system and write a metrics report.")
    p.add_argument("--configs", default="configs")
    p.add_argument("--model", default="models/exercise_classifier.pkl")
    p.add_argument("--faults", default="data/test_faults",
                   help="clips with fault/expected_codes labels")
    p.add_argument("--unseen", default="data/test_unseen",
                   help="movements the classifier was never trained on")
    p.add_argument("--out", default="docs/metrics_report.md")
    p.add_argument("--json-out", default="docs/metrics_report.json")
    p.add_argument("--skip-mediapipe", action="store_true")
    args = p.parse_args(argv)

    library = ExerciseLibrary.from_dir(args.configs)
    bundle = None
    model_path = Path(args.model)
    if model_path.exists():
        bundle = ClassifierBundle.load(model_path)
        print(f"loaded classifier: {model_path} ({bundle.model_type}, {len(bundle.labels)} classes)")
    else:
        print(f"no classifier at {model_path} - classification sections will be skipped")

    fault_sessions = load_sessions(args.faults) if Path(args.faults).is_dir() else []
    unseen_sessions = load_sessions(args.unseen) if Path(args.unseen).is_dir() else []
    print(f"{len(fault_sessions)} fault-set clips, {len(unseen_sessions)} out-of-set clips")

    print("evaluating rep counting (forced exercise)...")
    rep_forced = eval_reps(fault_sessions, library, bundle, forced=True)
    print("evaluating rep counting (end to end)...")
    rep_e2e = eval_reps(fault_sessions, library, bundle, forced=False) if bundle else []
    print("evaluating mistake detection...")
    mistakes, per_clip = eval_mistakes(fault_sessions, library, bundle, forced=True)
    print("evaluating unknown fallback...")
    unknown = eval_unknown(unseen_sessions, library, bundle) if bundle else {}
    print("timing analysis pipeline...")
    latency = eval_latency(fault_sessions or unseen_sessions, library, bundle)
    mp_bench = None
    if not args.skip_mediapipe:
        print("benchmarking MediaPipe Pose...")
        mp_bench = benchmark_mediapipe()

    synthetic = bool(fault_sessions) and all(s.meta.get("synthetic") for s in fault_sessions)

    report = render_report(
        bundle, rep_forced, rep_e2e, mistakes, per_clip, unknown, latency, mp_bench,
        {"training data": Path(bundle.metadata.get("data_dir", "-")) if bundle else Path("-"),
         "fault test set": Path(args.faults).resolve(),
         "out-of-set clips": Path(args.unseen).resolve()},
        synthetic,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nwrote {out}")

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "synthetic_data": synthetic,
        "classifier": bundle.metadata if bundle else None,
        "rep_counting": {
            "forced": [r.__dict__ for r in rep_forced],
            "end_to_end": [r.__dict__ for r in rep_e2e],
        },
        "mistake_detection": {k: v.__dict__ | {"precision": v.precision, "recall": v.recall}
                              for k, v in mistakes.items()},
        "unknown_fallback": unknown,
        "latency": latency,
        "mediapipe": mp_bench,
    }
    jout = Path(args.json_out)
    jout.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"wrote {jout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
