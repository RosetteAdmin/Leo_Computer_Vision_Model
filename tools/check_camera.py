"""Pre-flight check: camera, pose detection, and real end-to-end speed.

    python -m tools.check_camera

Headless, so it works over a terminal with no display. Reports which camera
indices are usable, whether MediaPipe actually finds a person, the measured
end-to-end frame rate, and whether your camera angle can support depth
measurement. Run this before the first live session - it turns "it doesn't work"
into a specific answer.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from src.angle_engine import compute_metrics
from src.exercise_config import ExerciseLibrary
from src.pipeline import MonitorPipeline

OK = "  [ok]  "
BAD = "  [!!]  "
INFO = "        "


def probe_indices(max_index: int = 3) -> list[int]:
    import cv2

    working: list[int] = []
    for i in range(max_index + 1):
        cap = (cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform == "win32"
               else cv2.VideoCapture(i))
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    print(f"{OK}index {i}: {w}x{h}")
                    working.append(i)
                else:
                    print(f"{BAD}index {i}: opened but delivered no frame")
        finally:
            cap.release()
    return working


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=None, help="test only this index or video path")
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--configs", default="configs")
    p.add_argument("--model", default="models/exercise_classifier.pkl")
    p.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    args = p.parse_args(argv)

    problems: list[str] = []

    # --- 1. dependencies ---
    print("=" * 66)
    print("1. dependencies")
    print("=" * 66)
    for mod in ("cv2", "mediapipe", "numpy", "sklearn", "scipy", "yaml", "joblib"):
        try:
            m = __import__(mod)
            print(f"{OK}{mod:<12}{getattr(m, '__version__', 'installed')}")
        except ImportError as exc:
            print(f"{BAD}{mod:<12}MISSING - {exc}")
            problems.append(f"install {mod}: pip install -r requirements.txt")
    if problems:
        print("\n".join(problems))
        return 1

    # --- 2. configs and model ---
    print()
    print("=" * 66)
    print("2. configs and classifier")
    print("=" * 66)
    library = ExerciseLibrary.from_dir(args.configs)
    print(f"{OK}{len(library)} exercise configs: {', '.join(library.names)}")

    bundle = None
    model_path = Path(args.model)
    if model_path.exists():
        from src.classifier import ClassifierBundle

        bundle = ClassifierBundle.load(model_path)
        print(f"{OK}classifier: {bundle.model_type}, classes: {', '.join(bundle.labels)}")
        if bundle.metadata.get("synthetic_data"):
            print(f"{INFO}trained on SYNTHETIC data - expect real-world accuracy to differ")
    else:
        print(f"{BAD}no classifier at {model_path}")
        print(f"{INFO}generic mode will still work; train one with: python -m src.train")

    # --- 3. camera ---
    print()
    print("=" * 66)
    print("3. camera")
    print("=" * 66)
    if args.source is None:
        found = probe_indices()
        if not found:
            print(f"{BAD}no usable camera found on indices 0-3")
            print(f"{INFO}check: another app using the camera? Windows camera privacy")
            print(f"{INFO}settings? external webcam plugged in?")
            print(f"{INFO}you can still test everything with:")
            print(f"{INFO}  python -m src.main --replay data/test_faults/squat_good_00.npz")
            return 1
        source: int | str = found[0]
        print(f"{INFO}using index {source}")
    else:
        source = int(args.source) if str(args.source).isdigit() else args.source

    # --- 4. live pose + pipeline ---
    print()
    print("=" * 66)
    print(f"4. live capture ({args.frames} frames) - stand in view of the camera")
    print("=" * 66)
    import cv2

    from src.main import open_capture
    from src.pose_extraction import PoseExtractor

    cap = open_capture(str(source), args.width, args.height)
    extractor = PoseExtractor(model_complexity=args.model_complexity)
    pipe = MonitorPipeline(library, bundle=bundle)

    detected = 0
    total = 0
    latencies: list[float] = []
    frontality: list[float] = []
    labels: dict[str, int] = {}
    torso_px: list[float] = []

    t0 = time.perf_counter()
    try:
        while total < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.perf_counter()
            pf = extractor.process(frame)
            state = pipe.process_landmarks(pf.landmarks, pf.image_size, total, pf.timestamp)
            latencies.append((time.perf_counter() - t) * 1000.0)
            total += 1
            if pf.landmarks is not None:
                detected += 1
                m = compute_metrics(pf.landmarks, pf.image_size)
                if m["view_frontality"] == m["view_frontality"]:
                    frontality.append(m["view_frontality"])
                key = state.exercise or "generic"
                labels[key] = labels.get(key, 0) + 1
                from src import landmarks as LM

                sh = 0.5 * (pf.landmarks[LM.LEFT_SHOULDER, :2] + pf.landmarks[LM.RIGHT_SHOULDER, :2])
                hp = 0.5 * (pf.landmarks[LM.LEFT_HIP, :2] + pf.landmarks[LM.RIGHT_HIP, :2])
                torso_px.append(float(np.linalg.norm((sh - hp) * np.array(pf.image_size))))
    finally:
        cap.release()
        extractor.close()
    wall = time.perf_counter() - t0

    print()
    rate = detected / max(1, total)
    if rate > 0.9:
        print(f"{OK}person detected in {detected}/{total} frames ({rate:.0%})")
    elif rate > 0.3:
        print(f"{BAD}person detected in only {detected}/{total} frames ({rate:.0%})")
        print(f"{INFO}improve lighting, step back so your whole body is in frame")
    else:
        print(f"{BAD}almost no detections ({detected}/{total})")
        print(f"{INFO}was anyone in front of the camera? Full body must be visible.")
        problems.append("pose detection unreliable")

    arr = np.asarray(latencies)
    fps = total / wall if wall > 0 else 0.0
    print(f"{OK if fps >= 15 else BAD}measured {fps:.1f} FPS end to end "
          f"(pose+analysis {arr.mean():.1f} ms mean, {np.percentile(arr, 95):.1f} ms p95)")
    if fps < 15:
        print(f"{INFO}target is >=15 FPS. Try --model-complexity 0, or a smaller")
        print(f"{INFO}capture size: --width 480 --height 360")

    if torso_px:
        tp = float(np.median(torso_px))
        if tp < 45:
            print(f"{BAD}you look small in frame (torso ~{tp:.0f} px) - move closer")
        elif tp > 260:
            print(f"{BAD}you fill the frame (torso ~{tp:.0f} px) - step back so feet are visible")
        else:
            print(f"{OK}framing looks good (torso ~{tp:.0f} px)")

    if frontality:
        f = float(np.median(frontality))
        if 0.45 <= f <= 0.80:
            print(f"{OK}camera angle good for both depth and knee checks "
                  f"(view_frontality {f:.2f})")
        elif f > 0.80:
            print(f"{BAD}you are too square to the camera (view_frontality {f:.2f})")
            print(f"{INFO}turn ~45 degrees away, or depth/ROM checks will be suppressed")
        else:
            print(f"{BAD}you are nearly side-on (view_frontality {f:.2f})")
            print(f"{INFO}fine for depth, but knee-alignment checks will be skipped;")
            print(f"{INFO}turn toward the camera a little for both")

    if labels:
        top = sorted(labels.items(), key=lambda kv: -kv[1])
        summary = ", ".join(f"{k} {v / max(1, detected):.0%}" for k, v in top[:4])
        print(f"{INFO}classifier saw: {summary}")

    print()
    print("=" * 66)
    if problems:
        print("RESULT: issues found -")
        for x in problems:
            print(f"  - {x}")
        return 1
    print("RESULT: ready. Start a live session with:")
    print("  python -m src.main")
    print("  python -m src.main --exercise squat     (skip the classifier)")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
