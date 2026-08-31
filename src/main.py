"""Stage 7 - real-time application entry point.

    python -m src.main                          # webcam, classify automatically
    python -m src.main --exercise squat         # skip the classifier, drill one movement
    python -m src.main --record squat           # capture a labelled training clip
    python -m src.main --source clip.mp4        # analyse a video file
    python -m src.main --replay data/test_faults/squat_shallow_00.npz
    python -m src.main --benchmark 200          # headless FPS measurement

Keys while running:
    q / ESC  quit          r  reset counters      a  toggle the angle table
    m        mirror        p  pause               s  save the recording now

Everything runs locally. There is no network call anywhere in this program.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .classifier import ClassifierBundle
from .exercise_config import ExerciseLibrary
from .feedback import coach, short_for
from .overlay import blank_canvas, render
from .pipeline import MonitorPipeline
from .session import Session, SessionRecorder

KEYMAP = "q quit | r reset | a angles | m mirror | f fullscreen | p pause | s save"
WINDOW = "Exercise Monitor"


def _open_window(name: str, fullscreen: bool) -> None:
    """Create a resizable window.

    Capture stays at the camera's native size for speed; only the *displayed*
    copy is scaled up. Landmarks are normalised, so the overlay lands correctly
    at any display size, and the HUD is drawn after the upscale at a size derived
    from the canvas width (see :func:`src.overlay.ui_scale`) so the text grows
    with the window instead of staying small on a bigger picture.
    """
    import cv2

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def _for_display(frame: np.ndarray, target_width: int) -> np.ndarray:
    """Upscale a copy of the frame for display, preserving aspect ratio."""
    import cv2

    h, w = frame.shape[:2]
    if target_width <= 0 or w <= 0 or target_width == w:
        return frame
    scale = target_width / float(w)
    return cv2.resize(frame, (target_width, max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_LINEAR)


# --- sources ------------------------------------------------------------------
def open_capture(source: str, width: int, height: int):
    import cv2

    src: int | str = int(source) if str(source).isdigit() else source
    # CAP_DSHOW avoids the multi-second MSMF start-up delay on Windows.
    if isinstance(src, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(
            f"could not open video source '{source}'. "
            "Try a different index (--source 1), or check camera permissions."
        )
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def load_bundle(path: str | Path, quiet: bool = False) -> ClassifierBundle | None:
    p = Path(path)
    if not p.exists():
        if not quiet:
            print(f"[!] no classifier at {p} - running in generic mode.")
            print("    Train one with:  python -m src.train")
        return None
    bundle = ClassifierBundle.load(p)
    if not quiet:
        print(f"classifier: {p.name} ({bundle.model_type}, {len(bundle.labels)} classes: "
              f"{', '.join(bundle.labels)})")
    return bundle


def print_summary(pipe: MonitorPipeline, elapsed: float) -> None:
    s = pipe.summary()
    print("\n--- session summary -------------------------------------------")
    print(f"frames        : {s['frames']} ({s['frames_with_pose']} with a pose) "
          f"in {elapsed:.1f}s")
    if elapsed > 0:
        print(f"average FPS   : {s['frames'] / elapsed:.1f}")
    reps = s["reps"] or {}
    print(f"reps          : {reps if reps else 'none'}")
    mistakes = s["mistakes"] or {}
    if mistakes:
        print("form issues   :")
        for code, n in mistakes.items():
            print(f"    {n:>3} x {code:<26} {short_for(code)}")
    else:
        print("form issues   : none detected")
    print("---------------------------------------------------------------")


# --- live / video loop --------------------------------------------------------
def run_camera(args: argparse.Namespace) -> int:
    import cv2

    from .angle_engine import compute_metrics
    from .features import frame_features
    from .pose_extraction import PoseExtractor

    library = ExerciseLibrary.from_dir(args.configs)
    bundle = None if args.no_classifier else load_bundle(args.model, args.benchmark > 0)
    pipe = MonitorPipeline(
        library, bundle=bundle, forced_exercise=args.exercise,
        smoothing_seconds=args.smoothing_seconds, min_confidence=args.min_confidence,
        classify_every=args.classify_every,
    )
    recorder = (
        SessionRecorder(args.record, fault=args.fault, subject=args.subject,
                        expected_reps=args.expected_reps, source=str(args.source))
        if args.record else None
    )

    cap = open_capture(args.source, args.width, args.height)
    extractor = PoseExtractor(model_complexity=args.model_complexity)
    writer = None
    headless = args.headless or args.benchmark > 0

    mirror = args.mirror
    show_angles = True
    paused = False
    fullscreen = args.fullscreen
    if not headless:
        _open_window(WINDOW, fullscreen)
    frames_since_rep = -1
    t_start = time.perf_counter()
    frame_index = -1
    latencies: list[float] = []

    if not headless:
        print("\nrunning - press q to quit\n")

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                if mirror:
                    frame = cv2.flip(frame, 1)

                t0 = time.perf_counter()
                pf = extractor.process(frame)
                state = pipe.process_landmarks(
                    pf.landmarks, pf.image_size, frame_index, pf.timestamp
                )
                latencies.append((time.perf_counter() - t0) * 1000.0)

                if recorder is not None and pf.landmarks is not None:
                    m = compute_metrics(pf.landmarks, pf.image_size)
                    recorder.add(pf.landmarks, m,
                                 frame_features(pf.landmarks, pf.image_size, m), pf.timestamp)

                frames_since_rep = 0 if state.new_rep is not None else (
                    frames_since_rep + 1 if frames_since_rep >= 0 else -1
                )
                for msg in coach(state.new_mistakes):
                    print(f"  [{state.display_name}] {msg}")
                if state.new_rep is not None:
                    print(f"  rep {state.rep_count} - {state.display_name} "
                          f"({state.new_rep.duration:.2f}s, "
                          f"ROM {state.new_rep.rom:.0f})")

                if args.benchmark and frame_index + 1 >= args.benchmark:
                    break

            if not headless:
                cfg = library[state.exercise] if state.exercise else None
                extra = "REC" if recorder is not None else ""
                canvas = _for_display(frame, args.display_width)
                render(canvas, state, cfg, show_angles, frames_since_rep, extra, KEYMAP)
                if args.save_video:
                    if writer is None:
                        h, w = canvas.shape[:2]
                        writer = cv2.VideoWriter(
                            args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                            args.save_fps, (w, h))
                    writer.write(canvas)
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        WINDOW, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    pipe.reset()
                    print("counters reset")
                if key == ord("a"):
                    show_angles = not show_angles
                if key == ord("m"):
                    mirror = not mirror
                if key == ord("p"):
                    paused = not paused
                if key == ord("s") and recorder is not None and len(recorder):
                    print(f"saved {recorder.save(args.out_dir)}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        cap.release()
        extractor.close()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print_summary(pipe, elapsed)
    if latencies:
        arr = np.asarray(latencies)
        print(f"per-frame pose+analysis: mean {arr.mean():.1f} ms, "
              f"median {np.median(arr):.1f} ms, p95 {np.percentile(arr, 95):.1f} ms "
              f"-> {1000.0 / arr.mean():.1f} FPS capability")
    if recorder is not None and len(recorder):
        print(f"recording saved: {recorder.save(args.out_dir)}")
    return 0


# --- replay of a recorded session --------------------------------------------
def run_replay(args: argparse.Namespace) -> int:
    """Render a stored session. Useful for demos and debugging without a camera."""
    import cv2

    library = ExerciseLibrary.from_dir(args.configs)
    bundle = None if args.no_classifier else load_bundle(args.model)
    session = Session.load(args.replay)
    print(f"replaying {Path(args.replay).name}: label={session.label} "
          f"fault={session.fault} frames={session.n_frames} "
          f"expected_reps={session.expected_reps}")

    pipe = MonitorPipeline(library, bundle=bundle, forced_exercise=args.exercise,
                           smoothing_seconds=args.smoothing_seconds,
                           min_confidence=args.min_confidence,
                           classify_every=args.classify_every)
    writer = None
    frames_since_rep = -1
    delay = max(1, int(1000.0 / max(1.0, session.fps)))
    t_start = time.perf_counter()

    for i in range(session.n_frames):
        state = pipe.process_landmarks(
            session.landmarks[i], session.image_size, i, float(session.timestamps[i])
        )
        frames_since_rep = 0 if state.new_rep is not None else (
            frames_since_rep + 1 if frames_since_rep >= 0 else -1
        )
        for msg in coach(state.new_mistakes):
            print(f"  frame {i:>4} [{state.display_name}] {msg}")

        if args.headless:
            continue
        if i == 0:
            _open_window(WINDOW + " - replay", args.fullscreen)
        side = max(360, args.display_width * 3 // 4)
        canvas = blank_canvas(side, side)
        cfg = library[state.exercise] if state.exercise else None
        render(canvas, state, cfg, True, frames_since_rep, "REPLAY", "q quit | p pause")
        if args.save_video:
            if writer is None:
                writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                         session.fps, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
        cv2.imshow(WINDOW + " - replay", canvas)
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            while (cv2.waitKey(30) & 0xFF) != ord("p"):
                pass

    if writer is not None:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()
    print_summary(pipe, time.perf_counter() - t_start)
    return 0


# --- cli ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="webcam index or video file path")
    p.add_argument("--replay", help="replay a recorded .npz session instead of a camera")
    p.add_argument("--configs", default="configs")
    p.add_argument("--model", default="models/exercise_classifier.pkl")
    p.add_argument("--exercise", help="force this exercise and skip classification")
    p.add_argument("--no-classifier", action="store_true",
                   help="force generic mode for everything")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2],
                   help="MediaPipe model: 0 lite (fastest), 1 full, 2 heavy")
    p.add_argument("--smoothing-seconds", type=float, default=0.15,
                   help="metric moving-average window in SECONDS. Time-based on "
                        "purpose: a frame count over-smooths a slow camera")
    p.add_argument("--min-confidence", type=float, default=None,
                   help="override the classifier's unknown threshold")
    p.add_argument("--classify-every", type=int, default=3,
                   help="run the classifier every Nth frame")
    p.add_argument("--mirror", action="store_true", default=True,
                   help="mirror the webcam image (default on, feels natural)")
    p.add_argument("--no-mirror", dest="mirror", action="store_false")
    p.add_argument("--display-width", type=int, default=1280,
                   help="width of the DISPLAY window in pixels. Capture stays at "
                        "--width for speed; only the shown copy is scaled")
    p.add_argument("--fullscreen", action="store_true", help="start full screen (toggle with f)")
    p.add_argument("--headless", action="store_true", help="no window")
    p.add_argument("--benchmark", type=int, default=0,
                   help="process N frames headless and report FPS")
    p.add_argument("--save-video", help="write the annotated output to this mp4")
    p.add_argument("--save-fps", type=float, default=20.0)
    # recording
    p.add_argument("--record", metavar="LABEL",
                   help="record a labelled session (the exercise name)")
    p.add_argument("--fault", default="good",
                   help="fault tag for the recording, e.g. good / shallow / valgus")
    p.add_argument("--subject", default="unknown", help="who is in the clip")
    p.add_argument("--expected-reps", type=int, help="ground-truth rep count for evaluation")
    p.add_argument("--out-dir", default="data/raw_sessions")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.record and args.exercise is None:
        # Recording clean training clips is much easier when the classifier is
        # not allowed to guess: the label you type is the label you get.
        args.exercise = args.record if args.record in ExerciseLibrary.from_dir(
            args.configs) else None
    if args.replay:
        return run_replay(args)
    return run_camera(args)


if __name__ == "__main__":
    sys.exit(main())
