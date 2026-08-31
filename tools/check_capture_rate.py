"""Isolate the camera from the analysis: how fast can we even READ frames?

    python -m tools.check_capture_rate

Answers a question the end-to-end FPS number cannot: is the pipeline slow, or is
the webcam slow? Webcams routinely drop to 15 or even 7.5 FPS in dim light,
because the sensor lengthens its exposure. No amount of code optimisation fixes
that - turning a light on does.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def main(argv: list[str] | None = None) -> int:
    import cv2

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=int, default=0)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args(argv)

    cap = (cv2.VideoCapture(args.source, cv2.CAP_DSHOW) if sys.platform == "win32"
           else cv2.VideoCapture(args.source))
    if not cap.isOpened():
        print(f"could not open camera index {args.source}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    reported = cap.get(cv2.CAP_PROP_FPS)
    for _ in range(10):        # let auto-exposure settle
        cap.read()

    gaps: list[float] = []
    brightness: list[float] = []
    last = time.perf_counter()
    for _ in range(args.frames):
        ok, frame = cap.read()
        now = time.perf_counter()
        if not ok:
            break
        gaps.append((now - last) * 1000.0)
        last = now
        brightness.append(float(frame.mean()))
    h, w = frame.shape[:2]
    cap.release()

    g = np.asarray(gaps)
    b = float(np.mean(brightness))
    print(f"resolution         : {w}x{h}")
    print(f"driver-reported FPS: {reported:.1f}")
    print(f"measured capture   : {1000.0 / g.mean():.1f} FPS "
          f"({g.mean():.1f} ms/frame, p95 {np.percentile(g, 95):.1f} ms)")
    print(f"mean frame brightness: {b:.0f} / 255")
    print()
    if 1000.0 / g.mean() < 20:
        print("The CAMERA is the bottleneck, not the analysis pipeline.")
        if b < 70:
            print(f"  The image is dark ({b:.0f}/255). Webcams lengthen exposure in low")
            print("  light, which directly halves or quarters the frame rate AND blurs")
            print("  motion, which degrades landmark accuracy. Add light - it is the")
            print("  single highest-value change you can make.")
        else:
            print("  Brightness looks adequate, so this is a driver/USB limit.")
            print("  Try --width 480 --height 360, or a different USB port.")
    else:
        print("Capture rate is healthy; any shortfall is in processing or display.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
