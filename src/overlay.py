"""OpenCV rendering: skeleton, HUD, live angle table, warnings.

Kept entirely separate from analysis - :class:`~src.pipeline.FrameState` in,
pixels out - so the pipeline can run headless (evaluation, benchmarking, a
future embedded build with no display) without dragging in any drawing code.

Colour convention (BGR, because OpenCV):
    green  = good / on target
    amber  = warning
    red    = error, and the joints responsible
    grey   = generic / unknown mode

Legibility rules, all three learned from squinting at the real thing:

1. **Everything scales with the canvas.** Sizes are authored for
   :data:`UI_BASELINE_WIDTH` and multiplied by the actual width. Fixed pixel
   sizes drawn onto an upscaled frame produce the same small text on a bigger
   picture, which is exactly the complaint that prompted this.
2. **Every glyph gets a dark outline.** Camera feeds are arbitrary colours; a
   thin bright glyph over a bright wall is unreadable no matter the palette.
3. **Panels are opaque enough to actually mask the video.** At alpha 0.55 the
   background bleeds through and text on top of it competes with whatever the
   camera happens to see.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

from . import landmarks as LM
from .exercise_config import ExerciseConfig
from .feedback import describe, message_for, short_for
from .pipeline import FrameState

# --- palette -----------------------------------------------------------------
WHITE = (245, 245, 245)
GREY = (185, 185, 185)          # lifted from 150: too dim once outlined
DARK = (26, 26, 28)
OUTLINE = (0, 0, 0)
BORDER = (90, 90, 96)
GREEN = (90, 230, 130)
AMBER = (70, 200, 255)
RED = (80, 80, 250)
BLUE = (240, 180, 90)
CYAN = (235, 235, 110)

SEVERITY_COLOUR = {"info": BLUE, "warning": AMBER, "error": RED}

FONT = cv2.FONT_HERSHEY_SIMPLEX

#: The canvas width every size in this module is authored against.
UI_BASELINE_WIDTH = 640

#: Height reserved along the bottom edge for the key map, so the blocks that
#: stack upwards from there do not overlap it.
KEYMAP_RESERVE = 22

#: Shown in generic mode, when no exercise config nominates its own metrics.
DEFAULT_TRACKED: tuple[str, ...] = (
    "knee_mean", "hip_mean", "elbow_mean", "shoulder_mean", "torso_lean", "neck",
)


def ui_scale(frame: np.ndarray) -> float:
    """Multiplier taking authored sizes to this canvas.

    Never below 1.0: shrinking the HUD on a small capture would make it worse,
    and the layout has no room to give back.
    """
    return max(1.0, frame.shape[1] / float(UI_BASELINE_WIDTH))


def _to_px(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    out = np.empty((pts.shape[0], 2), dtype=np.int32)
    out[:, 0] = np.clip(pts[:, 0] * w, -1e5, 1e5)
    out[:, 1] = np.clip(pts[:, 1] * h, -1e5, 1e5)
    return out


def _panel(frame: np.ndarray, x: int, y: int, w: int, h: int,
           alpha: float = 0.78, border: bool = True) -> None:
    """Translucent dark rectangle. Keeps text readable over any background."""
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    overlay = np.full_like(roi, DARK)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    if border:
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), BORDER, 1, cv2.LINE_AA)


def _text(frame: np.ndarray, s: str, x: int, y: int, scale: float = 0.5,
          colour: tuple[int, int, int] = WHITE, thick: int = 1,
          outline: bool = True) -> None:
    """Draw text with a dark halo so it survives any background behind it."""
    if outline:
        cv2.putText(frame, s, (x, y), FONT, scale, OUTLINE,
                    thick + 2, cv2.LINE_AA)
    cv2.putText(frame, s, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def _width(s: str, scale: float, thick: int = 1) -> int:
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


# --- skeleton -----------------------------------------------------------------
def draw_skeleton(
    frame: np.ndarray,
    pts: np.ndarray,
    highlight: Iterable[str] = (),
    min_visibility: float = 0.35,
    colour: tuple[int, int, int] = GREEN,
) -> None:
    """Draw all 33 landmarks and the bone topology.

    Landmarks named in ``highlight`` (the joints a mistake was attributed to)
    are drawn larger and in red with a ring, so the user can see *where* the
    problem is instead of only reading about it.
    """
    h, w = frame.shape[:2]
    s = ui_scale(frame)
    px = _to_px(pts, w, h)
    vis = pts[:, 3]
    hi = {LM.NAME_TO_INDEX[n] for n in highlight if n in LM.NAME_TO_INDEX}

    bone = max(2, int(round(2 * s)))
    for a, b in LM.SKELETON_EDGES:
        if vis[a] < min_visibility or vis[b] < min_visibility:
            continue
        c = RED if (a in hi and b in hi) else colour
        cv2.line(frame, tuple(px[a]), tuple(px[b]), c, bone, cv2.LINE_AA)

    for i in range(LM.NUM_LANDMARKS):
        if vis[i] < min_visibility:
            continue
        if i in hi:
            cv2.circle(frame, tuple(px[i]), int(round(9 * s)), RED, bone, cv2.LINE_AA)
            cv2.circle(frame, tuple(px[i]), int(round(4 * s)), RED, -1, cv2.LINE_AA)
        elif i in LM.CORE_LANDMARKS:
            cv2.circle(frame, tuple(px[i]), int(round(3 * s)), WHITE, -1, cv2.LINE_AA)


# --- HUD ----------------------------------------------------------------------
def draw_status(frame: np.ndarray, state: FrameState, extra: str = "") -> None:
    """Top-left block: exercise, rep count, phase, FPS."""
    h, w = frame.shape[:2]
    s = ui_scale(frame)
    def p(v: float) -> int:
        return int(round(v * s))

    _panel(frame, p(8), p(8), p(346), p(112))

    if state.generic:
        title, colour = "GENERIC MOVEMENT", GREY
    elif state.label_source == "ambiguous":
        title, colour = state.display_name.upper() + " ?", AMBER
    else:
        title, colour = state.display_name.upper(), GREEN
    _text(frame, title, p(18), p(34), 0.70 * s, colour, max(2, p(2)))

    # Always say HOW the label was reached. A rule-based guess and a trained
    # recognition look identical on screen otherwise, and they are not the same.
    if state.label_source == "forced":
        sub = "forced (--exercise)"
    elif state.label_source == "model":
        sub = f"recognised by model, {state.confidence:.0%}"
    elif state.label_source == "matched":
        sub = f"matched to config, {state.confidence:.0%}"
    elif state.label_source == "ambiguous":
        rivals = ", ".join(state.match.ambiguous_with) if state.match else ""
        sub = f"could also be {rivals}" if rivals else "ambiguous"
    elif state.confidence:
        sub = f"unrecognised ({state.confidence:.0%} best guess)"
    else:
        sub = "unrecognised - tracking joints only"
    _text(frame, sub, p(18), p(54), 0.46 * s,
          AMBER if state.label_source == "ambiguous" else GREY)

    _text(frame, f"REPS {state.rep_count}", p(18), p(90), 0.88 * s, WHITE, max(2, p(2)))
    _text(frame, f"phase: {state.phase}", p(168), p(80), 0.46 * s, CYAN)
    signal = state.primary_metric or "-"
    val = state.metrics.get(signal, float("nan")) if state.primary_metric else float("nan")
    val_s = "-" if val != val else f"{val:.0f}"
    _text(frame, f"signal: {signal} = {val_s}", p(168), p(98), 0.44 * s, GREY)

    fps_s = f"{state.fps:4.1f} FPS"
    _text(frame, fps_s, w - p(12) - _width(fps_s, 0.54 * s), p(30), 0.54 * s, GREY)
    if extra:
        _text(frame, extra, w - p(12) - _width(extra, 0.5 * s), p(52), 0.5 * s, AMBER)


def draw_angles(
    frame: np.ndarray,
    state: FrameState,
    config: ExerciseConfig | None,
    show: bool = True,
) -> None:
    """Right-hand live angle table - the "monitor every joint" view."""
    if not show:
        return
    names: Sequence[str] = (
        config.tracked_metrics if (config and config.tracked_metrics) else DEFAULT_TRACKED
    )
    h, w = frame.shape[:2]
    s = ui_scale(frame)
    def p(v: float) -> int:
        return int(round(v * s))

    label_scale = 0.52 * s
    value_scale = 0.56 * s
    row = p(24)
    # Measure rather than assume: metric names differ per exercise and a fixed
    # panel width either clips "shoulder_mean" or wastes half the frame.
    name_w = max([_width(n[:20], label_scale) for n in names] or [0])
    pw = p(22) + name_w + p(16) + _width("-000.0", value_scale) + p(14)
    ph = p(34) + row * len(names) + p(10)
    x, y = w - pw - p(10), p(60)
    _panel(frame, x, y, pw, ph)
    _text(frame, "JOINT ANGLES", x + p(14), y + p(24), 0.52 * s, CYAN, max(1, p(1)))
    for i, name in enumerate(names):
        v = state.metrics.get(name, float("nan"))
        yy = y + p(34) + row * (i + 1) - p(6)
        measured = v == v
        s_val = "--" if not measured else (f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}")
        _text(frame, name[:20], x + p(14), yy, label_scale, WHITE)
        # Unmeasurable is stated in amber, not the same green as a real reading:
        # "--" in the colour of a good value reads as a number that failed to load.
        _text(frame, s_val, x + pw - p(14) - _width(s_val, value_scale), yy,
              value_scale, GREEN if measured else AMBER, max(1, p(1)))


def draw_warnings(frame: np.ndarray, state: FrameState, limit: int = 4) -> None:
    """Bottom block: active mistakes, worst first, with the measured value."""
    active = state.active_mistakes[:limit]
    if not active:
        return
    h, w = frame.shape[:2]
    s = ui_scale(frame)
    def p(v: float) -> int:
        return int(round(v * s))

    row = p(30)
    ph = p(18) + row * len(active)
    y = h - ph - p(10) - p(KEYMAP_RESERVE)
    _panel(frame, p(8), y, min(w - p(16), p(600)), ph, alpha=0.82)
    for i, m in enumerate(active):
        c = SEVERITY_COLOUR.get(m.severity, AMBER)
        yy = y + p(28) + row * i
        cv2.rectangle(frame, (p(15), yy - p(13)), (p(22), yy + p(5)), c, -1)
        _text(frame, message_for(m.code), p(30), yy, 0.58 * s, c, max(1, p(1)))
        _text(frame, describe(m, include_numbers=True),
              p(30) + p(360), yy, 0.44 * s, GREY)


#: Body regions the user can actually act on, and the landmarks that define them.
_REGIONS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("head", (LM.NOSE,)),
    ("shoulders", (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)),
    ("arms", (LM.LEFT_ELBOW, LM.RIGHT_ELBOW, LM.LEFT_WRIST, LM.RIGHT_WRIST)),
    ("hips", (LM.LEFT_HIP, LM.RIGHT_HIP)),
    ("knees", (LM.LEFT_KNEE, LM.RIGHT_KNEE)),
    ("feet", (LM.LEFT_ANKLE, LM.RIGHT_ANKLE)),
)


def missing_regions(pts: np.ndarray, threshold: float = 0.55) -> list[str]:
    """Body regions that are not confidently visible.

    Surfaced on screen because it is the difference between "this app is broken"
    and "your legs are out of frame". Every metric that depends on a missing
    region is NaN by design, so the user needs to know which ones those are.
    """
    out: list[str] = []
    for name, idx in _REGIONS:
        if not any(float(pts[i, 3]) >= threshold for i in idx):
            out.append(name)
    return out


def draw_framing(frame: np.ndarray, state: FrameState) -> None:
    """Bottom-left note about what the camera cannot see."""
    if not state.detected or state.landmarks is None:
        return
    missing = missing_regions(state.landmarks)
    if not missing:
        return
    h, w = frame.shape[:2]
    s = ui_scale(frame)
    def p(v: float) -> int:
        return int(round(v * s))

    msg = f"not in frame: {', '.join(missing)}"
    hint = "step back / tilt the camera down so more of you is visible"
    msg_scale, hint_scale = 0.56 * s, 0.48 * s
    width = max(_width(msg, msg_scale), _width(hint, hint_scale)) + p(30)
    ph = p(54)
    # Stacks above the key map and above the warning block, which itself grows
    # with the number of active mistakes.
    warn_h = (p(18) + p(30) * min(4, len(state.active_mistakes)) + p(10)
              if state.active_mistakes else 0)
    y = h - ph - p(10) - p(KEYMAP_RESERVE) - warn_h
    _panel(frame, p(8), y, min(w - p(16), width), ph, alpha=0.82)
    _text(frame, msg, p(18), y + p(22), msg_scale, AMBER, max(1, p(1)))
    _text(frame, hint, p(18), y + p(44), hint_scale, WHITE)


def draw_rep_flash(frame: np.ndarray, frames_since_rep: int, hold: int = 8) -> None:
    """Brief border flash on a completed rep - visible without reading text."""
    if frames_since_rep < 0 or frames_since_rep > hold:
        return
    h, w = frame.shape[:2]
    s = ui_scale(frame)
    alpha = 1.0 - frames_since_rep / float(hold)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), GREEN,
                  max(2, int(round(10 * alpha * s))))


def draw_hint(frame: np.ndarray, text: str) -> None:
    h, w = frame.shape[:2]
    s = ui_scale(frame)
    scale = 0.64 * s
    size = cv2.getTextSize(text, FONT, scale, 2)[0]
    x = (w - size[0]) // 2
    _panel(frame, x - int(16 * s), h // 2 - int(30 * s),
           size[0] + int(32 * s), int(50 * s), alpha=0.82)
    _text(frame, text, x, h // 2, scale, AMBER, max(2, int(round(2 * s))))


def draw_keymap(frame: np.ndarray, keys: str) -> None:
    h, w = frame.shape[:2]
    s = ui_scale(frame)
    _text(frame, keys, int(12 * s), h - int(10 * s), 0.44 * s, GREY)


def render(
    frame: np.ndarray,
    state: FrameState,
    config: ExerciseConfig | None,
    show_angles: bool = True,
    frames_since_rep: int = -1,
    status_extra: str = "",
    keymap: str = "",
) -> np.ndarray:
    """Compose the full overlay onto ``frame`` in place and return it."""
    if state.detected and state.landmarks is not None:
        highlight = [n for m in state.active_mistakes for n in m.highlight]
        draw_skeleton(frame, state.landmarks,
                      highlight=highlight,
                      colour=GREY if state.generic else GREEN)
    else:
        draw_hint(frame, "No person detected - step into frame")

    draw_status(frame, state, status_extra)
    draw_angles(frame, state, config, show_angles)
    draw_warnings(frame, state)
    draw_framing(frame, state)
    draw_rep_flash(frame, frames_since_rep)
    if keymap:
        draw_keymap(frame, keymap)
    return frame


def blank_canvas(width: int = 720, height: int = 720) -> np.ndarray:
    """Dark canvas for replaying recorded sessions with no source video."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (18, 18, 20)
    step = 60
    for x in range(0, width, step):
        cv2.line(img, (x, 0), (x, height), (30, 30, 34), 1)
    for y in range(0, height, step):
        cv2.line(img, (0, y), (width, y), (30, 30, 34), 1)
    return img
