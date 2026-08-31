"""Stage 1a - MediaPipe Pose wrapper.

Responsibility: turn a BGR frame into a :class:`PoseFrame` containing *all* 33
landmarks. This module deliberately knows nothing about exercises; it is the
universal front-end of the pipeline.

MediaPipe is imported lazily so that the rest of the project (angle engine,
rep counter, mistake detector, training, tests) can be used on recorded /
synthetic landmark data on a machine without mediapipe installed.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import landmarks as LM


def _quieten_mediapipe() -> None:
    """Silence MediaPipe's third-party log noise.

    MediaPipe emits a protobuf ``SymbolDatabase.GetPrototype()`` deprecation
    warning on essentially every call, plus TensorFlow-Lite and absl banners.
    Left alone it prints hundreds of lines per session and buries the rep counts
    and coaching messages this program actually wants to show the user. None of
    it is actionable from here - it comes from inside the library.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.*")
    warnings.filterwarnings("ignore", message=r".*SymbolDatabase\.GetPrototype\(\).*")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "2")
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:       # absl is a MediaPipe dependency, but never assume
        pass


@dataclass
class PoseFrame:
    """One frame of pose data.

    Attributes
    ----------
    index:
        Monotonic frame counter.
    timestamp:
        Seconds (``time.perf_counter``) when the frame was captured.
    landmarks:
        ``(33, 4)`` float32 array ``[x, y, z, visibility]`` in normalised image
        coordinates, or ``None`` when no person was detected.
    image_size:
        ``(width, height)`` of the source frame in pixels.
    """

    index: int
    timestamp: float
    landmarks: Optional[np.ndarray]
    image_size: tuple[int, int] = (0, 0)
    world_landmarks: Optional[np.ndarray] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def detected(self) -> bool:
        return self.landmarks is not None

    @property
    def mean_visibility(self) -> float:
        if self.landmarks is None:
            return 0.0
        return float(self.landmarks[list(LM.CORE_LANDMARKS), 3].mean())

    def trunk_visible(self, threshold: float = 0.5) -> bool:
        """True when shoulders+hips are confidently visible.

        Every angle in the engine is normalised by torso length, so a missing
        trunk means nothing downstream can be trusted.
        """
        if self.landmarks is None:
            return False
        return bool((self.landmarks[list(LM.TRUNK_LANDMARKS), 3] > threshold).all())


class PoseExtractor:
    """Thin, reusable wrapper around ``mediapipe.solutions.pose.Pose``.

    Parameters
    ----------
    model_complexity:
        0 = lite (fastest, use on weak CPUs / Raspberry Pi), 1 = full (default,
        best speed/accuracy trade-off on a laptop), 2 = heavy (too slow on CPU).
    static_image_mode:
        Must stay ``False`` for video: MediaPipe then tracks between frames
        instead of re-running detection every frame, which is the main reason
        this is real-time on CPU.
    smooth_landmarks:
        MediaPipe's built-in temporal filter. Keep on for live use; turn off
        when re-processing recorded clips frame-by-frame for reproducibility.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth_landmarks: bool = True,
        static_image_mode: bool = False,
    ) -> None:
        _quieten_mediapipe()
        try:
            import mediapipe as mp  # noqa: PLC0415  (lazy on purpose)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "mediapipe is required for live/video pose extraction. "
                "Install it with `pip install -r requirements.txt`."
            ) from exc

        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._frame_index = -1

    # -- lifecycle ------------------------------------------------------------
    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- main entry point -----------------------------------------------------
    def process(self, frame_bgr: np.ndarray, timestamp: float | None = None) -> PoseFrame:
        """Run pose estimation on a single BGR frame."""
        import cv2  # noqa: PLC0415

        self._frame_index += 1
        h, w = frame_bgr.shape[:2]
        ts = time.perf_counter() if timestamp is None else timestamp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        # The filter is re-applied per call: MediaPipe re-registers protobuf
        # messages internally, which resets warning state on some versions.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self._pose.process(rgb)

        if result.pose_landmarks is None:
            return PoseFrame(self._frame_index, ts, None, (w, h))

        pts = np.empty((LM.NUM_LANDMARKS, 4), dtype=np.float32)
        for i, lm in enumerate(result.pose_landmarks.landmark):
            pts[i] = (lm.x, lm.y, lm.z, lm.visibility)

        world = None
        if getattr(result, "pose_world_landmarks", None) is not None:
            world = np.array(
                [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_world_landmarks.landmark],
                dtype=np.float32,
            )

        return PoseFrame(self._frame_index, ts, pts, (w, h), world)


# --- normalisation -----------------------------------------------------------
def normalise_landmarks(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Translate to hip-centre and scale by torso length.

    Returns ``(normalised_xy, torso_length)`` where ``normalised_xy`` has shape
    ``(33, 2)``. This removes the person's position in frame and their distance
    from the camera, which is what makes classifier features transferable
    between people and camera setups.

    ``torso_length`` is the mid-hip -> mid-shoulder distance in normalised image
    units; it is also the unit used by every distance-style metric in the angle
    engine.
    """
    hip = 0.5 * (pts[LM.LEFT_HIP, :2] + pts[LM.RIGHT_HIP, :2])
    shoulder = 0.5 * (pts[LM.LEFT_SHOULDER, :2] + pts[LM.RIGHT_SHOULDER, :2])
    torso = float(np.linalg.norm(shoulder - hip))
    if torso < 1e-6:
        torso = 1e-6
    return (pts[:, :2] - hip) / torso, torso


def aspect_correct(pts: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Convert normalised (0..1, 0..1) coords to a square, isotropic space.

    MediaPipe normalises x by width and y by height independently, so on a 16:9
    frame a 45-degree limb does not measure 45 degrees. Multiplying x by the
    aspect ratio restores true angles. Every consumer of landmark geometry
    should call this first.
    """
    w, h = image_size
    if not w or not h:
        return pts[:, :2].astype(np.float32, copy=True)
    out = pts[:, :2].astype(np.float32, copy=True)
    out[:, 0] *= float(w) / float(h)
    return out
