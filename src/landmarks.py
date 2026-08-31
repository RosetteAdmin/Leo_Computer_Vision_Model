"""MediaPipe Pose (BlazePose) landmark constants.

BlazePose returns 33 landmarks covering the whole body. Keeping the indices and
the skeleton topology in one place means every other module (angle engine,
overlay renderer, synthetic data generator) agrees on the layout.

Landmark array convention used across the project:
    ``np.ndarray`` of shape ``(33, 4)`` and dtype float32, columns are
    ``[x, y, z, visibility]``.

    * ``x``, ``y``  -> normalised image coordinates (0..1), ``y`` grows downward.
    * ``z``         -> BlazePose depth estimate, roughly in the same scale as x
                       (negative = closer to the camera). Treated as advisory
                       only; all form logic is 2D so that it stays stable on a
                       single webcam.
    * ``visibility``-> 0..1 confidence that the landmark is visible.
"""

from __future__ import annotations

NUM_LANDMARKS = 33

# --- indices -----------------------------------------------------------------
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

NAME_TO_INDEX = {name: i for i, name in enumerate(LANDMARK_NAMES)}

# --- topology ----------------------------------------------------------------
#: Bones drawn by the overlay renderer. Torso is closed so posture reads clearly.
SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    # face (minimal - just enough to see head orientation)
    (LEFT_EAR, LEFT_EYE),
    (LEFT_EYE, NOSE),
    (NOSE, RIGHT_EYE),
    (RIGHT_EYE, RIGHT_EAR),
    # arms
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (LEFT_WRIST, LEFT_INDEX),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (RIGHT_WRIST, RIGHT_INDEX),
    # torso
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    # legs
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL),
    (LEFT_HEEL, LEFT_FOOT_INDEX),
    (LEFT_ANKLE, LEFT_FOOT_INDEX),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (RIGHT_HEEL, RIGHT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
)

#: Landmarks that carry the movement signal. Used for the classifier's
#: coordinate features and for the "is the person actually in frame" check.
CORE_LANDMARKS: tuple[int, ...] = (
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX,
)

#: Minimum set that must be visible for angle/rep logic to be trustworthy.
TRUNK_LANDMARKS: tuple[int, ...] = (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
)
