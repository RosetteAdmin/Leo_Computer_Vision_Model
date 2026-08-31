"""Structured error code -> user-facing coaching text.

Keeping this separate from detection means the detector only ever deals in
stable machine codes, and the wording (or language, or TTS voice) can change
without touching any logic. Unknown codes degrade to a readable fallback rather
than raising, so a new code added in a YAML config still shows something useful.
"""

from __future__ import annotations

from typing import Iterable

from .mistake_detector import Mistake

MESSAGES: dict[str, str] = {
    # --- angle mistakes -----------------------------------------------------
    "EXCESSIVE_TORSO_LEAN": "Chest up - you're leaning too far forward.",
    "KNEE_ALIGNMENT_POOR": "Keep your knees aligned with your toes.",
    "NECK_MISALIGNED": "Keep your head in line with your spine.",
    "HIP_HINGE_TOO_EARLY": "Sit down, not back - lead with your hips less.",
    "HIP_SAG": "Tighten your core - your hips are sagging.",
    "HIP_PIKE": "Lower your hips - keep a straight line.",
    "ELBOW_FLARE": "Tuck your elbows closer to your body.",
    "SHOULDER_SWING": "Keep your upper arms still - only the elbow bends.",
    "TORSO_SWING": "Stop swinging your body - control the weight.",
    "WRIST_BENT": "Keep your wrists straight.",
    "INCOMPLETE_LOCKOUT": "Push all the way up to a full lockout.",
    # --- repetition mistakes -------------------------------------------------
    "PARTIAL_REP": "Partial rep - use the full range of motion.",
    "INSUFFICIENT_DEPTH": "Go deeper - that rep was too shallow.",
    "TOO_FAST": "Slow down - that rep was rushed.",
    "TOO_SLOW": "That rep was very slow - keep a steady tempo.",
    "ASYMMETRIC_MOVEMENT": "Uneven sides - balance left and right.",
    "INCONSISTENT_FORM": "That rep looked different from your others.",
    # --- generic mode --------------------------------------------------------
    "EXTREME_JOINT_ANGLE": "Extreme joint angle detected - check your position.",
    "LARGE_ASYMMETRY": "Large left/right difference detected.",
    "JERKY_MOTION": "Movement is jerky - try to move smoothly.",
    # --- pipeline / setup ----------------------------------------------------
    "SUBOPTIMAL_CAMERA_VIEW": "Turn about 45 degrees to the camera for accurate depth.",
}

#: Short labels for the on-screen list (long sentences do not fit next to a
#: skeleton overlay).
SHORT_LABELS: dict[str, str] = {
    "EXCESSIVE_TORSO_LEAN": "Torso lean",
    "KNEE_ALIGNMENT_POOR": "Knees caving",
    "NECK_MISALIGNED": "Neck alignment",
    "HIP_HINGE_TOO_EARLY": "Early hip hinge",
    "HIP_SAG": "Hips sagging",
    "HIP_PIKE": "Hips too high",
    "ELBOW_FLARE": "Elbow flare",
    "SHOULDER_SWING": "Shoulder swing",
    "TORSO_SWING": "Body swing",
    "WRIST_BENT": "Bent wrists",
    "INCOMPLETE_LOCKOUT": "No lockout",
    "PARTIAL_REP": "Partial rep",
    "INSUFFICIENT_DEPTH": "Too shallow",
    "TOO_FAST": "Too fast",
    "TOO_SLOW": "Too slow",
    "ASYMMETRIC_MOVEMENT": "Asymmetric",
    "INCONSISTENT_FORM": "Inconsistent",
    "EXTREME_JOINT_ANGLE": "Extreme angle",
    "LARGE_ASYMMETRY": "Asymmetry",
    "JERKY_MOTION": "Jerky motion",
    "SUBOPTIMAL_CAMERA_VIEW": "Camera angle",
}


def _humanise(code: str) -> str:
    return code.replace("_", " ").capitalize() + "."


def message_for(code: str) -> str:
    """Full coaching sentence for a code."""
    return MESSAGES.get(code, _humanise(code))


def short_for(code: str) -> str:
    """Compact label for the overlay."""
    return SHORT_LABELS.get(code, code.replace("_", " ").title())


def describe(mistake: Mistake, include_numbers: bool = True) -> str:
    """One line describing a mistake, optionally with the measured value."""
    text = short_for(mistake.code)
    if mistake.detail:
        text = f"{text} ({mistake.detail})"
    if include_numbers and mistake.value is not None:
        unit = "" if mistake.metric is None or "_" in (mistake.metric or "") else ""
        text = f"{text}: {mistake.value:.0f}{unit}"
        if mistake.limit is not None:
            text += f" / {mistake.limit:.0f}"
    return text


def coach(mistakes: Iterable[Mistake], limit: int = 3) -> list[str]:
    """Deduplicated coaching sentences, worst first, capped at ``limit``."""
    order = {"error": 0, "warning": 1, "info": 2}
    ranked = sorted(mistakes, key=lambda m: order.get(m.severity, 3))
    out: list[str] = []
    seen: set[str] = set()
    for m in ranked:
        if m.code in seen:
            continue
        seen.add(m.code)
        out.append(message_for(m.code))
        if len(out) >= limit:
            break
    return out
