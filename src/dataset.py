"""Stage 3 - dataset assembly from recorded sessions.

Turns a directory of ``.npz`` sessions into windowed feature matrices with group
labels, then splits them **by clip** (optionally by subject).

Why the grouping matters: consecutive windows from one clip overlap and are
almost identical. A plain random split puts near-duplicates on both sides and
inflates accuracy - typically by 10-20 points on data like this. Every split
here keeps a whole clip on one side.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .features import DEFAULT_STRIDE, DEFAULT_WINDOW, windows_from_sequence
from .session import Session, load_sessions


@dataclass
class Dataset:
    X: np.ndarray                 # (n_windows, n_features)
    y: np.ndarray                 # (n_windows,) str labels
    groups: np.ndarray            # (n_windows,) clip id
    subjects: np.ndarray          # (n_windows,) subject id
    clips: list[Session]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.y.tolist())))

    def counts(self) -> dict[str, int]:
        vals, cnt = np.unique(self.y, return_counts=True)
        return {str(v): int(c) for v, c in zip(vals, cnt)}

    def subset(self, idx: np.ndarray) -> "Dataset":
        return Dataset(self.X[idx], self.y[idx], self.groups[idx], self.subjects[idx], self.clips)


def build_dataset(
    sessions: Sequence[Session],
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    only_good_form: bool = True,
) -> Dataset:
    """Window every session into classifier features.

    ``only_good_form=True`` keeps just ``fault == "good"`` clips. Training the
    *identity* classifier on deliberately broken reps would teach it that a
    sagging push-up is a different exercise; bad-form clips belong in the
    mistake-detection test set, not here.
    """
    Xs, ys, gs, ss = [], [], [], []
    for i, sess in enumerate(sessions):
        if only_good_form and sess.fault != "good":
            continue
        W = windows_from_sequence(sess.features, window=window, stride=stride)
        if W.shape[0] == 0:
            continue
        Xs.append(W)
        ys.append(np.full(W.shape[0], sess.label, dtype=object))
        gid = str(sess.path.name) if sess.path else f"clip{i:04d}"
        gs.append(np.full(W.shape[0], gid, dtype=object))
        ss.append(np.full(W.shape[0], sess.subject, dtype=object))

    if not Xs:
        raise ValueError("no usable windows - are the clips shorter than the window length?")
    return Dataset(
        X=np.vstack(Xs).astype(np.float32),
        y=np.concatenate(ys).astype(str),
        groups=np.concatenate(gs).astype(str),
        subjects=np.concatenate(ss).astype(str),
        clips=list(sessions),
    )


def build_from_dir(
    directory: str | Path,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    only_good_form: bool = True,
) -> Dataset:
    sessions = load_sessions(directory)
    if not sessions:
        raise FileNotFoundError(f"no .npz sessions found under {directory}")
    return build_dataset(sessions, window, stride, only_good_form)


def grouped_split(
    ds: Dataset,
    test_size: float = 0.25,
    seed: int = 0,
    group_by: str = "clip",
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices so no clip (or subject) appears on both sides.

    Uses ``StratifiedGroupKFold`` rather than ``GroupShuffleSplit``: with only a
    handful of clips per exercise a plain group shuffle regularly produces a test
    fold that is missing an entire class, which makes the macro-F1 meaningless.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    groups = ds.groups if group_by == "clip" else ds.subjects
    n_splits = int(round(1.0 / max(0.05, min(0.5, test_size))))
    n_groups = len(set(groups.tolist()))
    n_splits = max(2, min(n_splits, n_groups))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(ds.X, ds.y, groups))
    return train_idx, test_idx


def describe(ds: Dataset) -> str:
    counts = ds.counts()
    lines = [
        f"{len(ds)} windows, {ds.X.shape[1]} features, "
        f"{len(set(ds.groups.tolist()))} clips, {len(set(ds.subjects.tolist()))} subjects",
    ]
    for label, n in sorted(counts.items()):
        lines.append(f"  {label:<18} {n:>6} windows")
    return "\n".join(lines)


def concat(datasets: Iterable[Dataset]) -> Dataset:
    datasets = list(datasets)
    return Dataset(
        X=np.vstack([d.X for d in datasets]),
        y=np.concatenate([d.y for d in datasets]),
        groups=np.concatenate([d.groups for d in datasets]),
        subjects=np.concatenate([d.subjects for d in datasets]),
        clips=[c for d in datasets for c in d.clips],
    )
