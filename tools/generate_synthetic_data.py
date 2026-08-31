"""Generate the synthetic development dataset.

    python -m tools.generate_synthetic_data

Writes three sets:

``data/raw_sessions``   good-form clips of the five trained exercises -> training
``data/test_faults``    good + deliberately-faulty clips -> mistake-detection test
``data/test_unseen``    movements never trained on -> "unknown" fallback test

This is a **development test bench**, not a dataset in the scientific sense. See
the warning at the top of ``src/synthetic.py``, and replace it with real
recordings (``python -m src.main --record squat``) before believing any number.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.synthetic import (FAULTS, GOOD, TRAINING_EXERCISES, UNSEEN_EXERCISES,
                           generate_clip)


def _clean(path: Path, wipe: bool) -> Path:
    if wipe and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--train-clips", type=int, default=8,
                   help="good-form clips per exercise for training")
    p.add_argument("--fault-clips", type=int, default=3, help="clips per fault variant")
    p.add_argument("--train-fault-clips", type=int, default=2,
                   help="bad-form clips added to the TRAINING set (identity is "
                        "independent of form quality; 0 to disable)")
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--keep", action="store_true", help="do not wipe existing directories")
    args = p.parse_args(argv)

    root = Path(args.root)
    train_dir = _clean(root / "raw_sessions", not args.keep)
    fault_dir = _clean(root / "test_faults", not args.keep)
    unseen_dir = _clean(root / "test_unseen", not args.keep)

    n = 0

    # Training clips span a range of depths and tempos on purpose. Without that
    # spread the classifier's novelty gate rejects shallow or hurried reps as
    # "unknown", which silently disables feedback for exactly the users who
    # need it. See src/synthetic.generate_clip.
    import numpy as np

    rng = np.random.default_rng(args.seed)

    print(f"[train] good-form clips (varied depth/tempo) -> {train_dir}")
    for ex in TRAINING_EXERCISES:
        for k in range(args.train_clips):
            n += 1
            s = generate_clip(
                ex, reps=args.reps, fault=GOOD, seed=args.seed * 1000 + n,
                subject=f"subj{k % 4}",
                depth_scale=float(rng.uniform(0.72, 1.02)),
                period_jitter=(0.62, 1.45),
                pose_jitter=10.0,
            )
            s.save(train_dir / f"{ex}_good_{k:02d}.npz")
        print(f"  {ex:<16} {args.train_clips} clips")

    # Bad-form clips in the training set. A leaning lunge or a sagging push-up
    # is still a lunge / a push-up: exercise identity is independent of form
    # quality. Without these the classifier drops to "unknown" on badly executed
    # reps and the rep counter stops - measured end-to-end rep MAE went from
    # 1.01 to well under 1 once they were included. They are marked
    # fault="good" only so build_dataset keeps them; the fault name stays in
    # `form_variant` for traceability. Test-set clips are sampled independently.
    if args.train_fault_clips > 0:
        print(f"\n[train] bad-form clips (same labels) -> {train_dir}")
        for ex in TRAINING_EXERCISES:
            for fault in FAULTS.get(ex, ()):
                if fault is GOOD:
                    continue
                for k in range(args.train_fault_clips):
                    n += 1
                    s = generate_clip(ex, reps=args.reps, fault=fault,
                                      seed=args.seed * 1000 + 500_000 + n,
                                      subject=f"subj{k % 4}", pose_jitter=6.0)
                    s.meta["form_variant"] = fault.name
                    s.meta["fault"] = "good"        # keep in the identity training set
                    s.meta["expected_codes"] = []   # not a mistake-detection label
                    s.save(train_dir / f"{ex}_form_{fault.name}_{k:02d}.npz")
            print(f"  {ex:<16} {len(FAULTS.get(ex, ())) - 1} variants "
                  f"x{args.train_fault_clips}")

    print(f"\n[faults] good + bad-form clips -> {fault_dir}")
    for ex in TRAINING_EXERCISES:
        for fault in FAULTS.get(ex, (GOOD,)):
            for k in range(args.fault_clips):
                n += 1
                s = generate_clip(ex, reps=args.reps, fault=fault, seed=args.seed * 1000 + n,
                                  subject=f"subj{k % 4}")
                s.save(fault_dir / f"{ex}_{fault.name}_{k:02d}.npz")
            codes = ",".join(fault.expected_codes) or "-"
            print(f"  {ex:<16} {fault.name:<12} x{args.fault_clips}  expect: {codes}")

    print(f"\n[unseen] out-of-set movements -> {unseen_dir}")
    for ex in UNSEEN_EXERCISES:
        for k in range(args.fault_clips):
            n += 1
            s = generate_clip(ex, reps=args.reps, fault=GOOD, seed=args.seed * 1000 + n,
                              subject=f"subj{k % 4}")
            s.save(unseen_dir / f"{ex}_{k:02d}.npz")
        print(f"  {ex:<16} {args.fault_clips} clips")

    print(f"\n{n} clips written under {root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
