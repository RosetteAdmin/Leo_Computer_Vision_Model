"""Stage 4 - classifier training.

    python -m src.train --data data/raw_sessions --out models/exercise_classifier.pkl

Trains ``SimpleImputer -> StandardScaler -> classifier`` on windowed features,
fits the IsolationForest novelty detector that backs the "unknown" path, prints a
held-out report, and saves one ``.pkl`` bundle plus a ``.json`` summary.

Model choices, in the order worth trying:

``random_forest``
    Default. No feature scaling sensitivity, handles the mixed angle/coordinate
    feature space, trains in seconds, and predicts a 285-dim vector in well under
    a millisecond.
``svm``
    RBF SVM. Sometimes better with very little data; slower at inference.
``logreg``
    Linear baseline. Use it to check the task is not trivially separable - if
    logreg already scores 99%, the test split is probably leaking.
``xgboost``
    Needs the optional ``xgboost`` package.

Escalating to an LSTM/BiLSTM (as the brief allows) is only worth it if the
windowed-statistics features genuinely plateau; they capture posture and motion
well enough that the sequence model is usually not the bottleneck.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .classifier import ClassifierBundle
from .dataset import build_from_dir, describe, grouped_split
from .features import DEFAULT_STRIDE, DEFAULT_WINDOW, WINDOW_FEATURE_NAMES


def make_estimator(model_type: str, seed: int = 0):
    """Build the sklearn pipeline for a model type."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        # n_jobs=1 on purpose: at inference the pipeline predicts ONE window at
        # a time, and joblib's thread dispatch costs more than the trees do.
        # Measured on this machine: 300 trees @ n_jobs=-1 ~29 ms/window vs
        # 200 trees @ n_jobs=1 ~2 ms/window. Training is ~1 s either way.
        model = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=seed,
        )
    elif model_type == "svm":
        from sklearn.svm import SVC

        model = SVC(C=10.0, gamma="scale", probability=True, class_weight="balanced",
                    random_state=seed)
    elif model_type == "logreg":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("xgboost is not installed. `pip install xgboost`") from exc

        model = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.8, tree_method="hist",
            random_state=seed, n_jobs=-1,
        )
    else:
        raise SystemExit(f"unknown --model-type '{model_type}'")

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def make_novelty(contamination: float, seed: int = 0):
    """Out-of-distribution detector backing the "unknown" label."""
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("iso", IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def train(args: argparse.Namespace) -> ClassifierBundle:
    ds = build_from_dir(args.data, window=args.window, stride=args.stride, only_good_form=True)
    print(describe(ds))

    train_idx, test_idx = grouped_split(ds, test_size=args.test_size, seed=args.seed,
                                        group_by=args.group_by)
    tr, te = ds.subset(train_idx), ds.subset(test_idx)
    print(f"\nsplit by {args.group_by}: {len(tr)} train / {len(te)} test windows")

    pipe = make_estimator(args.model_type, args.seed)
    t0 = time.perf_counter()
    pipe.fit(tr.X, tr.y)
    fit_seconds = time.perf_counter() - t0

    novelty = make_novelty(args.contamination, args.seed)
    novelty.fit(tr.X)

    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    pred = pipe.predict(te.X)
    acc = float(accuracy_score(te.y, pred))
    f1 = float(f1_score(te.y, pred, average="macro"))
    labels = sorted(set(ds.y.tolist()))
    cm = confusion_matrix(te.y, pred, labels=labels)

    print(f"\ntrained {args.model_type} in {fit_seconds:.1f}s")
    print(f"held-out accuracy {acc:.4f}   macro-F1 {f1:.4f}\n")
    print(classification_report(te.y, pred, labels=labels, zero_division=0))
    print("confusion matrix (rows = truth, cols = predicted)")
    print("            " + "".join(f"{l[:10]:>12}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l[:11]:<12}" + "".join(f"{v:>12}" for v in cm[i]))

    # inference latency of the classifier alone
    t0 = time.perf_counter()
    n_bench = min(500, len(te))
    for i in range(n_bench):
        pipe.predict_proba(te.X[i:i + 1])
    per_call_ms = (time.perf_counter() - t0) / max(1, n_bench) * 1000.0
    print(f"\nclassifier latency: {per_call_ms:.2f} ms per window")

    bundle = ClassifierBundle(
        pipeline=pipe,
        labels=tuple(labels),
        feature_names=WINDOW_FEATURE_NAMES,
        window=args.window,
        stride=args.stride,
        min_confidence=args.min_confidence,
        novelty=novelty,
        novelty_override_confidence=args.novelty_override,
        model_type=args.model_type,
        metadata={
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "data_dir": str(Path(args.data).resolve()),
            "n_train_windows": len(tr),
            "n_test_windows": len(te),
            "group_by": args.group_by,
            "fit_seconds": round(fit_seconds, 2),
            "test_accuracy": round(acc, 4),
            "test_macro_f1": round(f1, 4),
            "classifier_latency_ms": round(per_call_ms, 3),
            "labels": labels,
            "confusion_matrix": cm.tolist(),
            "class_report": classification_report(
                te.y, pred, labels=labels, zero_division=0, output_dict=True
            ),
            "window_counts": ds.counts(),
            "synthetic_data": all(
                bool(c.meta.get("synthetic")) for c in ds.clips
            ),
        },
    )
    out = bundle.save(args.out)
    print(f"\nsaved -> {out}")
    print(f"        {out.with_suffix('.json')}")
    if bundle.metadata["synthetic_data"]:
        print(
            "\nNOTE: every training clip is synthetic. These accuracy figures\n"
            "      describe the pipeline logic, not real-world performance.\n"
            "      Record real clips (main.py --record) before quoting them."
        )
    return bundle


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the exercise classifier.")
    p.add_argument("--data", default="data/raw_sessions", help="directory of .npz sessions")
    p.add_argument("--out", default="models/exercise_classifier.pkl")
    p.add_argument("--model-type", default="random_forest",
                   choices=["random_forest", "svm", "logreg", "xgboost"])
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--group-by", default="clip", choices=["clip", "subject"],
                   help="hold out whole clips (default) or whole subjects")
    p.add_argument("--min-confidence", type=float, default=0.65,
                   help="below this max-probability a window is labelled unknown")
    p.add_argument("--contamination", type=float, default=0.05,
                   help="IsolationForest contamination for the novelty gate")
    p.add_argument("--novelty-override", type=float, default=0.85,
                   help="probability above which the novelty gate is ignored")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
