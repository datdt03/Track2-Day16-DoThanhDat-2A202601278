#!/usr/bin/env python3
"""Benchmark LightGBM for the Credit Card Fraud Detection dataset.

Expected CSV format: a target column named ``Class`` and all other columns
as numerical features (the Kaggle creditcard.csv format).
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="creditcard.csv", help="Path to CSV dataset")
    parser.add_argument("--output", default="benchmark_result.json", help="Path to result JSON")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test-set fraction (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of boosting rounds")
    parser.add_argument("--latency-runs", type=int, default=1000, help="Repeated one-row predictions")
    return parser.parse_args()


def seconds_to_ms(seconds: float) -> float:
    return round(seconds * 1000, 6)


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    output_path = Path(args.output)

    if not data_path.is_file():
        print(f"Dataset not found: {data_path}", file=sys.stderr)
        return 2
    if not 0 < args.test_size < 1:
        print("--test-size must be between 0 and 1.", file=sys.stderr)
        return 2

    load_start = time.perf_counter()
    data = pd.read_csv(data_path)
    load_seconds = time.perf_counter() - load_start

    target_column = "Class"
    if target_column not in data.columns:
        print(f"Dataset must contain a '{target_column}' target column.", file=sys.stderr)
        return 2
    if data[target_column].nunique() != 2:
        print(f"'{target_column}' must contain exactly two classes.", file=sys.stderr)
        return 2

    X = data.drop(columns=[target_column])
    y = data[target_column]
    split_start = time.perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    split_seconds = time.perf_counter() - split_start

    model = LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    training_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - training_start

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "auc_roc": round(float(roc_auc_score(y_test, probabilities)), 6),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 6),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 6),
    }

    one_row = X_test.iloc[[0]]
    # Run one initial prediction so library/model initialization is not counted.
    model.predict_proba(one_row)
    latency_start = time.perf_counter()
    for _ in range(args.latency_runs):
        model.predict_proba(one_row)
    latency_seconds = (time.perf_counter() - latency_start) / args.latency_runs

    batch_size = min(1000, len(X_test))
    batch = X_test.iloc[:batch_size]
    model.predict_proba(batch)  # warm-up
    throughput_start = time.perf_counter()
    model.predict_proba(batch)
    batch_seconds = time.perf_counter() - throughput_start

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(data_path),
            "rows": int(len(data)),
            "features": int(X.shape[1]),
            "positive_rows": int(y.sum()),
            "positive_rate": round(float(y.mean()), 8),
        },
        "split": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "split_seconds": round(split_seconds, 6),
        },
        "model": {
            "type": "LGBMClassifier",
            "n_estimators_requested": args.n_estimators,
            "best_iteration": int(model.best_iteration_ or args.n_estimators),
        },
        "timing": {
            "data_load_seconds": round(load_seconds, 6),
            "training_seconds": round(training_seconds, 6),
            "inference_latency_one_row_ms": seconds_to_ms(latency_seconds),
            "inference_throughput_rows_per_second": round(batch_size / batch_seconds, 2),
            "throughput_batch_rows": batch_size,
            "throughput_batch_seconds": round(batch_seconds, 6),
            "latency_runs": args.latency_runs,
        },
        "metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2, ensure_ascii=False)
        result_file.write("\n")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved benchmark results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
