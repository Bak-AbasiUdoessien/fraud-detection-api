"""
train.py  -  Train, compare and save a fraud model.

    python -m fraud.train                                  # made-up data (generated on the fly)
    python -m fraud.train --data data/PS_20174392719_1491204439457_log.csv   # the real PaySim file

Fraud is RARE (well under 1% of transactions), so accuracy is useless: a model that says
"never fraud" is 99%+ accurate and catches nothing. We judge models by PR-AUC (average
precision) and by precision/recall at a chosen threshold instead.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib                                              # saves/loads trained models (ships with scikit-learn)
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, build_features
from .synthetic import generate

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def best_threshold(y_true, proba):
    """The probability cut-off that gives the best F1 (balance of precision and recall) on validation data."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)     # tiny number avoids 0/0
    i = int(f1[:-1].argmax())                                       # last point has no threshold, so skip it
    return float(thresholds[i])


def evaluate(y_true, proba, threshold, amounts):
    """All the numbers we report for one model on one dataset."""
    pred = (proba >= threshold).astype(int)
    caught = amounts[(pred == 1) & (y_true == 1)].sum()             # money in fraud we flagged
    total = amounts[y_true == 1].sum()                              # all money in fraud
    return {
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "fraud_amount_caught_pct": round(100 * caught / total, 2) if total else None,
        "alerts_per_1000_tx": round(1000 * pred.mean(), 2),         # workload for the fraud team
    }


def train(df, out_dir=MODELS_DIR, source="synthetic"):
    """Train two models, keep the better one (by validation PR-AUC), save it with its metadata."""
    X, y, amounts = build_features(df), df["isFraud"].to_numpy(), df["amount"].to_numpy()
    # 60% train / 20% validation (to pick the model + threshold) / 20% test (touched ONCE, at the end).
    # stratify=y keeps the same fraud rate in every split, which matters when fraud is rare.
    X_tmp, X_test, y_tmp, y_test, a_tmp, a_test = train_test_split(
        X, y, amounts, test_size=0.2, stratify=y, random_state=0)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=0)

    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(),     # scale features, then a linear model
                                             LogisticRegression(class_weight="balanced", max_iter=2000)),
        "gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced", max_iter=300,
                                                            learning_rate=0.1, random_state=0),
    }
    results = {}
    for name, model in candidates.items():
        start = time.perf_counter()
        model.fit(X_tr, y_tr)
        proba_val = model.predict_proba(X_val)[:, 1]               # column 1 = probability of fraud
        results[name] = {"model": model, "pr_auc_val": average_precision_score(y_val, proba_val),
                         "seconds": round(time.perf_counter() - start, 1)}
        print(f"  {name:<20} validation PR-AUC {results[name]['pr_auc_val']:.4f}  ({results[name]['seconds']}s)")

    best_name = max(results, key=lambda n: results[n]["pr_auc_val"])
    model = results[best_name]["model"]
    threshold = best_threshold(y_val, model.predict_proba(X_val)[:, 1])
    test_metrics = evaluate(y_test, model.predict_proba(X_test)[:, 1], threshold, a_test)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    metadata = {
        "model": best_name, "threshold": round(threshold, 4), "features": FEATURES,
        "test_metrics": test_metrics, "trained_on": source, "rows": len(df),
        "fraud_rate": round(float(y.mean()), 5),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,                     # models must be loaded with the same version
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    p = argparse.ArgumentParser(description="Train the fraud model")
    p.add_argument("--data", help="path to the PaySim CSV (default: generate made-up data)")
    p.add_argument("--rows", type=int, default=200_000, help="rows of made-up data to generate")
    p.add_argument("--sample", type=int, help="randomly sample this many rows (PaySim has 6.3 million)")
    a = p.parse_args()

    if a.data:
        df, source = pd.read_csv(a.data), f"PaySim ({Path(a.data).name})"
    else:
        df, source = generate(a.rows), "SYNTHETIC (made-up) data"
    if a.sample and a.sample < len(df):
        df = df.sample(a.sample, random_state=0)
    print(f"Training on {len(df):,} rows of {source}; fraud rate {df['isFraud'].mean():.3%}")
    meta = train(df, source=source)
    print(f"\nChosen model: {meta['model']}  (threshold {meta['threshold']})")
    print("Test-set results:")
    for k, v in meta["test_metrics"].items():
        print(f"  {k:<26}{v}")
    print(f"Saved model to {MODELS_DIR}")


if __name__ == "__main__":
    main()
