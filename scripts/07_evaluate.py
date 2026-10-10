"""Evaluate the trained model honestly: per product, baselines, CI, rolling origin.

Usage:
    python scripts/07_evaluate.py

Reads models/model.joblib and data/processed/examples.parquet, then updates
models/metrics.json with the detail that backtest_report serves.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mymarket import config  # noqa: E402
from mymarket.model.features import build_matrix  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger("evaluate")

BOOTSTRAP_ROUNDS = 1000


def bootstrap_ci(truth: np.ndarray, predicted: np.ndarray, rounds: int = BOOTSTRAP_ROUNDS):
    """Percentile confidence interval for macro F1, resampling rows."""
    from sklearn.metrics import f1_score

    if len(truth) < 5:
        return None, None
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(rounds):
        idx = rng.integers(0, len(truth), len(truth))
        scores.append(f1_score(truth[idx], predicted[idx], average="macro", zero_division=0))
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def directional_accuracy(truth: np.ndarray, predicted: np.ndarray) -> float | None:
    """Accuracy on the rows that actually moved, ignoring STABLE (PRD 6.5)."""
    mask = truth != "STABLE"
    if not mask.sum():
        return None
    return float((truth[mask] == predicted[mask]).mean())


def rolling_origin(frame: pd.DataFrame, bundle) -> list[dict]:
    """Train on everything before each quarter, test on that quarter."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import OneHotEncoder

    from mymarket.filter import embed_filter

    cache = config.DATA_PROCESSED / "embeddings.npy"
    if cache.exists() and len(np.load(cache)) == len(frame):
        vectors = np.load(cache)
    else:
        vectors = embed_filter.encode(frame["text"].fillna("").tolist())

    quarters = sorted(frame["quarter"].unique())
    results: list[dict] = []

    for quarter in quarters:
        is_test = (frame["quarter"] == quarter).to_numpy()
        is_train = (frame["quarter"] < quarter).to_numpy()
        if is_train.sum() < 50 or is_test.sum() < 10:
            continue
        if frame.loc[is_train, "label"].nunique() < 2:
            continue

        encoder = OneHotEncoder(handle_unknown="ignore")
        matrix = build_matrix(frame, vectors, encoder, fit=True)
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(matrix[is_train], frame.loc[is_train, "label"])

        truth = frame.loc[is_test, "label"].to_numpy()
        predicted = model.predict(matrix[is_test])
        majority = frame.loc[is_train, "label"].mode()[0]

        results.append(
            {
                "quarter": str(quarter),
                "n_train": int(is_train.sum()),
                "n_test": int(is_test.sum()),
                "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
                "majority_macro_f1": float(
                    f1_score(truth, [majority] * len(truth), average="macro", zero_division=0)
                ),
            }
        )

    return results


def main(argv: list[str] | None = None) -> int:
    import joblib
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    if not config.MODEL_FILE.exists():
        log.error(
            "%s not found. Train first with scripts/06_train_local.py or the Colab "
            "notebook (manual step M6).",
            config.MODEL_FILE,
        )
        return 1
    if not config.EXAMPLES.exists():
        log.error("%s not found. Run scripts/05_build_examples.py first.", config.EXAMPLES)
        return 1

    bundle = joblib.load(config.MODEL_FILE)
    frame = pd.read_parquet(config.EXAMPLES)
    frame = frame[frame["split"].isin({"train", "val", "test"})].reset_index(drop=True)

    from mymarket.filter import embed_filter

    cache = config.DATA_PROCESSED / "embeddings.npy"
    if cache.exists() and len(np.load(cache)) == len(frame):
        vectors = np.load(cache)
        log.info("reusing cached embeddings")
    else:
        log.info("embedding %d texts", len(frame))
        vectors = embed_filter.encode(frame["text"].fillna("").tolist())

    matrix = build_matrix(frame, vectors, bundle["ohe"], fit=False)
    model = bundle["model"]

    test = (frame["split"] == "test").to_numpy()
    truth = frame.loc[test, "label"].to_numpy()
    predicted = model.predict(matrix[test])
    probabilities = model.predict_proba(matrix[test])

    overall = float(f1_score(truth, predicted, average="macro", zero_division=0))
    low, high = bootstrap_ci(truth, predicted)

    print("\noverall test report:")
    print(classification_report(truth, predicted, zero_division=0))
    print("confusion matrix, labels", list(model.classes_))
    print(confusion_matrix(truth, predicted, labels=model.classes_))

    test_frame = frame[test].copy()
    test_frame["pred"] = predicted
    test_frame["confidence"] = probabilities.max(axis=1)

    per_product: dict[str, dict] = {}
    print("\nper product:")
    for product, group in test_frame.groupby("product"):
        product_truth = group["label"].to_numpy()
        product_pred = group["pred"].to_numpy()
        p_low, p_high = bootstrap_ci(product_truth, product_pred)
        stats = {
            "n_test": int(len(group)),
            "macro_f1": float(f1_score(product_truth, product_pred, average="macro", zero_division=0)),
            "macro_f1_ci95": [p_low, p_high],
            "directional_accuracy": directional_accuracy(product_truth, product_pred),
            "label_counts": group["label"].value_counts().to_dict(),
        }
        per_product[product] = stats
        print(
            f"  {product}: macroF1 {stats['macro_f1']:.3f} "
            f"n={stats['n_test']} dir_acc={stats['directional_accuracy']}"
        )

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import import_module

    baseline_module = import_module("06_train_local")
    base = baseline_module.baselines(frame, "test")

    print("\nmodel vs baselines on the test set:")
    print(f"  {'model':<12} {overall:.4f}")
    for name, value in base.items():
        print(f"  {name:<12} {value:.4f}")
    beats = bool(base) and overall > max(base.values())
    print(f"\nbeats every baseline: {beats}")
    if not beats:
        print(
            "This is a normal outcome for this task. Say so in the demo and show "
            "the keyword triggers and the price context as the value instead."
        )

    alert_mask = test_frame["confidence"] >= 0.70
    alert_precision = (
        float((test_frame.loc[alert_mask, "pred"] == test_frame.loc[alert_mask, "label"]).mean())
        if alert_mask.sum()
        else None
    )

    log.info("running rolling origin backtest")
    rolling = rolling_origin(frame, bundle)
    if rolling:
        print("\nrolling origin by quarter:")
        for row in rolling:
            print(
                f"  {row['quarter']}: macroF1 {row['macro_f1']:.3f} "
                f"(majority {row['majority_macro_f1']:.3f}) n={row['n_test']}"
            )

    existing = {}
    if config.METRICS_FILE.exists():
        try:
            existing = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    metrics = {
        **existing,
        "generated_at": pd.Timestamp.now(tz=config.TZ).isoformat(),
        "test_macro_f1": overall,
        "macro_f1_ci95": [low, high],
        "baselines_test": base,
        "beats_all_baselines": beats,
        "directional_accuracy": directional_accuracy(truth, predicted),
        "alert_precision_p70": alert_precision,
        "n_alerts_p70": int(alert_mask.sum()),
        "n_test": int(test.sum()),
        "per_product": per_product,
        "rolling_origin": rolling,
        "model_version": str(bundle.get("version", "unknown")),
        "horizon_days": int(frame["horizon_days"].iloc[0]),
        "limits": [
            "Direction over a fixed horizon, not a price level forecast.",
            "News is a weak signal for grocery prices. Expect modest numbers.",
            "Price truth is PriceCatcher, a KPDN price survey, not an inflation statistic.",
            "Labels come from a normalised index of a few mapped items per product.",
        ],
    }
    config.METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nwrote {config.METRICS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
