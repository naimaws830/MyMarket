"""Train the impact model on CPU and save the four model files.

Usage:
    python scripts/06_train_local.py
    python scripts/06_train_local.py --v0-only     # TF-IDF baseline only, fast

Trains the v0 TF-IDF baseline and the v1 embedding model, keeps whichever wins
on validation, calibrates it on validation, and scores the test set once.
The Colab notebook does the same thing on a GPU (manual step M6).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mymarket import config  # noqa: E402
from mymarket.model.features import (  # noqa: E402
    MERGED_LABEL,
    PRODUCT_COLUMN,
    build_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger("train")

EMB_CACHE = config.DATA_PROCESSED / "embeddings.npy"


def load_examples() -> pd.DataFrame:
    """Read examples.parquet and keep only the modelled splits."""
    if not config.EXAMPLES.exists():
        raise FileNotFoundError(
            f"{config.EXAMPLES} not found. Run scripts/05_build_examples.py first."
        )
    frame = pd.read_parquet(config.EXAMPLES)
    frame = frame[frame["split"].isin({"train", "val", "test"})].reset_index(drop=True)
    if frame.empty:
        raise ValueError("no train, val or test rows in examples.parquet")
    return frame


def embeddings_for(frame: pd.DataFrame, use_cache: bool = True):
    """Embed the example texts, caching to disk because this is the slow step."""
    from mymarket.filter import embed_filter

    if use_cache and EMB_CACHE.exists():
        cached = np.load(EMB_CACHE)
        if len(cached) == len(frame):
            log.info("reusing cached embeddings %s", cached.shape)
            return cached
        log.info("cached embeddings do not match the example count, re-embedding")

    log.info("embedding %d texts on CPU, this is the slow part", len(frame))
    vectors = embed_filter.encode(frame["text"].fillna("").tolist())
    np.save(EMB_CACHE, vectors)
    return vectors


def baselines(frame: pd.DataFrame, split: str) -> dict[str, float]:
    """Majority, momentum and keyword baselines on one split (PRD 6.5)."""
    from sklearn.metrics import f1_score

    train = frame[frame["split"] == "train"]
    target = frame[frame["split"] == split]
    if target.empty:
        return {}

    truth = target["label"].to_numpy()
    majority = train["label"].mode()[0] if not train.empty else "STABLE"

    # Momentum: the sign of each article's own trailing return, the price move
    # over the horizon just before it was published. An earlier version read the
    # last training row instead, which made it a constant per product predictor
    # and swung it by 0.25 macro F1 between runs.
    momentum = []
    for row in target.itertuples(index=False):
        previous = getattr(row, "prev_ret", None)
        if previous is None or pd.isna(previous):
            momentum.append(majority)
        else:
            theta = row.theta
            momentum.append("UP" if previous > theta else "DOWN" if previous < -theta else "STABLE")

    up_words = {"shortage", "flood", "drought", "bird flu", "feed cost", "import ban", "tariff", "levy"}
    down_words = {"subsidy", "subsidi", "ceiling price", "harga siling", "price control", "kawalan harga"}
    keyword = []
    for text in target["event_words"].fillna(""):
        found = {w.strip().lower() for w in str(text).split(",") if w.strip()}
        if found & up_words:
            keyword.append("UP")
        elif found & down_words:
            keyword.append("DOWN")
        else:
            keyword.append("STABLE")

    return {
        "majority": float(f1_score(truth, [majority] * len(truth), average="macro", zero_division=0)),
        "momentum": float(f1_score(truth, momentum, average="macro", zero_division=0)),
        "keyword": float(f1_score(truth, keyword, average="macro", zero_division=0)),
    }


def train_v0(frame: pd.DataFrame) -> float:
    """TF-IDF plus logistic regression. The honest floor for a text model."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val"]
    if train.empty or val.empty:
        return float("nan")

    vectoriser = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=2)
    x_train = vectoriser.fit_transform(train["text"].fillna(""))
    x_val = vectoriser.transform(val["text"].fillna(""))

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(x_train, train["label"])
    score = f1_score(val["label"], model.predict(x_val), average="macro", zero_division=0)
    log.info("v0 TF-IDF val macro F1: %.4f", score)
    return float(score)


def calibrate(estimator, x_val, y_val):
    """Calibrate probabilities on the validation set, or explain why it was skipped.

    Calibration needs every class the model knows to appear in the validation
    split. When one is missing, sklearn raises deep inside its cross validation,
    so this checks first and returns the uncalibrated model rather than failing
    the whole run. An uncalibrated model still predicts, its probabilities are
    just less trustworthy, and metrics.json records that.
    """
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV

    known = set(getattr(estimator, "classes_", []))
    present = set(np.unique(y_val))

    if len(y_val) < 30:
        return estimator, f"skipped, only {len(y_val)} validation rows"
    missing = known - present
    if missing:
        return estimator, f"skipped, validation split is missing class(es) {sorted(missing)}"

    smallest = min(int((y_val == c).sum()) for c in present)
    folds = max(2, min(5, smallest))

    try:
        from sklearn.frozen import FrozenEstimator

        model = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid", cv=folds)
    except ImportError:  # pragma: no cover - sklearn below 1.6
        model = CalibratedClassifierCV(estimator, method="sigmoid", cv="prefit")

    try:
        model.fit(x_val, y_val)
    except Exception as exc:  # noqa: BLE001 - calibration is a nice to have
        log.warning("calibration failed, using the uncalibrated model: %s", exc)
        return estimator, f"failed ({type(exc).__name__}), using uncalibrated probabilities"

    # Calibrating on a small, imbalanced validation set can collapse the model
    # onto the majority class, which is worse than not calibrating at all. Keep
    # it only if it does not hurt on the same data, and say which was chosen.
    from sklearn.metrics import f1_score

    raw_f1 = f1_score(y_val, estimator.predict(x_val), average="macro", zero_division=0)
    cal_f1 = f1_score(y_val, model.predict(x_val), average="macro", zero_division=0)
    if cal_f1 < raw_f1:
        return estimator, (
            f"rejected, it dropped validation macro F1 from {raw_f1:.3f} to {cal_f1:.3f} "
            f"on {len(y_val)} rows, so probabilities are uncalibrated"
        )
    return model, f"sigmoid on {len(y_val)} validation rows, cv={folds}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v0-only", action="store_true", help="stop after the TF-IDF baseline")
    parser.add_argument("--no-cache", action="store_true", help="ignore cached embeddings")
    args = parser.parse_args(argv)

    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.preprocessing import OneHotEncoder

    config.ensure_dirs()

    try:
        frame = load_examples()
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    log.info("examples %d, split counts %s", len(frame), frame["split"].value_counts().to_dict())

    v0_score = train_v0(frame)
    if args.v0_only:
        return 0

    merged = (
        PRODUCT_COLUMN in frame.columns
        and set(frame[PRODUCT_COLUMN].unique()) == {MERGED_LABEL}
    )
    log.info('products merged into one group: %s', merged)

    vectors = embeddings_for(frame, use_cache=not args.no_cache)

    encoder = OneHotEncoder(handle_unknown="ignore")
    matrix = build_matrix(frame, vectors, encoder, fit=True)
    labels = frame["label"].to_numpy()

    train = (frame["split"] == "train").to_numpy()
    val = (frame["split"] == "val").to_numpy()
    test = (frame["split"] == "test").to_numpy()

    candidates: dict[str, object] = {}

    logistic = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    logistic.fit(matrix[train], labels[train])
    candidates["logreg"] = logistic
    log.info("v1 logreg val macro F1: %.4f", f1_score(
        labels[val], logistic.predict(matrix[val]), average="macro", zero_division=0))

    try:
        import lightgbm as lgb

        booster = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, class_weight="balanced", verbose=-1
        )
        booster.fit(matrix[train], labels[train])
        candidates["lightgbm"] = booster
        log.info("v1 lightgbm val macro F1: %.4f", f1_score(
            labels[val], booster.predict(matrix[val]), average="macro", zero_division=0))
    except Exception as exc:  # noqa: BLE001 - lightgbm is optional
        log.warning("lightgbm unavailable, continuing with logistic regression: %s", exc)

    scores = {
        name: f1_score(labels[val], est.predict(matrix[val]), average="macro", zero_division=0)
        for name, est in candidates.items()
    }
    best_name = max(scores, key=scores.get)
    best = candidates[best_name]
    log.info("best on validation: %s (%.4f)", best_name, scores[best_name])

    calibrated, calibration_note = calibrate(best, matrix[val], labels[val])
    log.info("calibration: %s", calibration_note)

    # The test set is touched exactly once, here.
    predicted = calibrated.predict(matrix[test])
    probabilities = calibrated.predict_proba(matrix[test])
    test_f1 = float(f1_score(labels[test], predicted, average="macro", zero_division=0))

    print("\n" + classification_report(labels[test], predicted, zero_division=0))
    print("confusion matrix, labels", list(calibrated.classes_))
    print(confusion_matrix(labels[test], predicted, labels=calibrated.classes_))

    base = baselines(frame, "test")
    print(f"\nTEST macro F1: {test_f1:.4f}")
    for name, value in base.items():
        print(f"  {name} baseline: {value:.4f}")

    rng = np.random.default_rng(0)
    truth = labels[test]
    samples = [
        f1_score(truth[i], predicted[i], average="macro", zero_division=0)
        for i in (rng.integers(0, len(truth), len(truth)) for _ in range(1000))
    ]
    low, high = (float(x) for x in np.percentile(samples, [2.5, 97.5]))

    confidence = probabilities.max(axis=1)
    alert_mask = confidence >= 0.70
    alert_precision = (
        float((predicted[alert_mask] == truth[alert_mask]).mean()) if alert_mask.sum() else None
    )

    joblib.dump(
        {
            "model": calibrated,
            "ohe": encoder,
            "embedder": config.EMBED_MODEL,
            "prefix": config.EMBED_PREFIX,
            "version": f"v1-{best_name}",
            "merged_products": merged,
        },
        config.MODEL_FILE,
    )
    json.dump(
        {str(i): c for i, c in enumerate(calibrated.classes_)},
        config.LABEL_MAP_FILE.open("w", encoding="utf-8"),
    )
    json.dump(
        {"min_confidence": 0.55, "alert_confidence": 0.70},
        config.THRESHOLDS_FILE.open("w", encoding="utf-8"),
        indent=2,
    )
    json.dump(
        {
            "generated_at": pd.Timestamp.now(tz=config.TZ).isoformat(),
            "model": f"v1-{best_name}",
            "merged_products": merged,
            "calibration": calibration_note,
            "v0_tfidf_val_macro_f1": None if np.isnan(v0_score) else v0_score,
            "val_macro_f1": {k: float(v) for k, v in scores.items()},
            "test_macro_f1": test_f1,
            "macro_f1_ci95": [low, high],
            "baselines_test": base,
            "beats_all_baselines": bool(base) and test_f1 > max(base.values()),
            "alert_precision_p70": alert_precision,
            "n_alerts_p70": int(alert_mask.sum()),
            "n_test": int(test.sum()),
            "n_train": int(train.sum()),
            "n_val": int(val.sum()),
            "horizon_days": int(frame["horizon_days"].iloc[0]),
            "note": (
                "Macro F1 on a time split with a purge gap. News is a weak signal "
                "for grocery prices, so a result close to the baselines is the "
                "expected outcome, not a bug."
            ),
        },
        config.METRICS_FILE.open("w", encoding="utf-8"),
        indent=2,
    )

    print(f"\nmacro F1 95% CI: [{low:.4f}, {high:.4f}]")
    print(f"alerts at p>=0.70: {int(alert_mask.sum())}, precision: {alert_precision}")
    print(f"\nwrote model files to {config.MODELS_DIR}")
    print("Next: python scripts/07_evaluate.py for the per product breakdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
