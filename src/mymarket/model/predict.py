"""Load the trained model and predict price direction for an article.

Absence of a model is a normal state here, not a crash: the repo ships without
one because training needs the Kaggle news dump. Every entry point returns a
ModelMissing error that the MCP layer turns into the documented error shape.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from mymarket import config
from mymarket.filter.aliases import find_terms
from mymarket.model.features import (
    MERGED_LABEL,
    PRODUCT_COLUMN,
    build_matrix,
    model_text,
)

log = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {"min_confidence": 0.55, "alert_confidence": 0.70}


class ModelMissing(RuntimeError):
    """No trained model is available yet."""


@dataclass
class Prediction:
    """One model call for one (article, product) pair."""

    product: str
    direction: str
    probs: dict[str, float]
    confidence: float
    abstained: bool
    triggers: list[str] = field(default_factory=list)
    model_version: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        from mymarket.model.impact import describe

        reader = describe(self.direction, self.confidence)
        return {
            "product": self.product,
            # Reader facing. See model/impact.py for why the word never travels
            # alone: "positive" is ambiguous without the price direction.
            "impact": reader["impact"],
            "meaning": reader["meaning"],
            "confidence_band": reader["confidence_band"],
            "confidence_note": reader["confidence_note"],
            # Machine facing. Unchanged, and what every metric is computed on.
            "direction": self.direction,
            "probs": self.probs,
            "confidence": self.confidence,
            "abstained": self.abstained,
            "triggers": self.triggers,
            "model_version": self.model_version,
            "shopper_note": shopper_note(self.direction),
        }


def shopper_note(direction: str) -> str:
    """Plain tag for who a direction is bad news for, per PRD F1."""
    return {
        "UP": "bad for shoppers, prices may rise",
        "DOWN": "good for shoppers, prices may fall",
        "STABLE": "no clear move expected",
        "UNVALIDATED": "no trained model, keyword match only",
    }.get(direction, "no clear move expected")


def thresholds() -> dict[str, float]:
    """Thresholds saved beside the model, falling back to the defaults."""
    if not config.THRESHOLDS_FILE.exists():
        return dict(DEFAULT_THRESHOLDS)
    try:
        saved = json.loads(config.THRESHOLDS_FILE.read_text(encoding="utf-8"))
        return {**DEFAULT_THRESHOLDS, **saved}
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read thresholds.json: %s", exc)
        return dict(DEFAULT_THRESHOLDS)


def available() -> bool:
    """Whether a trained model file is present."""
    return config.MODEL_FILE.exists()


@lru_cache(maxsize=1)
def _bundle() -> dict[str, Any]:
    """Load the joblib bundle once per process."""
    if not available():
        raise ModelMissing(
            f"{config.MODEL_FILE.name} not found in models/. Build examples with "
            "scripts/05_build_examples.py then train with scripts/06_train_local.py "
            "or the Colab notebook (manual step M6)."
        )
    import joblib

    bundle = joblib.load(config.MODEL_FILE)
    missing = {"model", "ohe"} - set(bundle)
    if missing:
        raise ModelMissing(f"model bundle is missing keys {sorted(missing)}")
    return bundle


def trained_horizon_days() -> int | None:
    """Horizon the shipped model was trained on, or None when unknown.

    The config default can drift from the trained model. The model is the
    authority, because it is the thing making the prediction.
    """
    try:
        value = _bundle().get("horizon_days")
        return int(value) if value else None
    except ModelMissing:
        return None


def model_version() -> str:
    """Version string recorded in the bundle, or unknown."""
    try:
        return str(_bundle().get("version", "unknown"))
    except ModelMissing:
        return "none"


def _embed(texts: list[str]):
    from mymarket.filter import embed_filter

    return embed_filter.encode(texts)


def triggers_for(text: str, limit: int = 5) -> list[str]:
    """Event words present in the article, used to explain a prediction."""
    return find_terms(text, config.settings().event_words)[:limit]


def predict(
    title: str,
    summary: str = "",
    body: str = "",
    products: list[str] | None = None,
    min_confidence: float | None = None,
) -> list[Prediction]:
    """Predict a direction per product for one article.

    Raises ModelMissing when no model has been trained. The caller decides how
    to present that.
    """
    import pandas as pd

    bundle = _bundle()
    cfg = config.settings()
    wanted = products or cfg.product_keys
    floor = thresholds()["min_confidence"] if min_confidence is None else min_confidence

    text = model_text(title, summary, body)
    frame = pd.DataFrame({"product": wanted, "text": [text] * len(wanted)})
    # Serving features must be built the same way training built them.
    frame[PRODUCT_COLUMN] = MERGED_LABEL if bundle.get("merged_products") else frame["product"]

    embeddings = _embed([text] * len(wanted))
    matrix = build_matrix(frame, embeddings, bundle["ohe"], fit=False)

    estimator = bundle["model"]
    proba = estimator.predict_proba(matrix)
    classes = [str(c) for c in estimator.classes_]
    found = triggers_for(text)
    version = str(bundle.get("version", "unknown"))

    out: list[Prediction] = []
    for row, product in zip(proba, wanted):
        probs = {cls: round(float(p), 4) for cls, p in zip(classes, row)}
        best = max(probs, key=probs.get)
        confidence = probs[best]
        abstained = confidence < floor
        out.append(
            Prediction(
                product=product,
                direction="UNCERTAIN" if abstained else best,
                probs=probs,
                confidence=round(confidence, 4),
                abstained=abstained,
                triggers=found,
                model_version=version,
            )
        )
    return out
