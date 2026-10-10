"""Presentation layer: turn a predicted price direction into reader facing words.

The model predicts a price direction (UP, DOWN, STABLE) learned from
PriceCatcher returns. It does not classify news sentiment. These labels are a
presentation of that prediction, chosen so a reader sees the consequence rather
than the mechanism.

The mapping is: cheaper staples are good news, dearer staples are bad news.

    DOWN   -> POSITIVELY AFFECTED   prices likely to ease
    UP     -> NEGATIVELY AFFECTED   prices likely to rise
    STABLE -> NEUTRAL               no clear change

PRD flag F1 warned that "positive" is ambiguous, because a price rise is good
for a seller and bad for a shopper. Every label here therefore ships with an
explicit `price_direction` and a plain `meaning`, so the word is never alone.
"""

from __future__ import annotations

from typing import Any

# Confidence bands, cut where measured accuracy actually changes on the test
# set: 0.72 above 0.8, about 0.5 from 0.7 to 0.8, and 0.25 to 0.43 below that.
HIGH_CONFIDENCE = 0.70
MEDIUM_CONFIDENCE = 0.55

IMPACT = {
    "DOWN": {
        "impact": "POSITIVELY AFFECTED",
        "meaning": "prices look likely to ease, which is good for shoppers",
        "price_direction": "DOWN",
    },
    "UP": {
        "impact": "NEGATIVELY AFFECTED",
        "meaning": "prices look likely to rise, which is bad for shoppers",
        "price_direction": "UP",
    },
    "STABLE": {
        "impact": "NEUTRAL",
        "meaning": "no clear price change expected",
        "price_direction": "STABLE",
    },
    "UNCERTAIN": {
        "impact": "UNCLEAR",
        "meaning": "the model is not confident enough to call a direction",
        "price_direction": "UNCERTAIN",
    },
    "UNVALIDATED": {
        "impact": "NOT ASSESSED",
        "meaning": "no trained model, this is a keyword and price context match only",
        "price_direction": "UNVALIDATED",
    },
}


def band(confidence: float | None) -> str:
    """Plain confidence band. Low means the call should not be acted on."""
    if confidence is None:
        return "none"
    if confidence >= HIGH_CONFIDENCE:
        return "high"
    if confidence >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


def band_note(confidence: float | None) -> str:
    """What the band is worth, in measured terms rather than adjectives."""
    return {
        "high": "about 70 percent of calls at this confidence were right in testing",
        "medium": "roughly a coin flip in testing, read it as a hint only",
        "low": "below the useful threshold, treat as no signal",
        "none": "no confidence available",
    }[band(confidence)]


def describe(direction: str, confidence: float | None = None) -> dict[str, Any]:
    """Full reader facing block for one predicted direction."""
    base = dict(IMPACT.get(direction, IMPACT["UNCERTAIN"]))
    base["confidence"] = None if confidence is None else round(float(confidence), 3)
    base["confidence_band"] = band(confidence)
    base["confidence_note"] = band_note(confidence)
    return base
