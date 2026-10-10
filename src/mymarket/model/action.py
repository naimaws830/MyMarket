"""Turn a prediction into a prioritised, actionable recommendation.

A direction and a probability is not a decision. The user this is built for is
a small food business buyer, a kopitiam or caterer owner, or a household doing
a monthly shop. Their decision is narrow and concrete: buy now, wait, or do
nothing, and how much to stock.

This module answers three things a bare prediction does not:

  how urgent is it   an explicit score and level, so a list of alerts can be
                     ranked instead of read in feed order
  why now            the specific reasons that produced the score
  what to do         one concrete next step for that product and direction

Nothing here is financial advice, and the horizons are grocery buying
horizons, not investment ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mymarket import config

# Score bands. Chosen so that ACT NOW requires both a confident call and some
# corroboration from the price data, not confidence alone: the model has been
# confidently wrong before and the bands should not hide that.
ACT_NOW = 70
WATCH = 45
MONITOR = 25

# Component ceilings. They sum to 100.
W_CONFIDENCE = 40  # how sure the model is
W_DIRECTION = 20  # a rise costs the buyer money, a fall merely saves it
W_CORROBORATION = 25  # is the official price already moving the same way
W_RECENCY = 15  # a three week old story is not an alert

# Roughly how long a wholesale or policy move takes to reach a shelf price.
# Used only to phrase the advice, never to compute a number.
PASS_THROUGH_DAYS = {
    "rice": "2 to 4 weeks, and faster if there is panic buying",
    "egg": "1 to 3 weeks, eggs move quickly",
    "cooking_oil": "2 to 6 weeks, slower because much of it is subsidised",
    "chicken": "1 to 2 weeks, chicken is the fastest moving of the four",
}

# What a buyer can actually do. Keyed by (direction, how urgent).
ACTIONS = {
    "UP": {
        "high": (
            "Buy your next {product} stock now, before the increase reaches the shelf. "
            "If you can store it, bring forward roughly one extra cycle of purchasing."
        ),
        "low": (
            "Check {product} prices at your usual supplier this week and compare against "
            "the index below. Consider buying a little earlier than usual."
        ),
    },
    "DOWN": {
        "high": (
            "Hold off on buying {product} beyond what you need now. Prices look likely "
            "to ease, so a large purchase today probably costs more than next month's."
        ),
        "low": (
            "No rush on {product}. If you were about to stock up heavily, waiting a week "
            "or two may be slightly cheaper."
        ),
    },
    "STABLE": {
        "high": "Buy {product} as normal. Nothing in the news points to a change.",
        "low": "Buy {product} as normal.",
    },
}

UNCLEAR_ACTION = (
    "No action. The model is not confident enough about {product} to call a "
    "direction, so treat this as background reading only."
)


@dataclass(frozen=True)
class Recommendation:
    """A ranked, actionable alert for one product."""

    product: str
    urgency: int
    priority: str
    action: str
    why_now: list[str]
    pass_through: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "urgency_score": self.urgency,
            "priority": self.priority,
            "recommended_action": self.action,
            "why_now": self.why_now,
            "typical_pass_through": self.pass_through,
        }


def _level(score: int) -> str:
    if score >= ACT_NOW:
        return "ACT NOW"
    if score >= WATCH:
        return "WATCH"
    if score >= MONITOR:
        return "MONITOR"
    return "NO ACTION"


def _hours_since(published_at: str | None) -> float | None:
    """Age of the article in hours, or None when the date is unusable."""
    if not published_at:
        return None
    try:
        when = pd.Timestamp(published_at)
    except (ValueError, TypeError):
        return None
    if when.tzinfo is None:
        when = when.tz_localize(config.TZ)
    return max(0.0, (pd.Timestamp.now(tz=config.TZ) - when).total_seconds() / 3600)


def score(
    direction: str,
    confidence: float | None,
    published_at: str | None = None,
    price_context: dict[str, Any] | None = None,
) -> tuple[int, list[str]]:
    """Urgency from 0 to 100, with the reasons that produced it.

    The reasons are returned because an unexplained priority is just another
    opaque number, and the whole point of this layer is that a user can see
    why something was put at the top of their list.
    """
    reasons: list[str] = []
    total = 0.0

    if direction in {"UNCERTAIN", "UNVALIDATED"}:
        return 0, ["the model did not make a confident call"]

    # 1. Model confidence.
    if confidence is not None:
        total += W_CONFIDENCE * float(confidence)
        if confidence >= 0.70:
            reasons.append(f"model is confident ({confidence:.0%}), which tested at 70 percent correct")
        elif confidence >= 0.55:
            reasons.append(f"model is moderately confident ({confidence:.0%})")
        else:
            reasons.append(f"model confidence is low ({confidence:.0%})")

    # 2. Does the direction cost the buyer money.
    if direction == "UP":
        total += W_DIRECTION
        reasons.append("a rise costs you money, so it is worth acting on early")
    elif direction == "DOWN":
        total += W_DIRECTION * 0.5
        reasons.append("a fall saves you money, which is worth knowing but rarely urgent")

    # 3. Is the official price already moving the same way.
    change = (price_context or {}).get("change_7d_pct")
    if change is not None and direction in {"UP", "DOWN"}:
        agrees = (change > 0 and direction == "UP") or (change < 0 and direction == "DOWN")
        if agrees and abs(change) >= 0.5:
            total += W_CORROBORATION
            reasons.append(
                f"official prices are already moving this way, {change:+.1f} percent over 7 days"
            )
        elif agrees:
            total += W_CORROBORATION * 0.4
            reasons.append(f"official prices agree but have barely moved, {change:+.1f} percent over 7 days")
        else:
            reasons.append(
                f"official prices have moved the other way so far, {change:+.1f} percent over 7 days, "
                "which is a reason for caution"
            )

    # 4. Freshness.
    age = _hours_since(published_at)
    if age is None:
        total += W_RECENCY * 0.5
        reasons.append("publication date is estimated")
    elif age <= 24:
        total += W_RECENCY
        reasons.append("published in the last 24 hours")
    elif age <= 72:
        total += W_RECENCY * 0.6
        reasons.append(f"published about {int(age / 24)} days ago")
    else:
        reasons.append(f"published {int(age / 24)} days ago, so any move may already have happened")

    # Caps. Without these the free components (direction, corroboration,
    # recency) can push a coin flip call to the top of the list, which is
    # exactly the false alarm this layer is supposed to prevent.
    #
    # ACT NOW is reserved for the confidence band that actually tested at 70
    # percent correct. Anything less can reach WATCH at most.
    if confidence is None or confidence < 0.70:
        if total >= ACT_NOW:
            total = ACT_NOW - 1
            reasons.append("capped below ACT NOW because model confidence is under 70 percent")
    if confidence is not None and confidence < 0.55 and total >= WATCH:
        total = WATCH - 1
        reasons.append("capped below WATCH because model confidence is in the low band")

    # "Carry on as normal" is never urgent, whatever else is true.
    if direction == "STABLE" and total >= MONITOR:
        total = MONITOR - 1
        reasons.append("no action needed, the model expects no change")

    return int(round(min(100.0, max(0.0, total)))), reasons


def recommend(
    product: str,
    direction: str,
    confidence: float | None,
    published_at: str | None = None,
    price_context: dict[str, Any] | None = None,
) -> Recommendation:
    """Full recommendation for one (product, prediction)."""
    urgency, reasons = score(direction, confidence, published_at, price_context)
    level = _level(urgency)
    label = product.replace("_", " ")

    if direction in {"UNCERTAIN", "UNVALIDATED"}:
        action = UNCLEAR_ACTION.format(product=label)
    else:
        strength = "high" if urgency >= WATCH else "low"
        action = ACTIONS.get(direction, ACTIONS["STABLE"])[strength].format(product=label)

    return Recommendation(
        product=product,
        urgency=urgency,
        priority=level,
        action=action,
        why_now=reasons,
        pass_through=PASS_THROUGH_DAYS.get(product, "typically a few weeks"),
    )


def rank(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort alerts so the most urgent is first."""
    return sorted(alerts, key=lambda a: a.get("max_urgency", 0), reverse=True)
