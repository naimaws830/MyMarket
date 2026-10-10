"""Urgency scoring and recommendations: the decision support layer."""

from __future__ import annotations

import pandas as pd
import pytest

from mymarket.model import action

FRESH = (pd.Timestamp.now(tz="Asia/Kuala_Lumpur") - pd.Timedelta(hours=3)).isoformat()
STALE = (pd.Timestamp.now(tz="Asia/Kuala_Lumpur") - pd.Timedelta(days=12)).isoformat()
RISING = {"change_7d_pct": 2.1}
FALLING = {"change_7d_pct": -2.1}


def urgency(direction, confidence, published=FRESH, context=RISING):
    return action.score(direction, confidence, published, context)[0]


def test_act_now_requires_high_confidence():
    """ACT NOW is reserved for the band that tested at 70 percent correct."""
    assert action.recommend("rice", "UP", 0.85, FRESH, RISING).priority == "ACT NOW"
    assert action.recommend("rice", "UP", 0.65, FRESH, RISING).priority != "ACT NOW"


def test_low_confidence_cannot_reach_watch():
    assert urgency("UP", 0.45) < action.WATCH


def test_stable_is_never_urgent():
    """Carry on as normal is not an alert, however confident the model is."""
    rec = action.recommend("rice", "STABLE", 0.99, FRESH, RISING)

    assert rec.priority == "NO ACTION"
    assert "as normal" in rec.action


def test_abstention_scores_zero():
    score, reasons = action.score("UNCERTAIN", 0.4, FRESH, RISING)

    assert score == 0
    assert "did not make a confident call" in reasons[0]


def test_a_rise_outranks_an_equally_confident_fall():
    """A price rise costs the buyer money; a fall merely saves it."""
    assert urgency("UP", 0.8) > urgency("DOWN", 0.8, context=FALLING)


def test_corroborating_price_move_raises_urgency():
    agreeing = urgency("UP", 0.8, context=RISING)
    contradicting = urgency("UP", 0.8, context=FALLING)

    assert agreeing > contradicting


def test_contradicting_price_move_is_flagged_as_caution():
    _, reasons = action.score("UP", 0.8, FRESH, FALLING)

    assert any("other way" in r for r in reasons)


def test_stale_news_is_less_urgent_than_fresh():
    assert urgency("UP", 0.8, published=FRESH) > urgency("UP", 0.8, published=STALE)


def test_missing_date_does_not_crash():
    assert urgency("UP", 0.8, published=None) > 0
    assert urgency("UP", 0.8, published="not a date") > 0


def test_every_recommendation_names_a_concrete_step():
    for direction in ("UP", "DOWN", "STABLE", "UNCERTAIN"):
        rec = action.recommend("cooking_oil", direction, 0.8, FRESH, RISING)
        assert "cooking oil" in rec.action
        assert rec.why_now, "an unexplained priority is just another opaque number"
        assert rec.to_dict()["typical_pass_through"]


def test_rank_puts_the_most_urgent_first():
    alerts = [{"max_urgency": 10}, {"max_urgency": 90}, {"max_urgency": 50}]

    assert [a["max_urgency"] for a in action.rank(alerts)] == [90, 50, 10]


@pytest.mark.parametrize("score,expected", [(95, "ACT NOW"), (50, "WATCH"), (30, "MONITOR"), (5, "NO ACTION")])
def test_level_boundaries(score, expected):
    assert action._level(score) == expected
