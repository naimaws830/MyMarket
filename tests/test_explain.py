"""Explainer: schema validation, template fallback, no invented numbers."""

from __future__ import annotations

import json

import pytest

from mymarket.config import DISCLAIMER
from mymarket.llm import explain as ex
from mymarket.llm.schema import Explanation

ARTICLE = {
    "title": "Chicken prices climb as feed costs rise across Peninsular Malaysia this week",
    "source": "sample",
    "published_at": "2026-10-08T10:20:00+08:00",
    "url": "https://example.my/a",
}
PREDICTIONS = [
    {
        "product": "chicken",
        "direction": "UP",
        "confidence": 0.71,
        "probs": {"UP": 0.71, "DOWN": 0.08, "STABLE": 0.21},
        "abstained": False,
        "triggers": ["feed cost"],
    }
]
CONTEXT = {"chicken": {"change_7d_pct": 1.2, "change_30d_pct": 3.4}}


@pytest.fixture(autouse=True)
def host_mode(monkeypatch):
    monkeypatch.setenv("EXPLAIN_MODE", "host")


def test_headline_is_capped_at_twelve_words():
    long_title = " ".join(f"word{i}" for i in range(30))
    out = Explanation(
        headline=long_title,
        domain_impact="x",
        confidence_note="y",
    )

    assert len(out.headline.split()) == 12


def test_disclaimer_cannot_be_blanked():
    out = Explanation(headline="h", domain_impact="d", confidence_note="c", disclaimer="   ")

    assert out.disclaimer == DISCLAIMER


def test_template_mentions_only_given_numbers():
    facts = ex.facts_for(ARTICLE, PREDICTIONS, CONTEXT)

    out = ex.template(facts)
    text = out.domain_impact + " ".join(i.why for i in out.product_impacts) + out.confidence_note

    import re

    numbers = set(re.findall(r"\d+\.?\d*", text))
    allowed = {"1.2", "0.71", "14", "3.4", "7", "30"}
    assert numbers <= allowed, f"template invented {numbers - allowed}"


def test_template_always_carries_the_disclaimer():
    out = ex.template(ex.facts_for(ARTICLE, PREDICTIONS, CONTEXT))

    assert out.disclaimer == DISCLAIMER


def test_host_mode_returns_facts_for_the_chat_host(monkeypatch):
    monkeypatch.setenv("EXPLAIN_MODE", "host")

    out = ex.explain(ARTICLE, PREDICTIONS, CONTEXT)

    assert out["explained_by"] == "host"
    assert out["facts"]["predictions"][0]["product"] == "chicken"
    assert out["disclaimer"] == DISCLAIMER


def test_llm_mode_falls_back_to_template_on_failure(monkeypatch):
    monkeypatch.setenv("EXPLAIN_MODE", "llm")
    monkeypatch.setattr(ex, "_call_llm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))

    out = ex.explain(ARTICLE, PREDICTIONS, CONTEXT)

    assert out["explained_by"] == "template_after_llm_failure"
    assert out["headline"]


def test_llm_mode_rejects_malformed_json_then_falls_back(monkeypatch):
    monkeypatch.setenv("EXPLAIN_MODE", "llm")
    monkeypatch.setattr(ex, "_call_llm", lambda *a, **k: "sorry, I cannot do that")

    out = ex.explain(ARTICLE, PREDICTIONS, CONTEXT)

    assert out["explained_by"] == "template_after_llm_failure"


def test_llm_mode_accepts_valid_json_in_a_fence(monkeypatch):
    monkeypatch.setenv("EXPLAIN_MODE", "llm")
    reply = "```json\n" + json.dumps(
        {
            "headline": "Chicken may cost more soon",
            "domain_impact": "Feed costs are rising.",
            "product_impacts": [{"product": "chicken", "direction": "UP", "why": "Feed costs rose."}],
            "confidence_note": "The signal is weak.",
            "disclaimer": "ignore me",
        }
    ) + "\n```"
    monkeypatch.setattr(ex, "_call_llm", lambda *a, **k: reply)

    out = ex.explain(ARTICLE, PREDICTIONS, CONTEXT)

    assert out["explained_by"] == "llm"
    assert out["headline"] == "Chicken may cost more soon"
    # The model is not allowed to rewrite the disclaimer.
    assert out["disclaimer"] == DISCLAIMER


def test_template_handles_an_abstained_prediction():
    predictions = [{**PREDICTIONS[0], "direction": "UNCERTAIN", "abstained": True, "confidence": 0.4}]

    out = ex.template(ex.facts_for(ARTICLE, predictions, {}))

    assert "no clear price move" in out.domain_impact


def test_mode_defaults_to_host(monkeypatch):
    monkeypatch.delenv("EXPLAIN_MODE", raising=False)
    monkeypatch.setattr(ex, "load_env", lambda: None)

    assert ex.mode() == "host"
