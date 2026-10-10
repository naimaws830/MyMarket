"""Turn model output and price facts into an explanation.

Three modes, chosen by EXPLAIN_MODE in .env:

  host      (default) the tool returns the grounded facts and the chat host
            model writes the prose. No API key, and the server cannot invent a
            number because it never writes a sentence about one.
  llm       call an OpenAI compatible endpoint, then validate against the
            schema. Falls back to the template on any failure.
  template  deterministic sentences built from the facts. Never calls out.

The template always works, so an alert is never blocked on a provider.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from mymarket.config import DISCLAIMER, load_env
from mymarket.llm.prompts import SYSTEM, user_prompt
from mymarket.llm.schema import Explanation, ProductImpact

log = logging.getLogger(__name__)

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

MOVE_WORDS = {
    "UP": "may rise",
    "DOWN": "may fall",
    "STABLE": "looks steady",
    "UNCERTAIN": "is unclear",
    "UNVALIDATED": "is not predicted, no model is trained yet",
}


def mode() -> str:
    """Current explain mode, defaulting to host."""
    load_env()
    value = (os.environ.get("EXPLAIN_MODE") or "host").strip().lower()
    return value if value in {"host", "llm", "template"} else "host"


def facts_for(
    article: dict[str, Any],
    predictions: list[dict[str, Any]],
    price_context: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The only information any explainer is allowed to use.

    Everything here is either copied from the article or computed from the
    price data. Nothing is inferred.
    """
    return {
        "article": {
            "title": article.get("title"),
            "source": article.get("source"),
            "published_at": article.get("published_at"),
            "url": article.get("url"),
            "date_estimated": bool(article.get("date_estimated")),
        },
        "predictions": [
            {
                "product": p.get("product"),
                "direction": p.get("direction"),
                "confidence": p.get("confidence"),
                "probs": p.get("probs"),
                "abstained": p.get("abstained"),
                "triggers": p.get("triggers"),
            }
            for p in predictions
        ],
        "price_context": price_context or {},
        "horizon_days": article.get("horizon_days"),
        "rules": [
            "Direction is a 14 day price direction, not a price level forecast.",
            "The news signal for grocery prices is weak. Treat this as awareness only.",
            "Price data is PriceCatcher, a KPDN price survey, not an inflation statistic.",
        ],
    }


def template(facts: dict[str, Any]) -> Explanation:
    """Deterministic explanation built only from the facts. Never fails."""
    article = facts.get("article", {})
    predictions = facts.get("predictions", [])
    context = facts.get("price_context", {})

    impacts: list[ProductImpact] = []
    for pred in predictions:
        product = str(pred.get("product", "unknown"))
        direction = str(pred.get("direction", "UNCERTAIN"))
        triggers = pred.get("triggers") or []

        reason = f"The article mentions {product.replace('_', ' ')}"
        if triggers:
            reason += f" alongside {', '.join(triggers[:3])}"
        reason += f", so the model says the price {MOVE_WORDS.get(direction, 'is unclear')}"

        recent = context.get(product) or {}
        change = recent.get("change_7d_pct")
        if change is not None:
            reason += f", and the index moved {change} percent over the last 7 days"
        impacts.append(ProductImpact(product=product, direction=direction, why=reason + "."))

    moving = [i.product.replace("_", " ") for i in impacts if i.direction in {"UP", "DOWN"}]
    if moving:
        domain = (
            f"This story is linked to {', '.join(moving)}. "
            "The model sees a possible price move over the next 14 days."
        )
    else:
        domain = (
            "This story touches tracked food staples but the model sees no clear "
            "price move over the next 14 days."
        )

    confidences = [p.get("confidence") or 0 for p in predictions]
    best = max(confidences) if confidences else 0
    note = (
        f"Highest model confidence is {round(float(best), 2)}. "
        "News is a weak signal for grocery prices, so treat this as awareness only."
    )

    title = str(article.get("title") or "Food staple price signal")
    return Explanation(
        headline=" ".join(title.split()[:12]),
        domain_impact=domain,
        product_impacts=impacts,
        confidence_note=note,
        disclaimer=DISCLAIMER,
    )


def _parse(raw: str) -> Explanation:
    """Validate an LLM reply against the schema."""
    match = JSON_BLOCK.search(raw or "")
    if not match:
        raise ValueError("no JSON object in the reply")
    data = json.loads(match.group(0))
    data["disclaimer"] = DISCLAIMER  # never let the model rewrite this
    return Explanation.model_validate(data)


def _call_llm(facts: dict[str, Any], lang: str) -> str:
    """One call to an OpenAI compatible chat completions endpoint."""
    import httpx

    load_env()
    base = (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")
    key = (os.environ.get("LLM_API_KEY") or "").strip()
    model_name = (os.environ.get("LLM_MODEL") or "").strip()

    if not base or not model_name:
        raise RuntimeError("LLM_BASE_URL and LLM_MODEL must be set for EXPLAIN_MODE=llm")

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(facts, lang)},
        ],
        "temperature": 0.2,
        "max_tokens": 700,
    }

    with httpx.Client(timeout=45.0) as client:
        resp = client.post(f"{base}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    return data["choices"][0]["message"]["content"]


def explain(
    article: dict[str, Any],
    predictions: list[dict[str, Any]],
    price_context: dict[str, dict[str, Any]] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Produce the explanation block for one alert.

    In host mode the grounded facts are returned alongside a template
    explanation, so a chat host can rewrite them and a plain HTTP caller still
    gets usable sentences.
    """
    facts = facts_for(article, predictions, price_context)
    current = mode()
    fallback = template(facts)

    if current == "llm":
        for attempt in (1, 2):
            try:
                parsed = _parse(_call_llm(facts, lang))
                return {
                    **parsed.model_dump(),
                    "explained_by": "llm",
                    "facts": facts,
                }
            except Exception as exc:  # noqa: BLE001 - never fail an alert on the LLM
                log.warning("LLM explanation attempt %d failed: %s", attempt, exc)
        log.warning("falling back to the template explanation")
        return {**fallback.model_dump(), "explained_by": "template_after_llm_failure", "facts": facts}

    if current == "template":
        return {**fallback.model_dump(), "explained_by": "template", "facts": facts}

    return {
        **fallback.model_dump(),
        "explained_by": "host",
        "facts": facts,
        "host_instructions": (
            "These sentences are a deterministic fallback. Rewrite them for the "
            "reader using only the values in `facts`. Do not introduce any number "
            "that is not already there, and keep the disclaimer."
        ),
    }
