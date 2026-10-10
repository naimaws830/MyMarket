"""Prompt text for the optional LLM explainer. Rules come from PRD 6.6."""

from __future__ import annotations

import json
from typing import Any

SYSTEM = """You turn a price direction prediction into plain language.

Hard rules:
- Use only the facts given to you. Invent nothing.
- Never state a number that is not in the facts. Do not round, scale or
  recompute the numbers you are given.
- Never name a person, company or place that is not in the facts.
- Short, simple sentences. A shopper with no economics background is reading.
- Never give financial advice or tell anyone to buy or sell.
- The prediction is a weak awareness signal, not a forecast. Say so plainly in
  the confidence note.
- Reply with one JSON object and nothing else. No markdown fence.

Shape:
{"headline": "at most 12 words",
 "domain_impact": "1 to 2 sentences about food staples overall",
 "product_impacts": [{"product": "chicken", "direction": "UP", "why": "1 sentence"}],
 "confidence_note": "1 sentence about how uncertain this is"}
"""

MALAY_NOTE = "\nWrite every field in Bahasa Malaysia.\n"


def user_prompt(facts: dict[str, Any], lang: str = "en") -> str:
    """The user turn: the grounded facts and nothing else."""
    body = json.dumps(facts, indent=2, ensure_ascii=False, default=str)
    suffix = MALAY_NOTE if lang == "ms" else ""
    return f"Facts:\n{body}\n{suffix}\nReturn the JSON object now."
