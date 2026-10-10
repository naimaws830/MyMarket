"""Stage one relevance filter: word boundary alias match with exclude phrases.

PRD 6.2 stage 1. Runs on title + summary + the first N chars of body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from mymarket.config import Product, settings


@dataclass(frozen=True)
class AliasHit:
    """One (article, product) alias match."""

    product: str
    matched_terms: tuple[str, ...]
    event_words: tuple[str, ...]
    alias_count: int


def _boundary(term: str) -> re.Pattern[str]:
    """Compile a case-insensitive whole-phrase pattern for one term.

    Word boundaries are applied only where the phrase edge is a word character,
    so a term such as "11.11" or "SST" still matches cleanly.
    """
    escaped = re.escape(term.strip())
    left = r"\b" if term[:1].isalnum() else ""
    right = r"\b" if term[-1:].isalnum() else ""
    # Collapse escaped runs of whitespace so "cooking  oil" still matches.
    escaped = re.sub(r"(\\\s)+", r"\\s+", escaped)
    return re.compile(f"{left}{escaped}{right}", re.IGNORECASE)


@lru_cache(maxsize=4096)
def _pattern(term: str) -> re.Pattern[str]:
    return _boundary(term)


def build_text(title: str, summary: str, body: str, body_chars: int | None = None) -> str:
    """Join the fields the alias stage reads, per PRD 6.2."""
    limit = settings().body_chars if body_chars is None else body_chars
    parts = [
        (title or "").strip(),
        (summary or "").strip(),
        (body or "")[:limit].strip(),
    ]
    return ". ".join(p for p in parts if p)


def find_terms(text: str, terms: list[str]) -> list[str]:
    """Terms from the list that appear in the text, in list order, deduped."""
    seen: list[str] = []
    for term in terms:
        if not term:
            continue
        if _pattern(term).search(text) and term not in seen:
            seen.append(term)
    return seen


def _mask_excluded(text: str, exclude_phrases: list[str]) -> str:
    """Blank out every exclude phrase so its words cannot satisfy an alias.

    Masking rather than rejecting the whole article matters: an article can say
    "chicken rice stalls raise prices as chicken supply tightens". Dropping it
    on the phrase "chicken rice" alone would lose a real signal, so only the
    offending span is removed before alias matching.
    """
    for phrase in exclude_phrases:
        if not phrase:
            continue
        text = _pattern(phrase).sub(" ", text)
    return text


def match_product(text: str, product: Product, event_words: list[str]) -> AliasHit | None:
    """Alias match one product against prepared text, or None when it does not hit."""
    cleaned = _mask_excluded(text, product.exclude_phrases)
    hits = find_terms(cleaned, product.aliases)
    if not hits:
        return None

    alias_count = sum(len(_pattern(term).findall(cleaned)) for term in hits)
    return AliasHit(
        product=product.key,
        matched_terms=tuple(hits),
        event_words=tuple(find_terms(cleaned, event_words)),
        alias_count=alias_count,
    )


def match_all(title: str, summary: str = "", body: str = "") -> list[AliasHit]:
    """Alias match every configured product against one article."""
    cfg = settings()
    text = build_text(title, summary, body)
    hits = [match_product(text, p, cfg.event_words) for p in cfg.products.values()]
    return [h for h in hits if h is not None]


def event_flags(title: str, summary: str = "", body: str = "") -> dict[str, int]:
    """Binary event word features for the model, per PRD 6.3 and F5."""
    cfg = settings()
    text = build_text(title, summary, body)
    present = set(find_terms(text, cfg.event_words))
    return {f"ev_{w.lower().replace(' ', '_')}": int(w in present) for w in cfg.event_words}
