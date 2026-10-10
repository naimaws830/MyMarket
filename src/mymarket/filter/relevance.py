"""The two stage relevance filter the live pipeline and the MCP tools call."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from mymarket import config
from mymarket.filter import embed_filter
from mymarket.filter.aliases import build_text, find_terms, match_all

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Match:
    """One article matched to one product, with every filter stage scored."""

    product: str
    matched_terms: list[str]
    event_words: list[str]
    embed_score: float | None
    passed_embedding: bool
    price_cues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def price_cues_in(title: str, summary: str = "") -> list[str]:
    """Price cue words in the headline or summary.

    The body is deliberately not searched. A restaurant review will mention
    cost somewhere in paragraph nine; an article that is actually about a price
    says so in the headline.
    """
    cfg = config.settings()
    if not cfg.price_cue_words:
        return []
    return find_terms(f"{title or ''}. {summary or ''}", cfg.price_cue_words)


def match_article(
    title: str,
    summary: str = "",
    body: str = "",
    threshold: float | None = None,
    use_embeddings: bool = True,
    require_price_cue: bool | None = None,
) -> list[Match]:
    """Alias match, then embedding check, returning only the products that pass.

    When the embedding model cannot be loaded the alias result is returned with
    `embed_score` None, so the pipeline degrades to stage one rather than
    returning nothing. That is logged, never hidden.
    """
    hits = match_all(title, summary, body)
    if not hits:
        return []

    cfg = config.settings()
    needs_cue = cfg.requires_price_cue if require_price_cue is None else require_price_cue
    cues = price_cues_in(title, summary)
    if needs_cue and not cues:
        return []

    limit = cfg.embed_threshold if threshold is None else threshold

    if not use_embeddings or not embed_filter.available():
        if use_embeddings:
            log.warning("embedding stage skipped, alias matches returned unfiltered")
        return [
            Match(h.product, list(h.matched_terms), list(h.event_words), None, True, cues)
            for h in hits
        ]

    text = build_text(title, summary, body)
    scores = embed_filter.score_many(text, [h.product for h in hits])

    out: list[Match] = []
    for hit in hits:
        value = scores.get(hit.product, 0.0)
        out.append(
            Match(
                product=hit.product,
                matched_terms=list(hit.matched_terms),
                event_words=list(hit.event_words),
                embed_score=round(value, 4),
                passed_embedding=value >= limit,
                price_cues=cues,
            )
        )
    return [m for m in out if m.passed_embedding]


def match_many(
    articles: Sequence[dict[str, Any]],
    threshold: float | None = None,
    use_embeddings: bool = True,
    require_price_cue: bool | None = None,
) -> list[tuple[dict[str, Any], list[Match]]]:
    """Run the filter over a list of articles, keeping only those that matched."""
    out = []
    for article in articles:
        matches = match_article(
            article.get("title", ""),
            article.get("summary", ""),
            article.get("body", ""),
            threshold=threshold,
            use_embeddings=use_embeddings,
            require_price_cue=require_price_cue,
        )
        if matches:
            out.append((article, matches))
    return out
