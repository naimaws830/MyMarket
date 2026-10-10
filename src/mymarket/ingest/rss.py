"""Read the configured RSS feeds into normalised article dictionaries.

Feeds differ in what they carry. Malay Mail puts the full article HTML in
content:encoded, Bernama carries a short summary and no pubDate. Both cases are
handled here so the rest of the pipeline sees one shape (PRD 6.1).
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterable

import feedparser
import pandas as pd

from mymarket import config

log = logging.getLogger(__name__)

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

ARTICLE_KEYS = (
    "id",
    "url",
    "title",
    "source",
    "published_at",
    "summary",
    "body",
    "lang",
    "date_estimated",
    "fetched_at",
)


def strip_html(value: str | None) -> str:
    """Plain text from an HTML fragment."""
    if not value:
        return ""
    text = TAG_RE.sub(" ", value)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return WS_RE.sub(" ", text).strip()


def article_id(url: str) -> str:
    """Stable id for an article URL."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def now_kl() -> pd.Timestamp:
    """Current time in the project timezone."""
    return pd.Timestamp.now(tz=config.TZ)


def _parsed_time(entry: Any) -> pd.Timestamp | None:
    """Published time from a feed entry, converted to the project timezone."""
    for key in ("published_parsed", "updated_parsed"):
        value = getattr(entry, key, None) or entry.get(key) if hasattr(entry, "get") else None
        if value:
            try:
                stamp = datetime(*value[:6], tzinfo=timezone.utc)
                return pd.Timestamp(stamp).tz_convert(config.TZ)
            except (TypeError, ValueError):
                continue
    return None


def _full_content(entry: Any) -> str:
    """Longest content:encoded block on the entry, as plain text."""
    blocks = entry.get("content") or []
    texts = [strip_html(block.get("value", "")) for block in blocks]
    return max(texts, key=len) if texts else ""


def _clean_title(title: str) -> str:
    """Drop the category prefix Bernama puts on titles, such as 'Business :'."""
    return re.sub(r"^\s*[A-Za-z ]{3,20}\s*:\s*", "", title).strip() or title.strip()


def parse_feed(raw: str | bytes, feed: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise one parsed feed document into article dictionaries."""
    parsed = feedparser.parse(raw)
    if parsed.bozo and not parsed.entries:
        log.warning("feed %s could not be parsed: %s", feed.get("name"), parsed.bozo_exception)
        return []

    fetched = now_kl()
    out: list[dict[str, Any]] = []

    for entry in parsed.entries:
        url = (entry.get("link") or "").strip()
        title = _clean_title(strip_html(entry.get("title") or ""))
        if not url or not title:
            continue

        published = _parsed_time(entry)
        estimated = published is None
        if estimated:
            # No date in the feed, as with Bernama. The article page is fetched
            # later; until then the fetch time is used and flagged.
            published = fetched

        out.append(
            {
                "id": article_id(url),
                "url": url,
                "title": title,
                "source": feed.get("name", "unknown"),
                "published_at": published.isoformat(),
                "summary": strip_html(entry.get("summary") or ""),
                "body": _full_content(entry),
                "lang": entry.get("language") or "en",
                "date_estimated": int(estimated),
                "fetched_at": fetched.isoformat(),
            }
        )

    return out


def dedupe(articles: Iterable[dict[str, Any]], title_threshold: int = 90) -> list[dict[str, Any]]:
    """Drop repeats by URL, then near duplicate titles (PRD 6.1).

    Wire copy is republished across outlets with small title edits, so an exact
    title match is not enough.
    """
    from rapidfuzz import fuzz

    seen_urls: set[str] = set()
    kept: list[dict[str, Any]] = []

    for article in articles:
        if article["url"] in seen_urls:
            continue
        title = article["title"].lower()
        if any(fuzz.token_sort_ratio(title, k["title"].lower()) >= title_threshold for k in kept):
            continue
        seen_urls.add(article["url"])
        kept.append(article)

    return kept


def fetch_feed(feed: dict[str, Any], timeout: float = 20.0) -> list[dict[str, Any]]:
    """Download and parse one feed. Returns an empty list on any failure."""
    import httpx

    opts = config.fetch_opts()
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": opts["user_agent"]},
        ) as client:
            resp = client.get(feed["url"])
            resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - a dead feed must not stop the others
        log.warning("feed %s failed: %s", feed.get("name"), exc)
        return []

    articles = parse_feed(resp.content, feed)
    log.info("feed %s returned %d entries", feed.get("name"), len(articles))
    return articles


def fetch_all(hours: int | None = None, sources: list[str] | None = None) -> list[dict[str, Any]]:
    """Read every enabled feed, dedupe, and optionally keep only a recent window."""
    feeds = config.enabled_feeds()
    if sources:
        wanted = set(sources)
        feeds = [f for f in feeds if f.get("name") in wanted]

    collected: list[dict[str, Any]] = []
    for feed in feeds:
        collected.extend(fetch_feed(feed))

    articles = dedupe(collected)

    if hours is not None:
        cutoff = now_kl() - pd.Timedelta(hours=hours)
        articles = [
            a for a in articles if pd.Timestamp(a["published_at"]) >= cutoff
        ]

    articles.sort(key=lambda a: a["published_at"], reverse=True)
    return articles
