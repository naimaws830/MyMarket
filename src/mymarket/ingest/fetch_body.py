"""Fetch article body text with Scrapling, politely.

Rules that are not negotiable here (AGENT_PROMPT hard rules, PRD F9):
  - plain `Fetcher` only, never `StealthyFetcher` or any anti bot bypass
  - robots.txt is honoured
  - at most one request per second per host
  - every response is cached in SQLite so a page is never fetched twice
  - a paywalled or near empty body falls back to the RSS summary
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.robotparser
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from mymarket import config
from mymarket.store import db

log = logging.getLogger(__name__)

_last_request: dict[str, float] = {}
_lock = threading.Lock()


@dataclass(frozen=True)
class Body:
    """Result of one body fetch."""

    text: str
    status: str  # ok, cached, short, blocked, robots, error, skipped
    from_cache: bool = False

    @property
    def usable(self) -> bool:
        return self.status in {"ok", "cached"}


def host_of(url: str) -> str:
    """Network location of a URL."""
    return urlparse(url).netloc.lower()


@lru_cache(maxsize=64)
def _robots(host: str, scheme: str = "https") -> urllib.robotparser.RobotFileParser | None:
    """Fetch and cache robots.txt for a host. None when it cannot be read."""
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(f"{scheme}://{host}/robots.txt")
    try:
        parser.read()
        return parser
    except Exception as exc:  # noqa: BLE001 - an unreadable robots.txt is not fatal
        log.warning("could not read robots.txt for %s: %s", host, exc)
        return None


def allowed(url: str, user_agent: str = "*") -> bool:
    """Whether robots.txt permits fetching this URL.

    An unreadable robots.txt is treated as allowed, which matches what a normal
    browser does. A robots.txt that exists and says no is always obeyed.
    """
    opts = config.fetch_opts()
    if not opts.get("respect_robots", True):
        return True

    parsed = urlparse(url)
    parser = _robots(parsed.netloc.lower(), parsed.scheme or "https")
    if parser is None:
        return True
    try:
        return bool(parser.can_fetch(user_agent, url))
    except Exception:  # noqa: BLE001
        return True


def throttle(host: str, min_gap: float) -> None:
    """Block until at least `min_gap` seconds have passed since this host was hit."""
    with _lock:
        previous = _last_request.get(host)
        now = time.monotonic()
        if previous is not None:
            wait = min_gap - (now - previous)
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
        _last_request[host] = now


def extract(page: Any) -> str:
    """Readable text from a Scrapling page, preferring its markdown rendering."""
    for attempt in ("markdown", "get_all_text"):
        method = getattr(page, attempt, None)
        if method is None:
            continue
        try:
            text = method() if callable(method) else method
        except Exception as exc:  # noqa: BLE001
            log.debug("%s extraction failed: %s", attempt, exc)
            continue
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def fetch(url: str, conn=None, force: bool = False) -> Body:
    """Fetch one article body, using the cache, robots.txt and the rate limit."""
    opts = config.fetch_opts()
    min_chars = int(opts["min_body_chars"])

    owns_connection = conn is None
    conn = db.connect() if owns_connection else conn
    try:
        if not force:
            cached = db.cached_body(conn, url)
            if cached is not None:
                return Body(cached, "cached" if len(cached) >= min_chars else "short", True)

        if not allowed(url, opts["user_agent"]):
            log.info("robots.txt disallows %s", url)
            return Body("", "robots")

        throttle(host_of(url), float(opts["rate_limit_seconds"]))

        try:
            from scrapling.fetchers import Fetcher

            page = Fetcher.get(url, timeout=int(opts["timeout_seconds"]))
        except Exception as exc:  # noqa: BLE001 - one dead page must not stop a scan
            log.warning("fetch failed for %s: %s", url, exc)
            return Body("", "error")

        status_code = getattr(page, "status", 200)
        if status_code and int(status_code) >= 400:
            log.info("fetch returned %s for %s", status_code, url)
            return Body("", "blocked")

        text = extract(page)
        status = "ok" if len(text) >= min_chars else "short"

        db.cache_body(conn, url, text, status, _now())
        if owns_connection:
            conn.commit()

        if status == "short":
            # Paywalled or a stub page. The caller keeps the RSS summary (PRD 6.1).
            log.info("body for %s is only %d chars, treating as unusable", url, len(text))

        return Body(text, status)
    finally:
        if owns_connection:
            conn.close()


def _now() -> str:
    import pandas as pd

    return pd.Timestamp.now(tz=config.TZ).isoformat()


def enrich(articles: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    """Fill in missing bodies for articles whose feed carried none.

    Articles that already have a usable body are left alone, which is why
    Malay Mail costs no requests at all.
    """
    opts = config.fetch_opts()
    min_chars = int(opts["min_body_chars"])
    fetched = 0

    with db.session() as conn:
        for article in articles:
            if len(article.get("body") or "") >= min_chars:
                continue
            if limit is not None and fetched >= limit:
                article["body_status"] = "skipped"
                continue

            result = fetch(article["url"], conn=conn)
            fetched += 1
            article["body_status"] = result.status
            if result.usable:
                article["body"] = result.text

    return articles
