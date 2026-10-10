"""RSS parsing: content:encoded, missing dates, dedupe, html stripping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mymarket.ingest import rss

FIXTURE = Path(__file__).parent / "fixtures" / "sample_feed.xml"
FEED = {"name": "sample", "url": "https://example.my/feed"}


@pytest.fixture
def entries() -> list[dict]:
    return rss.parse_feed(FIXTURE.read_bytes(), FEED)


def by_title(entries: list[dict], fragment: str) -> dict:
    return next(e for e in entries if fragment in e["title"])


def test_items_without_a_link_are_skipped(entries):
    assert all(e["url"] for e in entries)
    assert not any("Missing link" in e["title"] for e in entries)


def test_full_content_is_used_when_the_feed_carries_it(entries):
    item = by_title(entries, "Chicken prices climb as feed costs")

    assert "feed cost pressure continued" in item["body"]
    assert "<b>" not in item["body"]


def test_summary_only_item_has_an_empty_body(entries):
    item = by_title(entries, "Egg supply steady")

    assert item["body"] == ""
    assert "Egg supply is steady" in item["summary"]


def test_published_date_is_parsed_into_the_project_timezone(entries):
    item = by_title(entries, "Chicken prices climb as feed costs")
    when = pd.Timestamp(item["published_at"])

    assert item["date_estimated"] == 0
    assert when.tz is not None
    assert when.strftime("%Y-%m-%d %H:%M") == "2026-10-08 10:20"


def test_missing_date_is_estimated_and_flagged(entries):
    item = by_title(entries, "Egg supply steady")

    assert item["date_estimated"] == 1
    assert pd.Timestamp(item["published_at"]).tz is not None


def test_category_prefix_is_stripped_from_the_title(entries):
    assert by_title(entries, "Egg supply steady")["title"] == "Egg supply steady, says ministry"


def test_ids_are_stable_and_unique(entries):
    ids = [e["id"] for e in entries]

    assert len(ids) == len(set(ids))
    assert rss.article_id("https://example.my/a") == rss.article_id("https://example.my/a")


def test_dedupe_drops_near_duplicate_titles(entries):
    deduped = rss.dedupe(entries)

    chicken = [e for e in deduped if e["title"].startswith("Chicken prices climb")]
    assert len(chicken) == 1
    assert len(deduped) == len(entries) - 1


def test_dedupe_drops_repeat_urls():
    one = {"url": "https://example.my/a", "title": "Alpha story"}
    two = {"url": "https://example.my/a", "title": "Completely different headline"}

    assert len(rss.dedupe([one, two])) == 1


def test_strip_html_unescapes_entities():
    assert rss.strip_html("<p>Harga &amp; kos naik</p>") == "Harga & kos naik"
    assert rss.strip_html(None) == ""


def test_parse_feed_on_garbage_returns_nothing():
    assert rss.parse_feed(b"not a feed at all", FEED) == []
