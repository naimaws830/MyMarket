"""SQLite store: upsert behaviour, prediction logging, outcome scoring."""

from __future__ import annotations

import pandas as pd
import pytest

from mymarket.store import db


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "test.sqlite")
    yield connection
    connection.close()


def article(**overrides) -> dict:
    base = {
        "id": "a1",
        "url": "https://example.my/a",
        "title": "Chicken prices climb",
        "source": "sample",
        "published_at": "2026-10-08T10:20:00+08:00",
        "summary": "A summary",
        "body": "A body",
        "lang": "en",
        "date_estimated": 0,
        "fetched_at": "2026-10-08T12:00:00+08:00",
    }
    base.update(overrides)
    return base


def test_schema_creates_every_table(conn):
    names = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }

    assert {"articles", "article_products", "predictions", "outcomes", "fetch_cache"} <= names


def test_upsert_is_idempotent_on_url(conn):
    db.upsert_article(conn, article())
    db.upsert_article(conn, article(title="Chicken prices climb further"))

    rows = conn.execute("SELECT title FROM articles").fetchall()
    assert len(rows) == 1
    assert rows[0]["title"] == "Chicken prices climb further"


def test_upsert_keeps_the_longer_body(conn):
    db.upsert_article(conn, article(body="a much longer body with real article text"))
    db.upsert_article(conn, article(body="short"))

    body = conn.execute("SELECT body FROM articles").fetchone()["body"]
    assert body == "a much longer body with real article text"


def test_link_product_is_idempotent(conn):
    db.upsert_article(conn, article())
    db.link_product(conn, "a1", "chicken", 0.08, ["chicken"])
    db.link_product(conn, "a1", "chicken", 0.09, ["chicken", "ayam"])

    rows = conn.execute("SELECT * FROM article_products").fetchall()
    assert len(rows) == 1
    assert rows[0]["match_score"] == 0.09


def prediction(**overrides) -> dict:
    base = {
        "article_id": "a1",
        "product": "chicken",
        "direction": "UP",
        "p_up": 0.7,
        "p_down": 0.1,
        "p_stable": 0.2,
        "model_version": "v1-test",
        "created_at": "2026-10-08T12:00:00+08:00",
        "horizon_days": 14,
    }
    base.update(overrides)
    return base


def test_prediction_starts_unscored_then_matures(conn):
    db.upsert_article(conn, article())
    pid = db.log_prediction(conn, prediction())

    assert len(db.unscored_predictions(conn)) == 1
    assert db.scored_predictions(conn) == []

    db.record_outcome(conn, pid, 0.031, "UP", "2026-10-22T12:00:00+08:00")

    assert db.unscored_predictions(conn) == []
    scored = db.scored_predictions(conn)
    assert len(scored) == 1
    assert scored[0]["realized_label"] == "UP"
    assert scored[0]["direction"] == "UP"


def test_record_outcome_is_idempotent(conn):
    db.upsert_article(conn, article())
    pid = db.log_prediction(conn, prediction())
    db.record_outcome(conn, pid, 0.03, "UP", "2026-10-22T12:00:00+08:00")
    db.record_outcome(conn, pid, 0.01, "STABLE", "2026-10-23T12:00:00+08:00")

    scored = db.scored_predictions(conn)
    assert len(scored) == 1
    assert scored[0]["realized_label"] == "STABLE"


def test_scored_predictions_can_filter_by_product(conn):
    db.upsert_article(conn, article())
    up = db.log_prediction(conn, prediction())
    egg = db.log_prediction(conn, prediction(product="egg", direction="DOWN"))
    db.record_outcome(conn, up, 0.03, "UP", "2026-10-22T12:00:00+08:00")
    db.record_outcome(conn, egg, -0.03, "DOWN", "2026-10-22T12:00:00+08:00")

    assert len(db.scored_predictions(conn, "chicken")) == 1
    assert len(db.scored_predictions(conn)) == 2


def test_fetch_cache_round_trip(conn):
    assert db.cached_body(conn, "https://example.my/a") is None

    db.cache_body(conn, "https://example.my/a", "body text", "ok", "2026-10-08T12:00:00+08:00")

    assert db.cached_body(conn, "https://example.my/a") == "body text"


def test_recent_articles_respects_the_window(conn):
    now = pd.Timestamp.now(tz="UTC")
    db.upsert_article(conn, article(id="new", url="https://example.my/new", published_at=now.isoformat()))
    db.upsert_article(
        conn,
        article(
            id="old",
            url="https://example.my/old",
            published_at=(now - pd.Timedelta(days=30)).isoformat(),
        ),
    )

    recent = db.recent_articles(conn, hours=48)

    assert [r["id"] for r in recent] == ["new"]


def test_deleting_an_article_cascades(conn):
    db.upsert_article(conn, article())
    db.link_product(conn, "a1", "chicken", 0.08, ["chicken"])
    db.log_prediction(conn, prediction())

    conn.execute("DELETE FROM articles WHERE id = 'a1'")

    assert conn.execute("SELECT COUNT(*) c FROM article_products").fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"] == 0
