"""SQLite cache and prediction log. Tables are those in PRD section 8."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from mymarket import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id            TEXT PRIMARY KEY,
    url           TEXT UNIQUE NOT NULL,
    title         TEXT NOT NULL,
    source        TEXT,
    published_at  TEXT,
    summary       TEXT,
    body          TEXT,
    lang          TEXT,
    date_estimated INTEGER DEFAULT 0,
    fetched_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS article_products (
    article_id    TEXT NOT NULL,
    product       TEXT NOT NULL,
    match_score   REAL,
    matched_terms TEXT,
    PRIMARY KEY (article_id, product),
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id    TEXT NOT NULL,
    product       TEXT NOT NULL,
    direction     TEXT NOT NULL,
    p_up          REAL,
    p_down        REAL,
    p_stable      REAL,
    model_version TEXT,
    created_at    TEXT NOT NULL,
    horizon_days  INTEGER NOT NULL,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id  INTEGER PRIMARY KEY,
    realized_ret   REAL,
    realized_label TEXT,
    scored_at      TEXT NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fetch_cache (
    url         TEXT PRIMARY KEY,
    body        TEXT,
    status      TEXT,
    fetched_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_predictions_article ON predictions(article_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the database, creating the schema when needed."""
    target = config.DB_PATH if path is None else path
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def session(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Connection context manager that commits on success and always closes."""
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_article(conn: sqlite3.Connection, article: dict[str, Any]) -> None:
    """Insert an article, refreshing the body when a better one arrives."""
    conn.execute(
        """
        INSERT INTO articles (id, url, title, source, published_at, summary, body,
                              lang, date_estimated, fetched_at)
        VALUES (:id, :url, :title, :source, :published_at, :summary, :body,
                :lang, :date_estimated, :fetched_at)
        ON CONFLICT(url) DO UPDATE SET
            title = excluded.title,
            summary = excluded.summary,
            body = CASE
                WHEN length(COALESCE(excluded.body, '')) > length(COALESCE(articles.body, ''))
                THEN excluded.body ELSE articles.body END,
            published_at = COALESCE(excluded.published_at, articles.published_at),
            date_estimated = excluded.date_estimated,
            fetched_at = excluded.fetched_at
        """,
        article,
    )


def link_product(
    conn: sqlite3.Connection,
    article_id: str,
    product: str,
    score: float,
    terms: list[str],
) -> None:
    """Record that an article matched a product."""
    conn.execute(
        """
        INSERT INTO article_products (article_id, product, match_score, matched_terms)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(article_id, product) DO UPDATE SET
            match_score = excluded.match_score,
            matched_terms = excluded.matched_terms
        """,
        (article_id, product, score, json.dumps(terms)),
    )


def log_prediction(conn: sqlite3.Connection, row: dict[str, Any]) -> int:
    """Append one prediction and return its id."""
    cur = conn.execute(
        """
        INSERT INTO predictions (article_id, product, direction, p_up, p_down,
                                 p_stable, model_version, created_at, horizon_days)
        VALUES (:article_id, :product, :direction, :p_up, :p_down, :p_stable,
                :model_version, :created_at, :horizon_days)
        """,
        row,
    )
    return int(cur.lastrowid)


def record_outcome(conn: sqlite3.Connection, prediction_id: int, ret: float, label: str, when: str) -> None:
    """Store the realised outcome for a matured prediction."""
    conn.execute(
        """
        INSERT INTO outcomes (prediction_id, realized_ret, realized_label, scored_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(prediction_id) DO UPDATE SET
            realized_ret = excluded.realized_ret,
            realized_label = excluded.realized_label,
            scored_at = excluded.scored_at
        """,
        (prediction_id, ret, label, when),
    )


def unscored_predictions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Predictions with no outcome row yet, oldest first."""
    return conn.execute(
        """
        SELECT p.*, a.published_at
        FROM predictions p
        JOIN articles a ON a.id = p.article_id
        LEFT JOIN outcomes o ON o.prediction_id = p.id
        WHERE o.prediction_id IS NULL
        ORDER BY p.created_at ASC
        """
    ).fetchall()


def scored_predictions(conn: sqlite3.Connection, product: str | None = None) -> list[sqlite3.Row]:
    """Matured predictions joined to their outcome."""
    sql = """
        SELECT p.product, p.direction, p.p_up, p.p_down, p.p_stable, p.created_at,
               p.horizon_days, o.realized_ret, o.realized_label, o.scored_at,
               a.title, a.url, a.source
        FROM predictions p
        JOIN outcomes o ON o.prediction_id = p.id
        JOIN articles a ON a.id = p.article_id
    """
    params: tuple = ()
    if product:
        sql += " WHERE p.product = ?"
        params = (product,)
    sql += " ORDER BY o.scored_at DESC"
    return conn.execute(sql, params).fetchall()


def cached_body(conn: sqlite3.Connection, url: str) -> str | None:
    """Previously fetched body for a URL, if any."""
    row = conn.execute("SELECT body FROM fetch_cache WHERE url = ?", (url,)).fetchone()
    return row["body"] if row else None


def cache_body(conn: sqlite3.Connection, url: str, body: str, status: str, when: str) -> None:
    """Store a fetched body so the same page is never fetched twice."""
    conn.execute(
        """
        INSERT INTO fetch_cache (url, body, status, fetched_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            body = excluded.body, status = excluded.status, fetched_at = excluded.fetched_at
        """,
        (url, body, status, when),
    )


def recent_articles(conn: sqlite3.Connection, hours: int = 48, limit: int = 100) -> list[sqlite3.Row]:
    """Articles published within the window, newest first."""
    return conn.execute(
        """
        SELECT * FROM articles
        WHERE published_at >= datetime('now', ?)
        ORDER BY published_at DESC
        LIMIT ?
        """,
        (f"-{int(hours)} hours", int(limit)),
    ).fetchall()
