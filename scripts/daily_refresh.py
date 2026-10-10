"""Daily job: refresh prices, log live predictions, score matured ones.

Usage:
    python scripts/daily_refresh.py
    python scripts/daily_refresh.py --skip-prices   # news and scoring only

This is what makes forward validation real: a prediction is written the day it
is made and scored against official prices `horizon_days` later, so it cannot
be quietly revised after the fact. Schedule it at manual step M8.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mymarket import config  # noqa: E402
from mymarket.filter import relevance  # noqa: E402
from mymarket.filter.aliases import match_all  # noqa: E402
from mymarket.ingest import fetch_body, rss  # noqa: E402
from mymarket.model import labels as lb  # noqa: E402
from mymarket.model import predict as predictor  # noqa: E402
from mymarket.prices import index as price_index  # noqa: E402
from mymarket.prices import pricecatcher as pc  # noqa: E402
from mymarket.store import db  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("daily_refresh")


def refresh_prices() -> bool:
    """Re-download the current month and rebuild the index. True when it worked."""
    month = pc.current_month()
    log.info("refreshing price data for %s", month)

    download = __import__("importlib").import_module("importlib.util")
    spec = download.spec_from_file_location(
        "download_prices", Path(__file__).resolve().parent / "01_download_prices.py"
    )
    module = download.module_from_spec(spec)
    spec.loader.exec_module(module)

    shard = pc.shard_path(month)
    if shard.exists():
        shard.unlink()  # the month is still filling up, so always take it fresh

    if module.main(["--month", month]) != 0:
        log.warning("price refresh failed, continuing with the existing index")
        return False

    build = download.spec_from_file_location(
        "build_index", Path(__file__).resolve().parent / "03_build_price_index.py"
    )
    index_module = download.module_from_spec(build)
    build.loader.exec_module(index_module)
    return index_module.main([]) == 0


def log_predictions() -> int:
    """Scan live news, predict, and write every prediction to the database."""
    if not predictor.available():
        log.warning(
            "no trained model, so nothing can be predicted. "
            "Price data was still refreshed. Train first (manual step M6)."
        )
        return 0

    articles = rss.fetch_all(hours=48)
    shortlist = [a for a in articles if match_all(a["title"], a["summary"], a["body"])]
    fetch_body.enrich(shortlist, limit=15)
    matched = relevance.match_many(shortlist)
    log.info("scanned %d articles, %d matched", len(articles), len(matched))

    horizon = config.settings().horizon_days
    now = pd.Timestamp.now(tz=config.TZ).isoformat()
    written = 0

    with db.session() as conn:
        for article, matches in matched:
            db.upsert_article(
                conn,
                {
                    "id": article["id"],
                    "url": article["url"],
                    "title": article["title"],
                    "source": article["source"],
                    "published_at": article["published_at"],
                    "summary": article["summary"],
                    "body": article["body"],
                    "lang": article.get("lang", "en"),
                    "date_estimated": int(article.get("date_estimated", 0)),
                    "fetched_at": article["fetched_at"],
                },
            )

            for match in matches:
                db.link_product(
                    conn, article["id"], match.product, match.embed_score or 0.0, match.matched_terms
                )

            predictions = predictor.predict(
                article["title"], article["summary"], article["body"], [m.product for m in matches]
            )
            for prediction in predictions:
                if prediction.abstained:
                    continue  # an abstention is not a claim, so it is not scored
                db.log_prediction(
                    conn,
                    {
                        "article_id": article["id"],
                        "product": prediction.product,
                        "direction": prediction.direction,
                        "p_up": prediction.probs.get("UP"),
                        "p_down": prediction.probs.get("DOWN"),
                        "p_stable": prediction.probs.get("STABLE"),
                        "model_version": prediction.model_version,
                        "created_at": now,
                        "horizon_days": horizon,
                    },
                )
                written += 1

    log.info("logged %d predictions", written)
    return written


def score_matured() -> int:
    """Score every prediction whose horizon has passed against the real index."""
    try:
        index = price_index.load()
    except price_index.PriceDataMissing as exc:
        log.warning("cannot score predictions: %s", exc)
        return 0

    series = {
        product: group.set_index("date")["index"].sort_index()
        for product, group in index.groupby("product")
    }
    cfg = config.settings()
    now = pd.Timestamp.now(tz=config.TZ)
    scored = 0

    with db.session() as conn:
        for row in db.unscored_predictions(conn):
            published = pd.Timestamp(row["published_at"])
            if published.tzinfo is None:
                published = published.tz_localize(config.TZ)
            horizon = int(row["horizon_days"])

            if now < published + pd.Timedelta(days=horizon):
                continue  # not matured yet

            product_series = series.get(row["product"])
            if product_series is None:
                continue

            ret = lb.forward_return(product_series, published.tz_localize(None).normalize(), horizon)
            if ret is None:
                continue  # the index does not reach far enough forward yet

            # The same theta the training labels used, so live and offline agree.
            theta = max(lb.MIN_THETA, lb.theta_from(product_series.pct_change(horizon).dropna(), cfg.theta_k))
            db.record_outcome(conn, int(row["id"]), float(ret), lb.label_of(ret, theta), now.isoformat())
            scored += 1

    log.info("scored %d matured predictions", scored)
    return scored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-prices", action="store_true", help="do not re-download prices")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    started = pd.Timestamp.now(tz=config.TZ)
    log.info("daily refresh started at %s", started.isoformat())

    if not args.skip_prices:
        try:
            refresh_prices()
        except Exception as exc:  # noqa: BLE001 - never let this kill the run
            log.warning("price refresh raised %s, continuing", exc)

    written = log_predictions()
    scored = score_matured()

    log.info(
        "daily refresh finished in %.1fs: %d predictions logged, %d scored",
        (pd.Timestamp.now(tz=config.TZ) - started).total_seconds(),
        written,
        scored,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
