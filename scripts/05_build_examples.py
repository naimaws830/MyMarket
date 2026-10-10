"""Join cleaned news to the price index and build labelled training examples.

Usage:
    python scripts/05_build_examples.py
    python scripts/05_build_examples.py --horizon 28 --k 0.3
    python scripts/05_build_examples.py --no-embed-filter   # alias stage only

Writes data/processed/examples.parquet with one row per (article, product).
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
from mymarket.model import labels as lb  # noqa: E402
from mymarket.model.features import (  # noqa: E402
    MERGED_LABEL,
    PRODUCT_COLUMN,
    event_frame,
    model_text,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger("build_examples")

MIN_EXAMPLES_PER_PRODUCT = 300


def candidates(news: pd.DataFrame, use_embeddings: bool) -> pd.DataFrame:
    """One row per (article, product) that passes the relevance filter."""
    rows: list[dict] = []
    total = len(news)

    for position, article in enumerate(news.itertuples(index=False), start=1):
        if position % 2000 == 0:
            log.info("filtered %d / %d articles", position, total)

        summary = getattr(article, "summary", "") or ""
        matches = relevance.match_article(
            article.title, summary, article.text, use_embeddings=use_embeddings
        )
        for match in matches:
            rows.append(
                {
                    "example_id": f"{article.url}|{match.product}",
                    "url": article.url,
                    "date": article.date,
                    "product": match.product,
                    "domain": config.DOMAIN,
                    "text": model_text(article.title, summary, article.text),
                    "title": article.title,
                    "matched_terms": ", ".join(match.matched_terms),
                    "event_words": ", ".join(match.event_words),
                    "embed_score": match.embed_score,
                }
            )

    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame) -> None:
    """Print label counts, class balance and the warnings that matter."""
    print("\nlabel counts per product:")
    counts = frame.pivot_table(
        index="product", columns="label", values="example_id", aggfunc="count", fill_value=0
    )
    print(counts.to_string())

    print("\nexamples per product per split:")
    print(
        frame.pivot_table(
            index="product", columns="split", values="example_id", aggfunc="count", fill_value=0
        ).to_string()
    )

    print("\noverall class balance:")
    print(frame["label"].value_counts(normalize=True).round(3).to_string())

    print("\ntheta per product (from TRAIN returns only):")
    for product, group in frame.groupby("product"):
        theta = group["theta"].iloc[0]
        floored = " <- at the floor, labels are decided by MIN_THETA not the data" if (
            abs(theta - lb.MIN_THETA) < 1e-12
        ) else ""
        print(f"  {product}: {theta:.5f}{floored}")

    thin = [p for p, g in frame.groupby("product") if len(g) < MIN_EXAMPLES_PER_PRODUCT]
    if thin:
        log.warning(
            "under %d examples for %s. Consider widening the horizon, loosening the "
            "filter, or merging these products into one group.",
            MIN_EXAMPLES_PER_PRODUCT,
            ", ".join(thin),
        )

    for product, group in frame.groupby("product"):
        for split in ("train", "val", "test"):
            if (group["split"] == split).sum() == 0:
                log.warning("product %s has no %s examples", product, split)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=None, help="label horizon in days")
    parser.add_argument("--k", type=float, default=None, help="theta multiplier")
    parser.add_argument("--purge-days", type=int, default=None, help="purge gap, defaults to the horizon")
    parser.add_argument("--no-embed-filter", action="store_true", help="skip the embedding stage")
    parser.add_argument(
        "--merge-products",
        action="store_true",
        help="train one model over all staples instead of per product features",
    )
    args = parser.parse_args(argv)

    cfg = config.settings()
    horizon = args.horizon or cfg.horizon_days
    k = cfg.theta_k if args.k is None else args.k
    purge = args.purge_days if args.purge_days is not None else horizon

    if not config.NEWS_CLEAN.exists():
        log.error(
            "%s not found. Run scripts/02_prepare_news.py first (manual step M1 "
            "must be done: the Kaggle files go in data/raw/news/).",
            config.NEWS_CLEAN,
        )
        return 1

    if not config.PRICE_INDEX.exists():
        log.error(
            "%s not found. Run scripts/01_download_prices.py, then "
            "scripts/04_map_items.py, then scripts/03_build_price_index.py.",
            config.PRICE_INDEX,
        )
        return 1

    news = pd.read_parquet(config.NEWS_CLEAN)
    index = pd.read_parquet(config.PRICE_INDEX)
    index["date"] = pd.to_datetime(index["date"]).dt.normalize()
    log.info("news rows %d, price index rows %d", len(news), len(index))

    # Only news inside the price window can ever be labelled.
    first, last = index["date"].min(), index["date"].max()
    inside = news[(news["date"] >= first) & (news["date"] <= last)]
    log.info(
        "price data covers %s to %s, so %d of %d articles are usable",
        first.date(), last.date(), len(inside), len(news),
    )
    if inside.empty:
        log.error("no article falls inside the price data window")
        return 1

    found = candidates(inside, use_embeddings=not args.no_embed_filter)
    if found.empty:
        log.error("no article matched any product. Check config/products.yaml")
        return 1
    log.info("relevance filter produced %d (article, product) rows", len(found))

    labelled = lb.build_labels(
        found, index, horizon=horizon, k=k, spec=lb.SplitSpec(purge_days=purge)
    )

    before = len(labelled)
    labelled = labelled[labelled["label"].notna()].reset_index(drop=True)
    log.info(
        "dropped %d rows with no price window, %d examples remain",
        before - len(labelled), len(labelled),
    )
    if labelled.empty:
        log.error("every row lost its price window. Check the overlap between news and prices.")
        return 1

    labelled = labelled.join(event_frame(labelled["text"].fillna("")))
    labelled[PRODUCT_COLUMN] = MERGED_LABEL if args.merge_products else labelled["product"]
    if args.merge_products:
        log.info("products merged into one group for modelling, %s", MERGED_LABEL)
    labelled.to_parquet(config.EXAMPLES, index=False)
    log.info("wrote %s with %d rows", config.EXAMPLES.name, len(labelled))

    summarise(labelled)

    proof = lb.purge_proof(labelled, horizon)
    print("\npurge gap proof (PRD F6):")
    for key, value in proof.items():
        print(f"  {key}: {value}")
    if not proof["ok"]:
        log.error("the purge gap is too small. The split would leak.")
        return 1

    modelled = labelled[labelled["split"].isin({"train", "val", "test"})]
    print(f"\nusable examples after purging: {len(modelled)} of {len(labelled)}")
    print("\nNext: python scripts/06_train_local.py (or manual step M6 on Colab)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
