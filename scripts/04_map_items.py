"""Build the item mapping review sheet for manual step M3.

Usage:
    python scripts/04_map_items.py

Writes data/processed/item_mapping_review.csv with one row per candidate
(product, item_code). The `suggested` column is a pre-filled recommendation:
rows marked 1 are the ones to keep. The coverage columns say how much history
each item actually has, which is the thing you cannot see on data.gov.my and
the main reason to drop an item.

After reviewing, save the kept rows as config/item_mapping.csv with at least
the columns product and item_code.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mymarket import config  # noqa: E402
from mymarket.filter.aliases import find_terms  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger("map_items")

REVIEW_COLUMNS = [
    "suggested",
    "product",
    "item_code",
    "item",
    "unit",
    "item_group",
    "item_category",
    "match_reason",
    "n_days",
    "first_date",
    "last_date",
    "median_price",
    "avg_premises",
    "note",
]


def coverage() -> pd.DataFrame:
    """Per item_code history stats from the daily price table, if it exists."""
    if not config.DAILY_PRICES.exists():
        log.warning(
            "%s not found. Coverage columns will be blank. "
            "Run scripts/01_download_prices.py first for a much easier review.",
            config.DAILY_PRICES.name,
        )
        return pd.DataFrame(
            columns=["item_code", "n_days", "first_date", "last_date", "median_price", "avg_premises"]
        )

    daily = pd.read_parquet(config.DAILY_PRICES)
    stats = daily.groupby("item_code").agg(
        n_days=("date", "nunique"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        median_price=("median", "median"),
        avg_premises=("n_premises", "mean"),
    ).reset_index()
    stats["first_date"] = stats["first_date"].dt.date
    stats["last_date"] = stats["last_date"].dt.date
    stats["median_price"] = stats["median_price"].round(2)
    stats["avg_premises"] = stats["avg_premises"].round(1)
    return stats


def candidates(items: pd.DataFrame) -> pd.DataFrame:
    """Candidate (product, item) rows with a suggestion and a reason."""
    cfg = config.settings()
    rows: list[dict] = []

    clean = items.dropna(subset=["item"]).copy()
    clean["item"] = clean["item"].astype(str)
    clean = clean[clean["item_code"] >= 0]

    for product in cfg.products.values():
        by_alias = clean["item"].apply(lambda n: bool(find_terms(n, product.aliases)))
        by_category = clean["item_category"].isin(product.preferred_categories)
        pool = clean.loc[by_alias | by_category]

        for row in pool.itertuples(index=False):
            name = str(row.item).upper()
            in_category = row.item_category in product.preferred_categories
            rejected = [t for t in product.reject_item_terms if t.upper() in name]

            if in_category and not rejected:
                suggested, reason = 1, "category match"
            elif in_category and rejected:
                suggested, reason = 0, f"category match but name has {', '.join(rejected)}"
            else:
                suggested, reason = 0, "alias only, outside the product category"

            rows.append(
                {
                    "suggested": suggested,
                    "product": product.key,
                    "item_code": int(row.item_code),
                    "item": row.item,
                    "unit": row.unit,
                    "item_group": row.item_group,
                    "item_category": row.item_category,
                    "match_reason": reason,
                    "note": "",
                }
            )

    return pd.DataFrame(rows)


MIN_DAYS = 365
MIN_PREMISES = 50


def write_default(review: pd.DataFrame, per_product: int) -> pd.DataFrame:
    """Pick a provisional config/item_mapping.csv so the pipeline can run before M3.

    Ranks suggested items by history length times average premise count, which
    favours items that are both long running and widely surveyed. This is a
    starting point for the M3 review, not a substitute for it.
    """
    pool = review[review["suggested"] == 1].copy()
    pool = pool[pool["n_days"].notna() & pool["avg_premises"].notna()]
    pool = pool[(pool["n_days"] >= MIN_DAYS) & (pool["avg_premises"] >= MIN_PREMISES)]
    if pool.empty:
        raise RuntimeError(
            "no item has both at least "
            f"{MIN_DAYS} days of history and {MIN_PREMISES} average premises. "
            "Run scripts/01_download_prices.py first."
        )

    pool["score"] = pool["n_days"] * pool["avg_premises"]
    picked = (
        pool.sort_values(["product", "score"], ascending=[True, False])
        .groupby("product")
        .head(per_product)
        .sort_values(["product", "score"], ascending=[True, False], ignore_index=True)
    )

    out = picked[["product", "item_code", "item", "unit", "item_group", "item_category"]]
    out.to_csv(config.ITEM_MAPPING, index=False, encoding="utf-8-sig")
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-default",
        action="store_true",
        help="also write a provisional config/item_mapping.csv so the pipeline can run",
    )
    parser.add_argument("--per-product", type=int, default=4, help="items to keep per product")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing config/item_mapping.csv",
    )
    args = parser.parse_args(argv)

    config.ensure_dirs()

    if not config.LOOKUP_ITEM.exists():
        log.error(
            "%s not found. Run scripts/01_download_prices.py first.",
            config.LOOKUP_ITEM,
        )
        return 1

    items = pd.read_parquet(config.LOOKUP_ITEM)
    review = candidates(items)
    if review.empty:
        log.error("no candidate items matched. Check aliases in config/products.yaml")
        return 1

    review = review.merge(coverage(), on="item_code", how="left")
    for col in REVIEW_COLUMNS:
        if col not in review.columns:
            review[col] = pd.NA

    review = review[REVIEW_COLUMNS].sort_values(
        ["product", "suggested", "n_days"], ascending=[True, False, False], ignore_index=True
    )
    review.to_csv(config.ITEM_MAPPING_REVIEW, index=False, encoding="utf-8-sig")

    print(f"\nwrote {config.ITEM_MAPPING_REVIEW} with {len(review)} candidate rows\n")
    print("suggested keeps per product:")
    summary = review[review["suggested"] == 1].groupby("product").agg(
        items=("item_code", "count"),
        median_days=("n_days", "median"),
    )
    print(summary.to_string() if not summary.empty else "  none")

    thin = review[(review["suggested"] == 1) & (review["n_days"].notna()) & (review["n_days"] < 180)]
    if not thin.empty:
        print(f"\n{len(thin)} suggested items have under 180 days of history:")
        print(thin[["product", "item_code", "item", "n_days"]].to_string(index=False))
        print("Consider dropping these. A short series weakens the price index.")

    print("\nPriceCatcher item groups present:")
    for group in sorted(items["item_group"].dropna().unique()):
        print(f"  {group}")

    if args.write_default:
        if config.ITEM_MAPPING.exists() and not args.force:
            print(
                f"\n{config.ITEM_MAPPING} already exists. Not overwriting. "
                "Pass --force if you want the default regenerated."
            )
        else:
            picked = write_default(review, args.per_product)
            print(f"\nwrote a PROVISIONAL {config.ITEM_MAPPING.name} with {len(picked)} rows:")
            print(
                picked[["product", "item_code", "item", "unit", "n_days", "avg_premises"]]
                .to_string(index=False)
            )
            print(
                "\nThis is a ranked guess, not a reviewed mapping. It lets the price\n"
                "index and the label builder run now. Replace it at M3."
            )

    print(
        "\nNext, manual step M3:\n"
        "  1. Open data/processed/item_mapping_review.csv\n"
        "  2. Keep the rows you want, 2 to 3 per product is enough. The suggested\n"
        "     column is a starting point, not an answer. Prefer items with a long\n"
        "     n_days and a high avg_premises.\n"
        "  3. Save the kept rows as config/item_mapping.csv in UTF-8\n"
        "  4. Re-run scripts/01_download_prices.py (fast, it resumes) then\n"
        "     scripts/03_build_price_index.py\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
