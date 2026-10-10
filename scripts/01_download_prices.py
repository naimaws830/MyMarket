"""Download PriceCatcher lookups and monthly files, aggregate to a daily table.

Usage:
    python scripts/01_download_prices.py                 # full run, resumes
    python scripts/01_download_prices.py --probe-only    # just find the first month
    python scripts/01_download_prices.py --month 2026-09 # one month only
    python scripts/01_download_prices.py --keep-raw      # do not delete raw files

One month at a time. Each month is reduced to a small daily aggregate and the
raw file is deleted, so peak disk use stays near one monthly file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mymarket import config  # noqa: E402
from mymarket.prices import pricecatcher as pc  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("download_prices")


def fetch_lookups() -> pd.DataFrame:
    """Download both lookup tables, validate them and return the item lookup."""
    config.ensure_dirs()

    for name, dest, expected in (
        ("item", config.LOOKUP_ITEM, pc.ITEM_COLUMNS),
        ("premise", config.LOOKUP_PREMISE, pc.PREMISE_COLUMNS),
    ):
        if dest.exists():
            log.info("lookup_%s already present, skipping download", name)
        else:
            size = pc.download(pc.lookup_url(name), dest)
            log.info("lookup_%s downloaded, %.1f KB", name, size / 1024)
        pc.check_columns(pd.read_parquet(dest), expected, f"lookup_{name}")

    items = pd.read_parquet(config.LOOKUP_ITEM)
    log.info("lookup_item has %d items", len(items))
    return items


def find_first_month(end: str) -> str | None:
    """Probe months forward from PROBE_START and return the earliest that exists."""
    for month in pc.months_range(config.PROBE_START, end):
        if pc.month_exists(month):
            log.info("first month found: %s", month)
            return month
        log.info("no file for %s", month)
    return None


def wanted_item_codes(items: pd.DataFrame) -> list[int]:
    """Item codes to keep.

    After the M3 review this is the curated mapping. Before it, every item code
    whose name matches a food alias is kept, so the raw files can be reduced now
    and the mapping narrowed later without re-downloading.
    """
    from mymarket.filter.aliases import find_terms

    mapped = [c for p in config.settings().products.values() for c in p.item_codes]
    if mapped:
        log.info("using %d item codes from config/item_mapping.csv", len(set(mapped)))
        return sorted(set(mapped))

    cfg = config.settings()
    terms = [t for p in cfg.products.values() for t in p.aliases]
    categories = {c for p in cfg.products.values() for c in p.preferred_categories}

    rows = items.dropna(subset=["item"]).copy()
    rows["item"] = rows["item"].astype(str)
    by_alias = rows["item"].apply(lambda n: bool(find_terms(n, terms)))
    by_category = rows["item_category"].isin(categories)

    codes = sorted(
        rows.loc[by_alias | by_category, "item_code"].astype("int64").unique().tolist()
    )
    log.warning(
        "config/item_mapping.csv not found. Keeping a superset of %d item codes "
        "(%d by alias, %d by category) so the M3 review can narrow it without "
        "re-downloading. Run scripts/04_map_items.py next.",
        len(codes),
        int(by_alias.sum()),
        int(by_category.sum()),
    )
    return codes


def process_month(month: str, item_codes: list[int], keep_raw: bool) -> int:
    """Download, aggregate and discard one month. Returns aggregated row count."""
    raw = config.PRICES_RAW / f"pricecatcher_{month}.parquet"
    try:
        size = pc.download(pc.month_url(month), raw)
    except FileNotFoundError:
        log.info("%s does not exist (404), skipping", month)
        return 0

    log.info("%s downloaded, %.1f MB", month, size / (1024 * 1024))
    try:
        raw_df = pd.read_parquet(raw)
        agg = pc.aggregate_month(raw_df, item_codes)
        del raw_df
        pc.DAILY_SHARDS.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(pc.shard_path(month), index=False)
        log.info("%s aggregated to %d daily rows", month, len(agg))
        return len(agg)
    finally:
        if not keep_raw and raw.exists():
            raw.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", help="process a single YYYY-MM month")
    parser.add_argument("--probe-only", action="store_true", help="only find the first month")
    parser.add_argument("--keep-raw", action="store_true", help="do not delete raw monthly files")
    parser.add_argument("--start", default=config.PROBE_START, help="first month to consider")
    args = parser.parse_args(argv)

    end = pc.current_month()
    items = fetch_lookups()

    if args.probe_only:
        first = find_first_month(end)
        if first is None:
            log.error(
                "no PriceCatcher month found between %s and %s. "
                "The URL pattern may have changed: check the URLs to dataset section "
                "at data.gov.my/data-catalogue/pricecatcher",
                args.start,
                end,
            )
            return 1
        return 0

    codes = wanted_item_codes(items)
    if not codes:
        log.error("no item codes to keep. Check the aliases in config/products.yaml")
        return 1

    months = [args.month] if args.month else list(pc.months_range(args.start, end))
    done = pc.processed_months()
    first_found: str | None = None
    total = 0

    for month in months:
        if month in done:
            log.info("%s already aggregated, skipping", month)
            first_found = first_found or month
            continue
        rows = process_month(month, codes, args.keep_raw)
        if rows:
            first_found = first_found or month
            total += rows

    combined = pc.combine_shards()
    if combined.empty:
        log.error(
            "no months were aggregated. Nothing to write. "
            "Run with --probe-only to check whether any month URL resolves."
        )
        return 1

    combined.to_parquet(config.DAILY_PRICES, index=False)

    log.info("first month found: %s", first_found)
    log.info(
        "wrote %s with %d rows, %s to %s, %d distinct items",
        config.DAILY_PRICES.name,
        len(combined),
        combined["date"].min().date(),
        combined["date"].max().date(),
        combined["item_code"].nunique(),
    )
    log.info("new rows this run: %d", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
