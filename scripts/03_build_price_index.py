"""Build the daily price index per product and save a sanity plot for each.

Usage:
    python scripts/03_build_price_index.py

Reads data/processed/daily_prices.parquet and config/item_mapping.csv.
Writes data/processed/price_index.parquet and data/processed/plots/*.png.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mymarket import config  # noqa: E402
from mymarket.prices import index as price_index  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger("build_price_index")


def plot(frame: pd.DataFrame) -> list[Path]:
    """Save one PNG per product so the index can be eyeballed (PRD M3 check)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for product, group in frame.groupby("product"):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(group["date"], group["index"], linewidth=1.2)
        ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(f"{product}: normalised price index (7 day mean)")
        ax.set_ylabel("index, 1.0 = first 90 day median")
        ax.set_xlabel("date")
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()

        dest = config.PLOTS_DIR / f"price_index_{product}.png"
        fig.savefig(dest, dpi=110)
        plt.close(fig)
        written.append(dest)

    return written


def report(frame: pd.DataFrame) -> None:
    """Print per product coverage and the biggest single day moves."""
    print("\nprice index per product:")
    summary = frame.groupby("product").agg(
        days=("date", "nunique"),
        first=("date", "min"),
        last=("date", "max"),
        index_min=("index", "min"),
        index_max=("index", "max"),
        avg_items=("n_items", "mean"),
    )
    summary["first"] = summary["first"].dt.date
    summary["last"] = summary["last"].dt.date
    print(summary.round(3).to_string())

    print("\nlargest 14 day moves per product (a sanity check, not a result):")
    for product, group in frame.groupby("product"):
        series = group.set_index("date")["index"].sort_index()
        change = series.pct_change(periods=14).dropna()
        if change.empty:
            print(f"  {product}: not enough history")
            continue
        up, down = change.idxmax(), change.idxmin()
        print(
            f"  {product}: max +{change.max() * 100:.1f}% to {up.date()}, "
            f"max {change.min() * 100:.1f}% to {down.date()}"
        )


def main(argv: list[str] | None = None) -> int:
    if not config.DAILY_PRICES.exists():
        log.error(
            "%s not found. Run scripts/01_download_prices.py first.", config.DAILY_PRICES
        )
        return 1

    if not config.ITEM_MAPPING.exists():
        log.error(
            "config/item_mapping.csv not found. Run "
            "scripts/04_map_items.py --write-default for a provisional mapping, "
            "or complete manual step M3."
        )
        return 1

    daily = pd.read_parquet(config.DAILY_PRICES)
    log.info("read %d daily price rows", len(daily))

    try:
        frame = price_index.build(daily)
    except price_index.PriceDataMissing as exc:
        log.error("%s", exc)
        return 1

    frame.to_parquet(config.PRICE_INDEX, index=False)
    log.info("wrote %s with %d rows", config.PRICE_INDEX.name, len(frame))

    report(frame)
    for path in plot(frame):
        print(f"plot: {path}")

    missing = set(config.settings().product_keys) - set(frame["product"].unique())
    if missing:
        log.warning(
            "no index built for %s. Check config/item_mapping.csv has rows for them.",
            ", ".join(sorted(missing)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
