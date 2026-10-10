"""Daily price index per product, and the price_context the MCP server serves.

Implements PRD 6.4 steps 1 to 3. The index is a unitless level, not a ringgit
price: each item is normalised by its own early median so a 10 kg rice sack and
a 1 kg oil packet can be averaged without the sack dominating (PRD F11).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mymarket import config

OUTLIER_HIGH = 3.0
OUTLIER_LOW = 1 / 3
ROLLING_WINDOW = 30
NORMALISE_DAYS = 90
SMOOTH_WINDOW = 7

# PriceCatcher does not survey every item every day, and the cadence thins out
# over time: by 2026 cooking oil items are surveyed about weekly while eggs are
# still near daily. A survey price is a step function, so the last observed
# price is the best estimate of the shelf price until the next survey. Each item
# is therefore carried forward for at most this many days before it goes stale.
FFILL_LIMIT_DAYS = 10

INDEX_COLUMNS = ["date", "product", "index", "n_items", "n_obs", "n_premises"]


class PriceDataMissing(RuntimeError):
    """The processed price index has not been built yet."""


def trim_outliers(daily: pd.DataFrame) -> pd.DataFrame:
    """Drop item-days whose median sits outside [1/3, 3] x its rolling 30 day median.

    The rolling median is centred and computed per item, so a genuine step change
    such as a subsidy removal survives while a one day data glitch does not.
    """
    out = daily.sort_values(["item_code", "date"]).copy()
    out = out[out["median"] > 0]

    reference = (
        out.groupby("item_code")["median"]
        .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=5, center=True).median())
    )
    # Items with too little history to form a reference are kept as they are.
    reference = reference.fillna(out["median"])

    keep = (out["median"] <= reference * OUTLIER_HIGH) & (out["median"] >= reference * OUTLIER_LOW)
    return out[keep].reset_index(drop=True)


def normalise(daily: pd.DataFrame) -> pd.DataFrame:
    """Add a `rel` column: each item divided by its own median over its first 90 days."""
    out = daily.sort_values(["item_code", "date"]).copy()

    def base(group: pd.DataFrame) -> float:
        start = group["date"].min()
        early = group.loc[group["date"] < start + pd.Timedelta(days=NORMALISE_DAYS), "median"]
        value = early.median()
        return float(value) if pd.notna(value) and value > 0 else float("nan")

    bases = out.groupby("item_code", group_keys=False).apply(base, include_groups=False)
    out["base"] = out["item_code"].map(bases)
    out = out[out["base"].notna() & (out["base"] > 0)]
    out["rel"] = out["median"] / out["base"]
    return out.reset_index(drop=True)


def carry_forward(daily: pd.DataFrame, limit: int = FFILL_LIMIT_DAYS) -> pd.DataFrame:
    """Put every item on a daily calendar, carrying each price forward up to `limit` days.

    Without this, a product whose items are surveyed weekly produces a mostly
    empty calendar and the smoothed index dies out. The `observed` column keeps
    the real survey days distinguishable from carried ones.
    """
    pieces: list[pd.DataFrame] = []

    for (product, code), group in daily.groupby(["product", "item_code"]):
        series = group.set_index("date").sort_index()
        series = series[~series.index.duplicated(keep="last")]
        calendar = pd.date_range(series.index.min(), series.index.max(), freq="D")

        filled = series.reindex(calendar)
        filled["observed"] = filled["rel"].notna()
        filled["rel"] = filled["rel"].ffill(limit=limit)
        filled["n_premises"] = filled["n_premises"].fillna(0)
        filled = filled[filled["rel"].notna()]

        filled["product"] = product
        filled["item_code"] = code
        filled.index.name = "date"
        pieces.append(filled.reset_index()[["date", "product", "item_code", "rel", "observed", "n_premises"]])

    if not pieces:
        return pd.DataFrame(columns=["date", "product", "item_code", "rel", "observed", "n_premises"])
    return pd.concat(pieces, ignore_index=True)


def build(daily: pd.DataFrame, min_premises: int | None = None) -> pd.DataFrame:
    """Build the per product daily index from the aggregated daily price table.

    Steps follow PRD 6.4: thin days out, outliers trimmed, each item normalised
    by its own early median, averaged across the items of a product, then a 7 day
    rolling mean.
    """
    cfg = config.settings()
    floor = cfg.min_premises if min_premises is None else min_premises

    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame[frame["n_premises"] >= floor]

    code_to_product = {
        code: product.key
        for product in cfg.products.values()
        for code in product.item_codes
    }
    if not code_to_product:
        raise PriceDataMissing(
            "no item codes are mapped. Complete manual step M3 and save "
            "config/item_mapping.csv, then re-run."
        )

    frame["product"] = frame["item_code"].map(code_to_product)
    frame = frame[frame["product"].notna()]
    if frame.empty:
        raise PriceDataMissing(
            "none of the mapped item codes appear in the daily price table. "
            "Check that config/item_mapping.csv uses PriceCatcher item_code values."
        )

    frame = normalise(trim_outliers(frame))
    daily_items = carry_forward(frame)

    grouped = daily_items.groupby(["product", "date"]).agg(
        raw=("rel", "mean"),
        n_items=("item_code", "nunique"),
        n_obs=("observed", "sum"),
        n_premises=("n_premises", "sum"),
    ).reset_index()

    pieces = []
    for product, group in grouped.groupby("product"):
        series = group.set_index("date").sort_index()
        # The calendar is already dense after the carry forward, so a 7 row
        # window is a 7 day window.
        series["index"] = series["raw"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
        series["product"] = product
        series.index.name = "date"
        pieces.append(series.reset_index())

    out = pd.concat(pieces, ignore_index=True)
    out = out[out["index"].notna()]
    for col in ("n_items", "n_obs", "n_premises"):
        out[col] = out[col].astype("Int64")
    return out[INDEX_COLUMNS].sort_values(["product", "date"], ignore_index=True)


def load() -> pd.DataFrame:
    """Read the built price index, raising a clear error when it is missing."""
    if not config.PRICE_INDEX.exists():
        raise PriceDataMissing(
            f"{config.PRICE_INDEX.name} not found. Run scripts/01_download_prices.py "
            "then scripts/03_build_price_index.py."
        )
    frame = pd.read_parquet(config.PRICE_INDEX)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def series_for(product: str, frame: pd.DataFrame | None = None) -> pd.Series:
    """Index series for one product, indexed by date."""
    data = load() if frame is None else frame
    rows = data[data["product"] == product]
    return rows.set_index("date")["index"].sort_index()


def pct_change(series: pd.Series, days: int) -> float | None:
    """Percent change over the trailing window, using the nearest earlier value."""
    if series.empty:
        return None
    latest_date = series.index[-1]
    past = series[series.index <= latest_date - pd.Timedelta(days=days)]
    if past.empty:
        return None
    base = float(past.iloc[-1])
    if base == 0:
        return None
    return round((float(series.iloc[-1]) / base - 1) * 100, 2)


@dataclass(frozen=True)
class Context:
    """What the price_context tool returns for one product."""

    product: str
    latest_index: float
    change_7d_pct: float | None
    change_30d_pct: float | None
    data_as_of: str
    n_days: int
    frequency: str
    series_tail: list[dict]

    def to_dict(self) -> dict:
        return {
            "product": self.product,
            "latest_index": self.latest_index,
            "change_7d_pct": self.change_7d_pct,
            "change_30d_pct": self.change_30d_pct,
            "data_as_of": self.data_as_of,
            "n_days": self.n_days,
            "frequency": self.frequency,
            "index_note": (
                "Unitless index. Each item is normalised by its own median over its "
                "first 90 days, then averaged across the items of this product."
            ),
            "source": "PriceCatcher (KPDN via data.gov.my), CC BY 4.0",
            "series_tail": self.series_tail,
        }


def context(product: str, days: int = 90, tail: int = 30) -> Context:
    """Build the price context for one product over the trailing window."""
    series = series_for(product)
    if series.empty:
        raise PriceDataMissing(f"no price index rows for product {product!r}")

    window = series[series.index >= series.index[-1] - pd.Timedelta(days=days)]
    points = [
        {"date": d.strftime("%Y-%m-%d"), "index": round(float(v), 4)}
        for d, v in window.tail(tail).items()
    ]

    return Context(
        product=product,
        latest_index=round(float(series.iloc[-1]), 4),
        change_7d_pct=pct_change(series, 7),
        change_30d_pct=pct_change(series, 30),
        data_as_of=series.index[-1].strftime("%Y-%m-%d"),
        n_days=int(len(series)),
        frequency="daily",
        series_tail=points,
    )
