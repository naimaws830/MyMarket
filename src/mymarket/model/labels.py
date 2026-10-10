"""Event study labels and the time based split.

PRD 6.4 for the label rule, PRD F6 for why the split is by time with a purge
gap, and PRD F7 for why the class balance is reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mymarket import config

UP, DOWN, STABLE = "UP", "DOWN", "STABLE"

# How far either side of the wanted date a value may be taken from. The index is
# dense after the carry forward, so this only absorbs a ragged series end.
LOOKUP_TOLERANCE_DAYS = 3

# Smallest usable threshold, as a fraction. A product under a price ceiling has
# an almost flat train period (Malaysian eggs were administratively priced into
# 2024), which drives k * std towards zero. Without a floor every later wiggle
# of a tenth of a percent would be called a move. With it, a move has to clear
# 0.5 percent to count. Products that hit this floor are reported, because the
# floor and not the data is then deciding the labels.
MIN_THETA = 0.005


@dataclass(frozen=True)
class SplitSpec:
    """Fractions and purge gap for the time split."""

    train: float = 0.70
    val: float = 0.15
    purge_days: int = 14


def value_at(series: pd.Series, when: pd.Timestamp, tolerance: int = LOOKUP_TOLERANCE_DAYS) -> float | None:
    """Index value at a date, allowing a short backward reach for ragged days."""
    if series.empty:
        return None
    window = series[(series.index <= when) & (series.index >= when - pd.Timedelta(days=tolerance))]
    if window.empty:
        return None
    return float(window.iloc[-1])


def forward_return(
    series: pd.Series,
    publish_date: pd.Timestamp,
    horizon: int,
) -> float | None:
    """Return over the horizon window, per PRD 6.4.

    base is the index the day before publication, fut is the index `horizon`
    days after it. Returns None when either end is missing, which drops the
    example rather than guessing.
    """
    if series.empty:
        return None
    base = value_at(series, publish_date - pd.Timedelta(days=1))
    future_date = publish_date + pd.Timedelta(days=horizon)
    if future_date > series.index.max():
        return None
    fut = value_at(series, future_date)
    if base is None or fut is None or base <= 0:
        return None
    return float(fut / base - 1.0)


def trailing_return(
    series: pd.Series,
    publish_date: pd.Timestamp,
    horizon: int,
) -> float | None:
    """Return over the `horizon` days ENDING the day before publication.

    This is what the momentum baseline is allowed to know: how the price moved
    just before the article appeared. It never looks forward, so it is a fair
    opponent for the model rather than a free pass.
    """
    if series.empty:
        return None
    end = publish_date - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=horizon)
    if start < series.index.min():
        return None
    base, last = value_at(series, start), value_at(series, end)
    if base is None or last is None or base <= 0:
        return None
    return float(last / base - 1.0)


def theta_from(returns: pd.Series, k: float, floor: float = MIN_THETA) -> float:
    """Per product threshold, k times the standard deviation of TRAIN returns only.

    Never returns less than `floor`, see MIN_THETA.
    """
    clean = pd.Series(returns).dropna()
    if len(clean) < 2:
        return floor
    value = float(clean.std(ddof=1)) * float(k)
    if not np.isfinite(value):
        return floor
    return max(value, floor)


def label_of(ret: float | None, theta: float) -> str | None:
    """UP, DOWN or STABLE for one return, per PRD 6.4."""
    if ret is None or not np.isfinite(ret):
        return None
    if theta <= 0:
        return STABLE
    if ret > theta:
        return UP
    if ret < -theta:
        return DOWN
    return STABLE


def split_bounds(dates: pd.Series, spec: SplitSpec) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The two date boundaries that cut the timeline into train, val and test."""
    ordered = pd.to_datetime(pd.Series(dates)).dropna().sort_values()
    if ordered.empty:
        raise ValueError("cannot split an empty set of dates")
    train_end = ordered.quantile(spec.train)
    val_end = ordered.quantile(spec.train + spec.val)
    return pd.Timestamp(train_end).normalize(), pd.Timestamp(val_end).normalize()


def assign_split(dates: pd.Series, spec: SplitSpec | None = None) -> pd.Series:
    """Label each date train, val, test or purge.

    An example is purged when its label window would reach across a boundary:
    an article published within `purge_days` of the train end is still looking
    forward into validation data, so keeping it would leak (PRD F6).
    """
    spec = spec or SplitSpec()
    when = pd.to_datetime(pd.Series(dates)).dt.normalize()
    train_end, val_end = split_bounds(when, spec)
    gap = pd.Timedelta(days=spec.purge_days)

    out = pd.Series("purge", index=when.index, dtype=object)
    out[when <= train_end - gap] = "train"
    out[(when > train_end) & (when <= val_end - gap)] = "val"
    out[when > val_end] = "test"
    return out


def quarter_id(dates: pd.Series) -> pd.Series:
    """Quarter string per row, used for the rolling origin backtest."""
    return pd.to_datetime(pd.Series(dates)).dt.to_period("Q").astype(str)


def build_labels(
    examples: pd.DataFrame,
    index_frame: pd.DataFrame,
    horizon: int | None = None,
    k: float | None = None,
    spec: SplitSpec | None = None,
) -> pd.DataFrame:
    """Attach ret, label, split and quarter to one row per (article, product).

    `examples` needs `date` and `product`. The split is assigned first so that
    theta can be computed from the train period only, which is the rule that
    keeps the threshold out of the test set.
    """
    cfg = config.settings()
    horizon = cfg.horizon_days if horizon is None else horizon
    k = cfg.theta_k if k is None else k
    spec = spec or SplitSpec()

    out = examples.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values("date", ignore_index=True)
    out["split"] = assign_split(out["date"], spec)
    out["quarter"] = quarter_id(out["date"])

    series_by_product = {
        product: group.set_index("date")["index"].sort_index()
        for product, group in index_frame.groupby("product")
    }

    out["ret"] = [
        forward_return(series_by_product.get(row.product, pd.Series(dtype=float)), row.date, horizon)
        for row in out.itertuples(index=False)
    ]
    # The momentum baseline reads this. It is computed here so that the
    # baseline and the labels come from exactly the same price series.
    out["prev_ret"] = [
        trailing_return(series_by_product.get(row.product, pd.Series(dtype=float)), row.date, horizon)
        for row in out.itertuples(index=False)
    ]

    thetas: dict[str, float] = {}
    for product, group in out.groupby("product"):
        train_returns = group.loc[group["split"] == "train", "ret"]
        thetas[product] = theta_from(train_returns, k)

    out["theta"] = out["product"].map(thetas)
    out["label"] = [label_of(r, t) for r, t in zip(out["ret"], out["theta"])]
    out["horizon_days"] = horizon
    return out


def purge_proof(frame: pd.DataFrame, horizon: int) -> dict:
    """Evidence that no test example sits within the horizon of a train example."""
    train = frame.loc[frame["split"] == "train", "date"]
    test = frame.loc[frame["split"] == "test", "date"]
    val = frame.loc[frame["split"] == "val", "date"]

    def gap(earlier: pd.Series, later: pd.Series) -> int | None:
        if earlier.empty or later.empty:
            return None
        return int((later.min() - earlier.max()).days)

    return {
        "train_end": str(train.max().date()) if not train.empty else None,
        "val_start": str(val.min().date()) if not val.empty else None,
        "test_start": str(test.min().date()) if not test.empty else None,
        "train_to_val_gap_days": gap(train, val),
        "val_to_test_gap_days": gap(val, test),
        "train_to_test_gap_days": gap(train, test),
        "required_gap_days": horizon,
        "ok": all(
            g is None or g >= horizon
            for g in (gap(train, val), gap(val, test))
        ),
    }
