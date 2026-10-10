"""Price index: outlier trim, normalisation, carry forward, smoothing, context."""

from __future__ import annotations

import pandas as pd
import pytest

from mymarket.prices import index as pi
from tests.conftest import daily_frame


def test_trim_outliers_drops_a_single_day_spike():
    frame = daily_frame(101, "2024-01-01", 60, 10.0)
    frame.loc[frame["date"] == pd.Timestamp("2024-02-01"), "median"] = 90.0

    out = pi.trim_outliers(frame)

    assert len(out) == 59
    assert pd.Timestamp("2024-02-01") not in set(out["date"])


def test_trim_outliers_keeps_a_genuine_step_change():
    # A subsidy removal doubles the price and it stays there. Nothing is dropped.
    frame = daily_frame(101, "2024-01-01", 120, 10.0, step_at=60, step_to=20.0)

    out = pi.trim_outliers(frame)

    assert len(out) == len(frame)


def test_trim_outliers_drops_non_positive_prices():
    frame = daily_frame(101, "2024-01-01", 30, 10.0)
    frame.loc[0, "median"] = 0.0

    assert len(pi.trim_outliers(frame)) == 29


def test_normalise_uses_the_first_90_day_median():
    frame = daily_frame(101, "2024-01-01", 200, 10.0, step_at=90, step_to=15.0)

    out = pi.normalise(frame)

    assert out["base"].unique().tolist() == [10.0]
    assert out.loc[out["date"] == pd.Timestamp("2024-01-05"), "rel"].iloc[0] == 1.0
    assert out.loc[out["date"] == pd.Timestamp("2024-06-01"), "rel"].iloc[0] == 1.5


def test_normalise_makes_different_units_comparable():
    # A 10 kg sack at RM 40 and a 1 kg packet at RM 4 both sit at 1.0.
    sack = pi.normalise(daily_frame(101, "2024-01-01", 100, 40.0))
    packet = pi.normalise(daily_frame(102, "2024-01-01", 100, 4.0))

    assert sack["rel"].iloc[0] == packet["rel"].iloc[0] == 1.0


def test_carry_forward_fills_a_weekly_survey():
    weekly = daily_frame(101, "2024-01-01", 29, 10.0, every=7)
    weekly = pi.normalise(weekly)
    weekly["product"] = "rice"

    out = pi.carry_forward(weekly)

    # 5 surveys on a 29 day span become a dense daily calendar.
    assert out["observed"].sum() == 5
    assert len(out) == 29
    assert out["rel"].notna().all()


def test_carry_forward_stops_after_the_limit():
    sparse = daily_frame(101, "2024-01-01", 61, 10.0, every=30)
    sparse = pi.normalise(sparse)
    sparse["product"] = "rice"

    out = pi.carry_forward(sparse, limit=5)

    # Three surveys 30 days apart, each carried at most 5 days forward.
    assert len(out) == 3 * (5 + 1) - 5  # the last survey has no gap to fill after it
    assert out["observed"].sum() == 3


def test_build_averages_items_and_smooths(test_settings):
    daily = pd.concat(
        [
            daily_frame(101, "2024-01-01", 200, 10.0),
            daily_frame(102, "2024-01-01", 200, 40.0),
            daily_frame(201, "2024-01-01", 200, 5.0),
        ],
        ignore_index=True,
    )

    out = pi.build(daily)

    assert set(out["product"]) == {"rice", "egg"}
    assert out["index"].round(6).unique().tolist() == [1.0]
    rice = out[out["product"] == "rice"]
    assert rice["n_items"].max() == 2


def test_build_skips_days_with_too_few_premises(test_settings):
    daily = daily_frame(101, "2024-01-01", 100, 10.0, n_premises=2)

    with pytest.raises(pi.PriceDataMissing):
        pi.build(daily)


def test_build_rejects_unmapped_item_codes(test_settings):
    daily = daily_frame(999, "2024-01-01", 100, 10.0)

    with pytest.raises(pi.PriceDataMissing):
        pi.build(daily)


def test_build_smoothing_lags_a_step(test_settings):
    daily = daily_frame(101, "2024-01-01", 200, 10.0, step_at=120, step_to=11.0)

    out = pi.build(daily)
    series = out.set_index("date")["index"]

    # The day the step lands, a 7 day mean has only caught one of seven days.
    step_day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=120)
    assert series[step_day] < 1.05
    assert series[step_day + pd.Timedelta(days=10)] == pytest.approx(1.1, abs=1e-6)


def test_pct_change_uses_the_nearest_earlier_value():
    series = pd.Series(
        [1.0, 1.05, 1.10],
        index=pd.to_datetime(["2024-01-01", "2024-01-15", "2024-01-29"]),
    )

    assert pi.pct_change(series, 14) == pytest.approx(4.76, abs=0.01)
    assert pi.pct_change(series, 28) == pytest.approx(10.0, abs=0.01)


def test_pct_change_returns_none_without_enough_history():
    series = pd.Series([1.0], index=pd.to_datetime(["2024-01-01"]))

    assert pi.pct_change(series, 30) is None


def test_pct_change_on_empty_series():
    assert pi.pct_change(pd.Series(dtype=float), 7) is None
