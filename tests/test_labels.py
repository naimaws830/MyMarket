"""Label builder: return window, theta from train only, purge gap, split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mymarket.model import labels as lb


def index_frame(product: str, start: str, days: int, values) -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq="D")
    series = values if hasattr(values, "__len__") else [values] * days
    return pd.DataFrame(
        {"date": dates, "product": product, "index": series, "n_items": 3, "n_obs": 3, "n_premises": 50}
    )


def test_forward_return_measures_base_to_horizon():
    rising = np.linspace(1.0, 2.0, 201)
    series = index_frame("rice", "2024-01-01", 201, rising).set_index("date")["index"]

    ret = lb.forward_return(series, pd.Timestamp("2024-01-11"), horizon=14)

    # base is day 9 (value at d-1), fut is day 24.
    assert ret == pytest.approx(rising[24] / rising[9] - 1, abs=1e-9)


def test_forward_return_is_none_past_the_series_end():
    series = index_frame("rice", "2024-01-01", 30, 1.0).set_index("date")["index"]

    assert lb.forward_return(series, pd.Timestamp("2024-01-25"), horizon=14) is None


def test_forward_return_is_none_before_the_series_start():
    series = index_frame("rice", "2024-02-01", 60, 1.0).set_index("date")["index"]

    assert lb.forward_return(series, pd.Timestamp("2024-01-01"), horizon=14) is None


def test_label_boundaries():
    assert lb.label_of(0.05, 0.02) == lb.UP
    assert lb.label_of(-0.05, 0.02) == lb.DOWN
    assert lb.label_of(0.01, 0.02) == lb.STABLE
    # Exactly at theta is STABLE: the rule is strictly greater than.
    assert lb.label_of(0.02, 0.02) == lb.STABLE
    assert lb.label_of(-0.02, 0.02) == lb.STABLE


def test_label_of_handles_missing_and_degenerate_input():
    assert lb.label_of(None, 0.02) is None
    assert lb.label_of(float("nan"), 0.02) is None
    assert lb.label_of(0.5, 0.0) == lb.STABLE


def test_theta_uses_k_times_std():
    returns = pd.Series([0.0, 0.1, -0.1, 0.2, -0.2])

    assert lb.theta_from(returns, 0.5) == pytest.approx(returns.std(ddof=1) * 0.5)


def test_theta_falls_back_to_the_floor_without_enough_returns():
    assert lb.theta_from(pd.Series([0.1]), 0.5) == lb.MIN_THETA
    assert lb.theta_from(pd.Series(dtype=float), 0.5) == lb.MIN_THETA


def test_theta_floor_protects_a_price_controlled_product():
    # A product held at a fixed price has near zero variance in train. Without
    # the floor every later rounding wiggle would be labelled a move.
    flat = pd.Series([0.0, 1e-9, -1e-9, 0.0])

    assert lb.theta_from(flat, 0.5) == lb.MIN_THETA


def test_assign_split_is_ordered_in_time():
    dates = pd.Series(pd.date_range("2024-01-01", periods=400, freq="D"))

    split = lb.assign_split(dates)
    frame = pd.DataFrame({"date": dates, "split": split})

    train_max = frame.loc[frame.split == "train", "date"].max()
    val_min = frame.loc[frame.split == "val", "date"].min()
    test_min = frame.loc[frame.split == "test", "date"].min()

    assert train_max < val_min < test_min
    assert set(frame["split"]) <= {"train", "val", "test", "purge"}


def test_assign_split_purges_the_gap_before_each_boundary():
    dates = pd.Series(pd.date_range("2024-01-01", periods=400, freq="D"))
    spec = lb.SplitSpec(purge_days=14)

    frame = pd.DataFrame({"date": dates, "split": lb.assign_split(dates, spec)})

    train_max = frame.loc[frame.split == "train", "date"].max()
    val_min = frame.loc[frame.split == "val", "date"].min()
    val_max = frame.loc[frame.split == "val", "date"].max()
    test_min = frame.loc[frame.split == "test", "date"].min()

    assert (val_min - train_max).days >= 14
    assert (test_min - val_max).days >= 14
    assert (frame["split"] == "purge").sum() > 0


def test_build_labels_computes_theta_from_train_only(test_settings):
    # Flat in train, violently volatile in test. A theta that saw the test set
    # would be large and would label the test moves STABLE.
    days = 500
    values = np.ones(days)
    values[400:] = np.tile([1.0, 1.6], 50)[: days - 400]
    index = index_frame("rice", "2023-01-01", days, values)

    examples = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=days, freq="D"),
            "product": "rice",
        }
    )

    out = lb.build_labels(examples, index, horizon=14, k=0.5)
    theta = out["theta"].iloc[0]

    # Train is flat, so theta lands on the floor rather than on the test volatility.
    assert theta == lb.MIN_THETA
    # A theta fitted on the test period would be ~0.15 and would call these STABLE.
    test_labels = set(out.loc[out["split"] == "test", "label"].dropna())
    assert test_labels & {lb.UP, lb.DOWN}


def test_build_labels_drops_rows_without_a_price_window(test_settings):
    index = index_frame("rice", "2024-01-01", 60, 1.0)
    examples = pd.DataFrame(
        {"date": [pd.Timestamp("2024-01-10"), pd.Timestamp("2024-02-25")], "product": "rice"}
    )

    out = lb.build_labels(examples, index, horizon=14, k=0.5)

    assert out["label"].isna().sum() == 1


def test_build_labels_handles_a_product_with_no_index(test_settings):
    index = index_frame("rice", "2024-01-01", 200, 1.0)
    examples = pd.DataFrame({"date": [pd.Timestamp("2024-02-01")], "product": "egg"})

    out = lb.build_labels(examples, index, horizon=14, k=0.5)

    assert out["label"].isna().all()


def test_purge_proof_reports_the_real_gaps():
    dates = pd.Series(pd.date_range("2024-01-01", periods=400, freq="D"))
    frame = pd.DataFrame({"date": dates, "split": lb.assign_split(dates)})

    proof = lb.purge_proof(frame, horizon=14)

    assert proof["ok"] is True
    assert proof["train_to_val_gap_days"] >= 14
    assert proof["val_to_test_gap_days"] >= 14


def test_quarter_id():
    dates = pd.Series(pd.to_datetime(["2024-01-15", "2024-05-02"]))

    assert lb.quarter_id(dates).tolist() == ["2024Q1", "2024Q2"]
