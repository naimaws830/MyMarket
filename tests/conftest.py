"""Shared fixtures. Nothing here touches the network or the real data files."""

from __future__ import annotations

import pandas as pd
import pytest

from mymarket import config


def make_settings(**overrides) -> config.Settings:
    """A small two product Settings object for tests."""
    products = {
        "rice": config.Product(
            key="rice",
            label="Rice",
            aliases_en=["rice"],
            aliases_ms=["beras"],
            exclude_phrases=["fried rice"],
            description_sentences=["The retail price of rice."],
            item_codes=[101, 102],
        ),
        "egg": config.Product(
            key="egg",
            label="Egg",
            aliases_en=["egg"],
            aliases_ms=["telur"],
            exclude_phrases=["Easter egg"],
            description_sentences=["The retail price of eggs."],
            item_codes=[201],
        ),
    }
    base = dict(
        products=products,
        event_words=["subsidy", "flood"],
        horizon_days=14,
        theta_k=0.5,
        min_premises=5,
        embed_threshold=0.055,
        body_chars=1500,
        min_confidence=0.55,
        distractor_sentences=["A recipe or a restaurant review."],
        requires_price_cue=False,
        price_cue_words=["price", "prices", "subsidy", "shortage", "cost"],
    )
    base.update(overrides)
    return config.Settings(**base)


@pytest.fixture
def test_settings(monkeypatch) -> config.Settings:
    """Replace the cached project settings with the small test set."""
    cfg = make_settings()
    monkeypatch.setattr(config, "settings", lambda: cfg)
    return cfg


def daily_frame(
    item_code: int,
    start: str,
    days: int,
    price: float,
    n_premises: int = 50,
    step_at: int | None = None,
    step_to: float | None = None,
    every: int = 1,
) -> pd.DataFrame:
    """Synthetic daily aggregate rows for one item.

    `every` controls survey cadence, so a weekly surveyed item can be simulated.
    `step_at` and `step_to` introduce a level change, as a subsidy removal would.
    """
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []
    for offset, date in enumerate(dates):
        if offset % every:
            continue
        value = price if step_at is None or offset < step_at else float(step_to)
        rows.append(
            {
                "date": date,
                "item_code": item_code,
                "median": value,
                "p25": value * 0.98,
                "p75": value * 1.02,
                "n_premises": n_premises,
            }
        )
    return pd.DataFrame(rows)
