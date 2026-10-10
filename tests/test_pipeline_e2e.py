"""End to end run of the offline pipeline on synthetic data.

This exists because the real pipeline cannot run until the Kaggle news dump is
downloaded by hand (manual step M1). Rather than ship untested code and hope,
this builds a small synthetic news set with a deliberate signal in it, runs
clean -> examples -> labels -> train -> evaluate, and checks that real model
files come out the other side.

It is slow by test standards (it embeds a few hundred strings on CPU), so it is
marked `slow` and can be skipped with: pytest -m "not slow"
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mymarket import config

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    """Import a numerically named script file as a module."""
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_news(n_per_product: int = 60) -> pd.DataFrame:
    """Kaggle shaped news with a learnable signal.

    Articles that carry a shortage word land on dates where the index rises,
    and subsidy articles land where it falls. A model that cannot beat the
    majority baseline here is broken.
    """
    rows = []
    start = pd.Timestamp("2023-01-01")

    for product, word in (("rice", "beras"), ("chicken", "ayam")):
        for i in range(n_per_product):
            date = start + pd.Timedelta(days=i * 7)
            rising = (i // 5) % 2 == 0
            event = "shortage" if rising else "subsidy"
            headline = (
                f"{word} prices under pressure as {event} continues in week {i}"
                if rising
                else f"{word} prices ease after {event} announced in week {i}"
            )
            rows.append(
                {
                    "content_id": f"{product}-{i}",
                    "title": headline,
                    "text": (
                        f"The retail price of {word} is in focus. Traders said {event} "
                        f"is the main driver this week. Officials are monitoring supply "
                        f"and cost across the market. Report number {i}."
                    ),
                    "section": "business",
                    "category": "news",
                    "content_tier": "Complimentary",
                    "content_length": 200,
                    "authors": "reporter",
                    "published_date": date.strftime("%Y-%m-%d"),
                    "keywords": event,
                    "summary": f"{word} price news, {event}.",
                    "url": f"https://example.my/{product}/{i}",
                }
            )
    return pd.DataFrame(rows)


def synthetic_index() -> pd.DataFrame:
    """A price index that actually moves, in blocks that match the news."""
    dates = pd.date_range("2022-12-01", periods=600, freq="D")
    frames = []
    for product, amplitude in (("rice", 0.12), ("chicken", 0.10)):
        block = ((np.arange(len(dates)) // 35) % 2)
        level = 1.0 + amplitude * block + np.linspace(0, 0.02, len(dates))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "product": product,
                    "index": level,
                    "n_items": 3,
                    "n_obs": 3,
                    "n_premises": 60,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every config path at a temporary directory."""
    processed = tmp_path / "processed"
    raw_news = tmp_path / "raw" / "news"
    models = tmp_path / "models"
    for path in (processed, raw_news, models):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "DATA_PROCESSED", processed)
    monkeypatch.setattr(config, "NEWS_RAW", raw_news)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "NEWS_CLEAN", processed / "news_clean.parquet")
    monkeypatch.setattr(config, "EXAMPLES", processed / "examples.parquet")
    monkeypatch.setattr(config, "PRICE_INDEX", processed / "price_index.parquet")
    monkeypatch.setattr(config, "MODEL_FILE", models / "model.joblib")
    monkeypatch.setattr(config, "LABEL_MAP_FILE", models / "label_map.json")
    monkeypatch.setattr(config, "THRESHOLDS_FILE", models / "thresholds.json")
    monkeypatch.setattr(config, "METRICS_FILE", models / "metrics.json")
    return tmp_path


@pytest.mark.slow
def test_offline_pipeline_produces_a_usable_model(sandbox, monkeypatch):
    prepare = load_script("02_prepare_news.py")
    build = load_script("05_build_examples.py")
    train = load_script("06_train_local.py")

    for module in (prepare, build, train):
        monkeypatch.setattr(module, "config", config)
    monkeypatch.setattr(train, "EMB_CACHE", config.DATA_PROCESSED / "embeddings.npy")

    # 1. clean the news
    cleaned = prepare.clean(synthetic_news())
    assert len(cleaned) > 100
    cleaned.to_parquet(config.NEWS_CLEAN, index=False)

    index = synthetic_index()
    index.to_parquet(config.PRICE_INDEX, index=False)

    # 2. filter to (article, product) candidates. The alias stage alone is used
    #    here: the embedding stage is quality, not wiring, and is tested apart.
    candidates = build.candidates(cleaned, use_embeddings=False)
    assert not candidates.empty
    assert set(candidates["product"]) == {"rice", "chicken"}

    # 3. label and split
    from mymarket.model import labels as lb
    from mymarket.model.features import event_frame

    labelled = lb.build_labels(candidates, index, horizon=14, k=0.5)
    labelled = labelled[labelled["label"].notna()].reset_index(drop=True)
    labelled = labelled.join(event_frame(labelled["text"].fillna("")))
    assert len(labelled) > 80
    assert labelled["label"].nunique() >= 2, "synthetic index must produce real moves"

    proof = lb.purge_proof(labelled, 14)
    assert proof["ok"], proof

    labelled.to_parquet(config.EXAMPLES, index=False)

    # 4. train
    assert train.main([]) == 0

    # 5. the four model files exist and say sane things
    import json

    import joblib

    for path in (
        config.MODEL_FILE,
        config.LABEL_MAP_FILE,
        config.THRESHOLDS_FILE,
        config.METRICS_FILE,
    ):
        assert path.exists(), f"{path.name} was not written"

    bundle = joblib.load(config.MODEL_FILE)
    assert {"model", "ohe", "version"} <= set(bundle)

    metrics = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
    assert 0.0 <= metrics["test_macro_f1"] <= 1.0
    assert set(metrics["baselines_test"]) == {"majority", "momentum", "keyword"}
    low, high = metrics["macro_f1_ci95"]
    assert low <= metrics["test_macro_f1"] <= high

    # 6. the saved bundle serves a prediction through the real inference path
    from mymarket.model import predict as predictor

    predictor._bundle.cache_clear()
    out = predictor.predict("Beras prices under pressure as shortage continues", products=["rice"])
    assert out[0].product == "rice"
    assert set(out[0].probs) <= {"UP", "DOWN", "STABLE"}
    assert 0.0 <= out[0].confidence <= 1.0
    predictor._bundle.cache_clear()


@pytest.mark.slow
def test_merged_product_model_serves_the_same_features_it_trained_on(sandbox, monkeypatch):
    """A model trained with products merged must be served with them merged.

    If serving passed the real product name to an encoder that only ever saw
    `all_staples`, the product block would silently become all zeros and the
    model would be fed features it never trained on.
    """
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    from mymarket.model import predict as predictor
    from mymarket.model.features import MERGED_LABEL, PRODUCT_COLUMN, build_matrix

    frame = pd.DataFrame(
        {
            "product": ["rice", "chicken", "egg", "rice"] * 5,
            PRODUCT_COLUMN: [MERGED_LABEL] * 20,
            "text": ["rice prices rise after export ban"] * 20,
        }
    )
    vectors = np.random.default_rng(0).normal(size=(20, 384)).astype("float32")

    encoder = OneHotEncoder(handle_unknown="ignore")
    matrix = build_matrix(frame, vectors, encoder, fit=True)
    assert encoder.categories_[0].tolist() == [MERGED_LABEL]

    model = LogisticRegression(max_iter=200)
    model.fit(matrix, ["UP", "DOWN"] * 10)

    joblib.dump(
        {"model": model, "ohe": encoder, "version": "test-merged", "merged_products": True},
        config.MODEL_FILE,
    )

    predictor._bundle.cache_clear()
    monkeypatch.setattr(predictor, "_embed", lambda texts: np.zeros((len(texts), 384), dtype="float32"))

    out = predictor.predict("Rice prices rise", products=["rice", "chicken"])

    assert [p.product for p in out] == ["rice", "chicken"]
    assert all(p.probs for p in out)
    predictor._bundle.cache_clear()
