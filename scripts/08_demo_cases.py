"""Pick demo cases from the test set, where the real outcome is already known.

Usage:
    python scripts/08_demo_cases.py
    python scripts/08_demo_cases.py --right 3 --wrong 2

Live news is the honest way to show the product, but on any given day it may
contain nothing about rice, eggs, cooking oil or chicken, and nothing at all
can be verified on the spot because the outcome is 28 days away.

These cases come from the held out test set, so for every one of them the real
price move is already known and the prediction can be marked right or wrong in
front of an audience. Showing a wrong one is the point, not an accident: it is
the most credible thing a demo can do.

Writes data/processed/demo_cases.json and prints a readable script.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mymarket import config  # noqa: E402
from mymarket.model.features import build_matrix  # noqa: E402
from mymarket.prices import index as price_index  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger("demo_cases")


def price_window(product: str, when: pd.Timestamp, horizon: int) -> dict:
    """The actual before and after index values an audience can be shown."""
    try:
        series = price_index.series_for(product)
    except price_index.PriceDataMissing:
        return {}
    before = series[series.index <= when - pd.Timedelta(days=1)]
    after = series[series.index <= when + pd.Timedelta(days=horizon)]
    if before.empty or after.empty:
        return {}
    return {
        "index_before": round(float(before.iloc[-1]), 4),
        "index_after": round(float(after.iloc[-1]), 4),
        "measured_on": str(before.index[-1].date()),
        "outcome_on": str(after.index[-1].date()),
    }


def build(right: int, wrong: int) -> list[dict]:
    """Score the test set and pick the clearest right and wrong cases."""
    import joblib

    bundle = joblib.load(config.MODEL_FILE)
    frame = pd.read_parquet(config.EXAMPLES)
    frame = frame[frame["split"].isin({"train", "val", "test"})].reset_index(drop=True)

    cache = config.DATA_PROCESSED / "embeddings.npy"
    if cache.exists() and len(np.load(cache)) == len(frame):
        vectors = np.load(cache)
    else:
        from mymarket.filter import embed_filter

        vectors = embed_filter.encode(frame["text"].fillna("").tolist())

    matrix = build_matrix(frame, vectors, bundle["ohe"], fit=False)
    test = (frame["split"] == "test").to_numpy()

    scored = frame[test].copy()
    scored["predicted"] = bundle["model"].predict(matrix[test])
    scored["confidence"] = bundle["model"].predict_proba(matrix[test]).max(axis=1)
    scored["correct"] = scored["predicted"] == scored["label"]

    horizon = int(scored["horizon_days"].iloc[0])

    # Prefer confident calls on articles that actually moved: a correct STABLE
    # on a quiet week is true but says nothing to an audience.
    def pick(correct: bool, n: int) -> pd.DataFrame:
        pool = scored[scored["correct"] == correct]
        moved = pool[pool["label"] != "STABLE"]
        chosen = moved.nlargest(n, "confidence")
        if len(chosen) < n:
            chosen = pd.concat([chosen, pool.nlargest(n - len(chosen), "confidence")])
        return chosen.head(n)

    cases = []
    for correct, count in ((True, right), (False, wrong)):
        for row in pick(correct, count).itertuples(index=False):
            cases.append(
                {
                    "verdict": "model was RIGHT" if correct else "model was WRONG",
                    "product": row.product,
                    "published": str(row.date.date()),
                    "headline": row.title,
                    "url": row.url,
                    "matched_terms": row.matched_terms,
                    "price_cue_or_event_words": row.event_words,
                    "predicted": row.predicted,
                    "confidence": round(float(row.confidence), 3),
                    "actual": row.label,
                    "actual_move_pct": round(float(row.ret) * 100, 2),
                    "threshold_pct": round(float(row.theta) * 100, 2),
                    "horizon_days": horizon,
                    **price_window(row.product, row.date, horizon),
                }
            )
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--right", type=int, default=2, help="correct cases to show")
    parser.add_argument("--wrong", type=int, default=1, help="incorrect cases to show")
    args = parser.parse_args(argv)

    if not config.MODEL_FILE.exists() or not config.EXAMPLES.exists():
        log.error("need models/model.joblib and data/processed/examples.parquet first")
        return 1

    cases = build(args.right, args.wrong)
    if not cases:
        log.error("no test cases available")
        return 1

    out = config.DATA_PROCESSED / "demo_cases.json"
    out.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("DEMO CASES, outcome already known")
    print("=" * 72)
    for i, case in enumerate(cases, 1):
        print(f"\n{i}. {case['verdict']}   [{case['product']}]")
        print(f"   headline   {case['headline'][:70]}")
        print(f"   published  {case['published']}")
        print(f"   model said {case['predicted']} at {case['confidence']:.0%} confidence")
        print(f"   actually   {case['actual']}  ({case['actual_move_pct']:+.2f}% "
              f"over {case['horizon_days']} days, threshold {case['threshold_pct']:.2f}%)")
        if "index_before" in case:
            print(f"   price      {case['index_before']} on {case['measured_on']} "
                  f"-> {case['index_after']} on {case['outcome_on']}")

    print(f"\nwrote {out}")
    print(
        "\nHow to use these in the demo:\n"
        "  1. Read the headline out, then ask the room which way prices went.\n"
        "  2. Show what the model said and how confident it was.\n"
        "  3. Show the official PriceCatcher move that actually followed.\n"
        "  4. Do the wrong one last and do not skip it. A demo that only shows\n"
        "     wins invites the question you cannot answer.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
