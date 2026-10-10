"""Clean the Kaggle news dump, sample it for audit, and tune the filter threshold.

Usage:
    python scripts/02_prepare_news.py                # clean and write news_clean.parquet
    python scripts/02_prepare_news.py --audit 120    # write an audit sample to label
    python scripts/02_prepare_news.py --tune-filter  # read the labelled sample, pick a threshold

Manual step M1 must be done first: the Kaggle files go in data/raw/news/.
Manual step M5 is the labelling of the audit sample.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mymarket import config  # noqa: E402
from mymarket.filter import embed_filter  # noqa: E402
from mymarket.filter.aliases import build_text, match_all  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger("prepare_news")

EXPECTED_COLUMNS = {
    "content_id", "title", "text", "section", "category", "content_tier",
    "content_length", "authors", "published_date", "keywords", "summary", "url",
}
REQUIRED_COLUMNS = {"title", "text", "published_date", "url"}
START_DATE = "2020-01-01"
CHUNK_ROWS = 50_000


def read_raw() -> pd.DataFrame:
    """Read every Kaggle file in data/raw/news, CSV in chunks, parquet whole."""
    files = sorted(
        p for p in config.NEWS_RAW.glob("*")
        if p.suffix.lower() in {".csv", ".parquet", ".json", ".jsonl"}
    )
    if not files:
        raise FileNotFoundError(
            f"no news files in {config.NEWS_RAW}. Do manual step M1: download "
            "kaggle.com/datasets/azraimohamad/news-article-weekly-updated and "
            "unzip it there."
        )

    frames: list[pd.DataFrame] = []
    for path in files:
        log.info("reading %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
        if path.suffix.lower() == ".parquet":
            frames.append(pd.read_parquet(path))
        elif path.suffix.lower() in {".json", ".jsonl"}:
            frames.append(pd.read_json(path, lines=path.suffix.lower() == ".jsonl"))
        else:
            # Chunked so a large export does not need to fit in memory at once.
            parts = [c for c in pd.read_csv(path, chunksize=CHUNK_ROWS, low_memory=False)]
            frames.append(pd.concat(parts, ignore_index=True))

    combined = pd.concat(frames, ignore_index=True)
    log.info("read %d raw rows from %d file(s)", len(combined), len(files))
    return combined


def validate(frame: pd.DataFrame) -> None:
    """Check the expected Kaggle schema, warning on extras and failing on essentials."""
    present = set(frame.columns)
    missing_required = REQUIRED_COLUMNS - present
    if missing_required:
        raise ValueError(
            f"news files are missing required columns {sorted(missing_required)}. "
            f"Found {sorted(present)}."
        )
    missing_optional = EXPECTED_COLUMNS - present
    if missing_optional:
        log.warning("news files are missing optional columns %s", sorted(missing_optional))


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, drop empties and duplicates, keep 2020 onward."""
    out = frame.copy()
    validate(out)

    out["published_date"] = pd.to_datetime(out["published_date"], errors="coerce", utc=True)
    out = out.dropna(subset=["published_date"])
    out["published_date"] = out["published_date"].dt.tz_convert(config.TZ)
    out = out[out["published_date"] >= pd.Timestamp(START_DATE, tz=config.TZ)]

    for column in ("title", "text", "summary"):
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].fillna("").astype(str)

    out = out[out["text"].str.strip().str.len() > 0]
    out = out.drop_duplicates(subset=["url"])

    # Near duplicate titles: same date and same normalised title is a reprint.
    key = out["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    out = out.loc[~pd.Series(list(zip(out["published_date"].dt.date, key))).duplicated().to_numpy()]

    out["date"] = out["published_date"].dt.tz_localize(None).dt.normalize()
    return normalise_dtypes(out.sort_values("date", ignore_index=True))


def normalise_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Force mixed object columns to string so parquet can store them.

    Reading a large CSV in chunks lets pandas infer a different dtype per
    chunk, so a column such as content_id arrives as int in one chunk and str
    in another. Parquet needs one type per column.
    """
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype != "object":
            continue
        kinds = {type(v) for v in out[column].dropna().head(10_000)}
        if len(kinds) > 1 or kinds == {bytes}:
            out[column] = out[column].astype(str)
    return out


def report(frame: pd.DataFrame) -> None:
    """Print what came out, so the numbers go in the README."""
    print(f"\nrows: {len(frame)}")
    print(f"date range: {frame['date'].min().date()} to {frame['date'].max().date()}")
    domains = frame["url"].str.extract(r"https?://([^/]+)")[0].value_counts().head(8)
    print("\ntop domains:")
    print(domains.to_string())
    print("\nrows per year:")
    print(frame["date"].dt.year.value_counts().sort_index().to_string())


def matched_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (article, product) that passes the alias stage."""
    rows = []
    for article in frame.itertuples(index=False):
        hits = match_all(article.title, getattr(article, "summary", ""), article.text)
        for hit in hits:
            rows.append(
                {
                    "url": article.url,
                    "date": article.date,
                    "product": hit.product,
                    "title": article.title,
                    "snippet": build_text(article.title, getattr(article, "summary", ""), article.text)[:500],
                    "matched_terms": ", ".join(hit.matched_terms),
                    "event_words": ", ".join(hit.event_words),
                }
            )
    return pd.DataFrame(rows)


def write_audit(frame: pd.DataFrame, n: int) -> None:
    """Write a stratified sample of alias matches for manual labelling (M5)."""
    candidates = matched_rows(frame)
    if candidates.empty:
        raise RuntimeError("no article matched any product alias. Check config/products.yaml")

    per_product = max(1, n // candidates["product"].nunique())
    sample = (
        candidates.groupby("product", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_product), random_state=0), include_groups=True)
        .reset_index(drop=True)
    )
    sample["relevant"] = ""  # the column you fill in with 1 or 0

    config.AUDIT_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(config.AUDIT_SAMPLE, index=False, encoding="utf-8-sig")

    print(f"\nwrote {config.AUDIT_SAMPLE} with {len(sample)} rows")
    print(sample["product"].value_counts().to_string())
    print(
        "\nManual step M5:\n"
        "  1. Open the file. Add 1 or 0 in the `relevant` column for every row.\n"
        "  2. 1 means the article is about that product's price, supply, cost,\n"
        "     subsidy, import, shortage or a policy affecting it.\n"
        "  3. 0 means the word is a brand, a recipe, a restaurant, a sports or\n"
        "     entertainment story, or any other unrelated sense.\n"
        "  4. Save as CSV UTF-8 in the same place with the same name.\n"
        "  5. Run: python scripts/02_prepare_news.py --tune-filter\n"
    )


def tune_filter(target_precision: float = 0.80) -> int:
    """Pick the embedding threshold that reaches the target precision on the audit set."""
    if not config.AUDIT_SAMPLE.exists():
        log.error("%s not found. Run --audit N first.", config.AUDIT_SAMPLE)
        return 1

    sample = pd.read_csv(config.AUDIT_SAMPLE, encoding="utf-8-sig")
    if "relevant" not in sample.columns:
        log.error("the audit file has no `relevant` column")
        return 1

    labelled = sample[sample["relevant"].astype(str).str.strip().isin({"0", "1"})].copy()
    if len(labelled) < 20:
        log.error(
            "only %d rows are labelled. Label at least 20, ideally 120, then re-run.",
            len(labelled),
        )
        return 1
    labelled["relevant"] = labelled["relevant"].astype(float).astype(int)

    log.info("scoring %d labelled rows with the embedding filter", len(labelled))
    labelled["score"] = [
        embed_filter.score(str(row.snippet), str(row.product))
        for row in labelled.itertuples(index=False)
    ]

    print(f"\nlabelled rows: {len(labelled)}, relevant: {int(labelled['relevant'].sum())}")
    print(f"alias only precision: {labelled['relevant'].mean():.3f}")
    print("\nthreshold   kept  precision  recall")

    best = None
    for threshold in [round(x / 200, 3) for x in range(0, 41)]:
        kept = labelled[labelled["score"] >= threshold]
        if kept.empty:
            continue
        precision = kept["relevant"].mean()
        recall = kept["relevant"].sum() / max(1, labelled["relevant"].sum())
        print(f"{threshold:9.3f} {len(kept):6d} {precision:10.3f} {recall:7.3f}")
        if precision >= target_precision and (best is None or recall > best[2]):
            best = (threshold, precision, recall)

    if best is None:
        print(
            f"\nNo threshold reaches {target_precision:.0%} precision.\n"
            "Look at the rows you marked 0, find the repeated noise phrases, add\n"
            "them to exclude_phrases for that product in config/products.yaml, then\n"
            "re-run --audit with a fresh sample and label again."
        )
        return 1

    threshold, precision, recall = best
    print(
        f"\nbest threshold: {threshold} with precision {precision:.3f} and recall {recall:.3f}\n"
        f"Set this in config/products.yaml under filter.embed_threshold:\n"
        f"  embed_threshold: {threshold}\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=int, metavar="N", help="write an audit sample of N rows")
    parser.add_argument("--tune-filter", action="store_true", help="tune the threshold on the labelled sample")
    parser.add_argument("--target-precision", type=float, default=0.80)
    args = parser.parse_args(argv)

    config.ensure_dirs()

    if args.tune_filter:
        return tune_filter(args.target_precision)

    try:
        frame = clean(read_raw())
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    if frame.empty:
        log.error("no usable rows after cleaning")
        return 1

    frame.to_parquet(config.NEWS_CLEAN, index=False)
    log.info("wrote %s with %d rows", config.NEWS_CLEAN.name, len(frame))
    report(frame)

    if args.audit:
        write_audit(frame, args.audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
