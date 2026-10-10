"""Download and aggregate PriceCatcher monthly files.

Raw monthly files hold over a million rows each (PRD F10). Nothing here is ever
called from the MCP server: it reduces each month to a small daily table and
deletes the raw file, so the server only ever reads the processed parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mymarket import config

log = logging.getLogger(__name__)

RECORD_COLUMNS = {"date", "premise_code", "item_code", "price"}
ITEM_COLUMNS = {"item_code", "item", "unit", "item_group", "item_category"}
PREMISE_COLUMNS = {"premise_code", "premise", "premise_type", "state", "district"}

DAILY_SHARDS = config.DATA_PROCESSED / "daily"


class SchemaError(RuntimeError):
    """A published file no longer has the columns the pipeline expects."""


def month_url(month: str) -> str:
    """Parquet URL for a YYYY-MM month."""
    return f"{config.PRICECATCHER_BASE}/pricecatcher_{month}.parquet"


def lookup_url(name: str) -> str:
    """Parquet URL for a lookup table, name being item or premise."""
    return f"{config.PRICECATCHER_BASE}/lookup_{name}.parquet"


def months_range(start: str, end: str) -> Iterator[str]:
    """Yield YYYY-MM strings from start to end inclusive."""
    cur = pd.Period(start, freq="M")
    last = pd.Period(end, freq="M")
    while cur <= last:
        yield str(cur)
        cur += 1


def current_month() -> str:
    """This month as YYYY-MM in the project timezone."""
    return pd.Timestamp.now(tz=config.TZ).strftime("%Y-%m")


def _client(timeout: float = 180.0) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "MyMarket/0.3 (research project)"},
    )


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)
def download(url: str, dest: Path) -> int:
    """Stream a URL to dest and return bytes written. Retries transport errors.

    A 404 raises FileNotFoundError and is not retried: it means the month does
    not exist, which the caller treats as a skip rather than a failure (PRD F3).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    written = 0
    with _client() as client, client.stream("GET", url) as resp:
        if resp.status_code == 404:
            raise FileNotFoundError(url)
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)
                written += len(chunk)
    tmp.replace(dest)
    return written


def month_exists(month: str) -> bool:
    """HEAD probe for one month. False on 404, True on 200."""
    try:
        with _client(timeout=30.0) as client:
            resp = client.head(month_url(month))
        if resp.status_code == 405:  # server refuses HEAD, fall back to a ranged GET
            with _client(timeout=30.0) as client:
                resp = client.get(month_url(month), headers={"Range": "bytes=0-0"})
        return resp.status_code < 400
    except httpx.HTTPError as exc:
        log.warning("probe failed for %s: %s", month, exc)
        return False


def check_columns(df: pd.DataFrame, expected: set[str], label: str) -> None:
    """Raise a clear error when a published schema changes (PRD risk table)."""
    missing = expected - set(df.columns)
    if missing:
        raise SchemaError(
            f"{label} is missing expected columns {sorted(missing)}. "
            f"Found {sorted(df.columns)}. "
            "Check data.gov.my for a schema change and update the pipeline."
        )


def aggregate_month(df: pd.DataFrame, item_codes: list[int] | None = None) -> pd.DataFrame:
    """Reduce raw monthly records to one row per (date, item_code).

    Keeps the median, the quartiles and the premise count so the index builder
    can judge how thin a day is without the raw file.
    """
    check_columns(df, RECORD_COLUMNS, "pricecatcher monthly file")

    out = df.loc[:, ["date", "premise_code", "item_code", "price"]].copy()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date", "price", "item_code"])
    out = out[out["price"] > 0]

    if item_codes:
        out = out[out["item_code"].isin(item_codes)]

    if out.empty:
        return pd.DataFrame(
            columns=["date", "item_code", "median", "p25", "p75", "n_premises"]
        )

    grouped = out.groupby(["date", "item_code"])["price"]
    agg = grouped.agg(
        median="median",
        p25=lambda s: s.quantile(0.25),
        p75=lambda s: s.quantile(0.75),
        n_premises="count",
    ).reset_index()
    agg["item_code"] = agg["item_code"].astype("int64")
    return agg.sort_values(["date", "item_code"], ignore_index=True)


def shard_path(month: str) -> Path:
    """Where the daily aggregate for one month is stored."""
    return DAILY_SHARDS / f"daily_{month}.parquet"


def processed_months() -> set[str]:
    """Months already aggregated, used to resume an interrupted run."""
    if not DAILY_SHARDS.exists():
        return set()
    return {p.stem.replace("daily_", "") for p in DAILY_SHARDS.glob("daily_*.parquet")}


def combine_shards() -> pd.DataFrame:
    """Concatenate every monthly shard into the single daily price table."""
    shards = sorted(DAILY_SHARDS.glob("daily_*.parquet"))
    if not shards:
        return pd.DataFrame(
            columns=["date", "item_code", "median", "p25", "p75", "n_premises"]
        )
    frames = [pd.read_parquet(p) for p in shards]
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["date", "item_code"], ignore_index=True)
