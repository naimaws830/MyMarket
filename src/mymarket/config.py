"""Project paths and configuration, resolved by absolute path from the repo root."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

DOMAIN = "food_staples"

# src/mymarket/config.py -> src/mymarket -> src -> repo root
ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT / "config"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

NEWS_RAW = DATA_RAW / "news"
PRICES_RAW = DATA_RAW / "pricecatcher"

LOOKUP_ITEM = DATA_PROCESSED / "lookup_item.parquet"
LOOKUP_PREMISE = DATA_PROCESSED / "lookup_premise.parquet"
DAILY_PRICES = DATA_PROCESSED / "daily_prices.parquet"
PRICE_INDEX = DATA_PROCESSED / "price_index.parquet"
NEWS_CLEAN = DATA_PROCESSED / "news_clean.parquet"
EXAMPLES = DATA_PROCESSED / "examples.parquet"
AUDIT_SAMPLE = DATA_PROCESSED / "audit_sample.csv"
ITEM_MAPPING_REVIEW = DATA_PROCESSED / "item_mapping_review.csv"
PLOTS_DIR = DATA_PROCESSED / "plots"

ITEM_MAPPING = CONFIG_DIR / "item_mapping.csv"
PRODUCTS_YAML = CONFIG_DIR / "products.yaml"
SOURCES_YAML = CONFIG_DIR / "sources.yaml"

DB_PATH = DATA_PROCESSED / "mymarket.sqlite"

MODEL_FILE = MODELS_DIR / "model.joblib"
LABEL_MAP_FILE = MODELS_DIR / "label_map.json"
THRESHOLDS_FILE = MODELS_DIR / "thresholds.json"
METRICS_FILE = MODELS_DIR / "metrics.json"

TZ = "Asia/Kuala_Lumpur"
LABELS = ("UP", "DOWN", "STABLE")
EMBED_MODEL = "intfloat/multilingual-e5-small"
EMBED_PREFIX = "query: "

# PriceCatcher files start somewhere in 2022. The downloader probes from here.
PRICECATCHER_BASE = "https://storage.data.gov.my/pricecatcher"
PROBE_START = "2022-01"

DISCLAIMER = (
    "Awareness only. This is not financial advice and not a price forecast."
)


@dataclass(frozen=True)
class Product:
    """One tracked food staple."""

    key: str
    label: str
    aliases_en: list[str]
    aliases_ms: list[str]
    exclude_phrases: list[str]
    description_sentences: list[str]
    preferred_categories: list[str] = field(default_factory=list)
    reject_item_terms: list[str] = field(default_factory=list)
    item_codes: list[int] = field(default_factory=list)

    @property
    def aliases(self) -> list[str]:
        """All aliases, longest first so the longest match wins."""
        return sorted(self.aliases_en + self.aliases_ms, key=len, reverse=True)


@dataclass(frozen=True)
class Settings:
    """Parsed products.yaml plus env overrides."""

    products: dict[str, Product]
    event_words: list[str]
    horizon_days: int
    theta_k: float
    min_premises: int
    embed_threshold: float
    body_chars: int
    min_confidence: float
    distractor_sentences: list[str] = field(default_factory=list)
    requires_price_cue: bool = False
    price_cue_words: list[str] = field(default_factory=list)

    @property
    def product_keys(self) -> list[str]:
        return list(self.products)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def load_env() -> None:
    """Load .env from the repo root by absolute path, needed under Claude Desktop."""
    load_dotenv(ROOT / ".env", override=False)


def read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file, raising a clear error when it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def item_codes_by_product() -> dict[str, list[int]]:
    """Read config/item_mapping.csv if the M3 review is done, else return empty lists."""
    if not ITEM_MAPPING.exists():
        return {}

    import csv

    out: dict[str, list[int]] = {}
    with ITEM_MAPPING.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            product = (row.get("product") or "").strip()
            code = (row.get("item_code") or "").strip()
            if not product or not code:
                continue
            try:
                out.setdefault(product, []).append(int(float(code)))
            except ValueError:
                continue
    return {k: sorted(set(v)) for k, v in out.items()}


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Load and cache products.yaml merged with item_mapping.csv and env overrides."""
    load_env()
    raw = read_yaml(PRODUCTS_YAML)

    domain = (raw.get("domains") or {}).get(DOMAIN)
    if not domain:
        raise ValueError(f"products.yaml has no domains.{DOMAIN} section")

    mapped = item_codes_by_product()
    products: dict[str, Product] = {}
    for key, spec in (domain.get("products") or {}).items():
        products[key] = Product(
            key=key,
            label=spec.get("label", key),
            aliases_en=list(spec.get("aliases_en") or []),
            aliases_ms=list(spec.get("aliases_ms") or []),
            exclude_phrases=list(spec.get("exclude_phrases") or []),
            description_sentences=list(spec.get("description_sentences") or []),
            preferred_categories=list(spec.get("preferred_categories") or []),
            reject_item_terms=list(spec.get("reject_item_terms") or []),
            item_codes=mapped.get(key) or list(spec.get("item_codes") or []),
        )

    if not products:
        raise ValueError(f"products.yaml defines no products under {DOMAIN}")

    lab = raw.get("label") or {}
    flt = raw.get("filter") or {}

    return Settings(
        products=products,
        event_words=list(domain.get("event_words") or []),
        horizon_days=_env_int("MYMARKET_HORIZON_DAYS", int(lab.get("horizon_days", 14))),
        theta_k=_env_float("MYMARKET_THETA_K", float(lab.get("theta_k", 0.5))),
        min_premises=int(lab.get("min_premises", 5)),
        embed_threshold=float(flt.get("embed_threshold", 0.055)),
        body_chars=int(flt.get("body_chars", 1500)),
        min_confidence=_env_float("MYMARKET_MIN_CONFIDENCE", 0.55),
        distractor_sentences=list(flt.get("distractor_sentences") or []),
        requires_price_cue=bool(domain.get("requires_price_cue", False)),
        price_cue_words=list(domain.get("price_cue_words") or []),
    )


@lru_cache(maxsize=1)
def sources() -> dict[str, Any]:
    """Load and cache sources.yaml."""
    return read_yaml(SOURCES_YAML)


def enabled_feeds() -> list[dict[str, Any]]:
    """Feeds marked enabled in sources.yaml."""
    return [f for f in (sources().get("feeds") or []) if f.get("enabled")]


def fetch_opts() -> dict[str, Any]:
    """Fetch settings with safe defaults. Rate limit is never allowed below 1 second."""
    opts = dict(sources().get("fetch") or {})
    opts["rate_limit_seconds"] = max(1.0, float(opts.get("rate_limit_seconds", 1.0)))
    opts.setdefault("timeout_seconds", 20)
    opts.setdefault("user_agent", "MyMarket/0.3")
    opts.setdefault("min_body_chars", 300)
    opts.setdefault("respect_robots", True)
    return opts


def ensure_dirs() -> None:
    """Create the data and model directories if they are missing."""
    for path in (NEWS_RAW, PRICES_RAW, DATA_PROCESSED, MODELS_DIR, PLOTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
