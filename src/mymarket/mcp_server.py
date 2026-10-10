"""MyMarket MCP server. Eight tools over Malaysian food staple price signals.

Transport is stdio by default, so nothing may ever be written to stdout: a
stray print breaks the protocol framing. All logging goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import wraps
from typing import Any, Callable

# stdout belongs to the protocol. Configure logging before anything else imports.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)

# Malay and Chinese characters appear in Malaysian headlines, and a Windows
# console defaults to cp1252, which would raise on them.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

try:
    # mcp 2.x renamed FastMCP to MCPServer. The decorator and run APIs match.
    from mcp.server.mcpserver import MCPServer as _Server  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # noqa: E402

from mymarket import config  # noqa: E402
from mymarket.filter import relevance  # noqa: E402
from mymarket.filter.aliases import match_all as alias_match  # noqa: E402
from mymarket.ingest import fetch_body, rss  # noqa: E402
from mymarket.llm import explain as explainer  # noqa: E402
from mymarket.model import action  # noqa: E402
from mymarket.model import impact  # noqa: E402
from mymarket.model import predict as predictor  # noqa: E402
from mymarket.prices import index as price_index  # noqa: E402
from mymarket.store import db  # noqa: E402

log = logging.getLogger("mymarket.mcp")

mcp = _Server(
    "mymarket",
    instructions=(
        "MyMarket links Malaysian news to the price direction of four food "
        "staples: rice, egg, cooking oil and chicken. Call list_domains first, "
        "scan_alerts for the main view, and backtest_report before trusting any "
        "direction. Every alert is an awareness signal over a 14 day horizon, "
        "never a forecast and never financial advice."
    ),
)

HINTS = {
    price_index.PriceDataMissing: (
        "Run scripts/01_download_prices.py then scripts/03_build_price_index.py."
    ),
    predictor.ModelMissing: (
        "Train a model first: scripts/05_build_examples.py then "
        "scripts/06_train_local.py, or the Colab notebook at manual step M6."
    ),
    FileNotFoundError: "A required data file is missing. Check the README data steps.",
}


def tool_errors(func: Callable) -> Callable:
    """Turn any exception into the documented error shape (PRD 7).

    An MCP client must never see a raw traceback, and the server must never
    die because one tool call failed.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - this is the boundary
            hint = next(
                (h for kind, h in HINTS.items() if isinstance(exc, kind)),
                None,
            )
            if hint is None:
                # Genuinely unexpected: keep the traceback for debugging.
                log.exception("tool %s failed", func.__name__)
                hint = "Check the server log on stderr for the full error."
            else:
                # A known, documented state such as "no model trained yet".
                log.warning("tool %s unavailable: %s", func.__name__, exc)
            return {"error": f"{type(exc).__name__}: {exc}", "hint": hint}

    return wrapper


def _price_contexts(products: list[str], days: int = 90) -> dict[str, dict[str, Any]]:
    """Price context per product, skipping any the index does not cover."""
    out: dict[str, dict[str, Any]] = {}
    for product in products:
        try:
            out[product] = price_index.context(product, days=days).to_dict()
        except price_index.PriceDataMissing:
            continue
    return out


def _scan(hours: int, limit: int) -> tuple[int, list[tuple[dict[str, Any], list[relevance.Match]]]]:
    """Fetch, shortlist, enrich only the shortlist, then filter properly.

    The cheap alias stage runs first so that article bodies are only fetched
    for articles that could plausibly matter. That keeps a scan to a handful of
    requests instead of one per headline, which is the polite thing to do and
    also the fast one.
    """
    # Fetch everything the feeds currently hold and remember it, then read the
    # window back out of the database. A feed only carries its newest items, so
    # without this step `hours` could never reach further back than the feed.
    fresh = rss.fetch_all()
    with db.session() as conn:
        db.save_articles(conn, fresh)
        rows = db.recent_articles(conn, hours=hours)

    articles = [dict(row) for row in rows]

    shortlist = [
        a for a in articles if alias_match(a.get("title", ""), a.get("summary", ""), a.get("body", ""))
    ]
    fetch_body.enrich(shortlist, limit=15)

    matched = relevance.match_many(shortlist)
    return len(articles), matched[:limit]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
@tool_errors
def list_domains() -> dict[str, Any]:
    """List the tracked domain and its products, with aliases and price source.

    Call this first to learn what MyMarket can talk about.
    """
    cfg = config.settings()
    return {
        "domains": [
            {
                "key": config.DOMAIN,
                "label": "Food Staples",
                "price_source": "PriceCatcher (KPDN via data.gov.my), daily",
                "products": [
                    {
                        "key": p.key,
                        "label": p.label,
                        "aliases": p.aliases,
                        "exclude_phrases": p.exclude_phrases,
                        "mapped_item_codes": len(p.item_codes),
                    }
                    for p in cfg.products.values()
                ],
            }
        ],
        "horizon_days": cfg.horizon_days,
        "event_words": cfg.event_words,
        "note": "Consumer electronics was removed in v0.3. Food staples only.",
    }


@mcp.tool()
@tool_errors
def fetch_news(hours: int = 48, sources: list[str] | None = None, limit: int = 100) -> dict[str, Any]:
    """Fetch recent Malaysian news headlines from the configured RSS feeds.

    hours: how far back to look. sources: optional feed names from
    config/sources.yaml. Returns article id, title, source, published_at, url
    and summary.
    """
    articles = rss.fetch_all(hours=hours, sources=sources)[:limit]
    return {
        "count": len(articles),
        "hours": hours,
        "feeds": [f["name"] for f in config.enabled_feeds()],
        "articles": [
            {
                "id": a["id"],
                "title": a["title"],
                "source": a["source"],
                "published_at": a["published_at"],
                "url": a["url"],
                "summary": a["summary"][:400],
                "date_estimated": bool(a["date_estimated"]),
            }
            for a in articles
        ],
    }


@mcp.tool()
@tool_errors
def filter_relevant(domain: str | None = None, hours: int = 48, limit: int = 100) -> dict[str, Any]:
    """Find recent articles that are genuinely about a tracked food staple price.

    Runs a word boundary alias match, then an embedding check that drops
    articles which only mention the word in passing, such as a recipe.
    """
    if domain and domain != config.DOMAIN:
        return {
            "error": f"unknown domain {domain!r}",
            "hint": f"the only domain is {config.DOMAIN!r}. Call list_domains.",
        }

    scanned, matched = _scan(hours, limit)

    return {
        "scanned": scanned,
        "matched": len(matched),
        "hours": hours,
        "articles": [
            {
                "id": a["id"],
                "title": a["title"],
                "source": a["source"],
                "published_at": a["published_at"],
                "url": a["url"],
                "products": [m.to_dict() for m in matches],
            }
            for a, matches in matched
        ],
    }


@mcp.tool()
@tool_errors
def predict_impact(
    url: str | None = None,
    text: str | None = None,
    title: str | None = None,
    products: list[str] | None = None,
) -> dict[str, Any]:
    """Predict the 14 day price direction for one article, per product.

    Give either a url or some text. Returns UP, DOWN or STABLE with calibrated
    probabilities, the trigger words found, and whether the model abstained
    because it was not confident enough.
    """
    if not url and not text and not title:
        return {"error": "nothing to predict on", "hint": "pass url, or text, or title."}

    article: dict[str, Any] = {"title": title or "", "summary": "", "body": text or "", "url": url}

    if url and not text:
        body = fetch_body.fetch(url)
        article["body"] = body.text
        article["body_status"] = body.status
        if not body.usable and not title:
            return {
                "error": f"could not read the article body ({body.status})",
                "hint": (
                    "The site may block automated access or require a subscription. "
                    "Pass the headline as `title` or the text as `text` instead."
                ),
            }
        if not article["title"]:
            article["title"] = body.text.strip().splitlines()[0][:200] if body.text else url

    wanted = products
    if not wanted:
        matches = relevance.match_article(article["title"], article["summary"], article["body"])
        wanted = [m.product for m in matches]
        if not wanted:
            return {
                "article": {"title": article["title"], "url": url},
                "products": [],
                "note": "No tracked food staple is discussed in this article.",
            }

    predictions = predictor.predict(article["title"], article["summary"], article["body"], wanted)
    return {
        "article": {"title": article["title"], "url": url},
        "horizon_days": predictor.trained_horizon_days() or config.settings().horizon_days,
        "model_version": predictor.model_version(),
        "products": [p.to_dict() for p in predictions],
    }


@mcp.tool()
@tool_errors
def price_context(product: str, days: int = 90) -> dict[str, Any]:
    """Show the recent official price trend for one food staple.

    Returns a normalised daily index, the 7 and 30 day percent change, and the
    tail of the series. The index is unitless, not ringgit.
    """
    cfg = config.settings()
    if product not in cfg.products:
        return {
            "error": f"unknown product {product!r}",
            "hint": f"known products are {', '.join(cfg.product_keys)}.",
        }
    return price_index.context(product, days=days).to_dict()


@mcp.tool()
@tool_errors
def scan_alerts(
    domain: str | None = None,
    hours: int = 48,
    min_confidence: float = 0.55,
    lang: str = "en",
    limit: int = 50,
) -> dict[str, Any]:
    """Scan recent news and return price alerts for food staples.

    This is the main tool. It fetches news, keeps the articles that are really
    about a tracked product, predicts a direction, attaches the official price
    trend, and returns grounded facts you can explain to the reader.
    """
    if domain and domain != config.DOMAIN:
        return {
            "error": f"unknown domain {domain!r}",
            "hint": f"the only domain is {config.DOMAIN!r}.",
        }

    scanned, matched = _scan(hours, limit)
    contexts = _price_contexts(config.settings().product_keys)
    model_ready = predictor.available()

    alerts: list[dict[str, Any]] = []
    for article, matches in matched:
        products = [m.product for m in matches]

        if model_ready:
            predictions = [
                p.to_dict()
                for p in predictor.predict(
                    article["title"],
                    article["summary"],
                    article["body"],
                    products,
                    min_confidence=min_confidence,
                )
            ]
        else:
            predictions = [
                {
                    "product": m.product,
                    "direction": "UNVALIDATED",
                    "probs": {},
                    "confidence": None,
                    "abstained": True,
                    "triggers": m.event_words,
                    "model_version": "none",
                    "impact": "NOT ASSESSED",
                    "meaning": "no trained model, keyword and price context match only",
                    "confidence_band": "none",
                    "shopper_note": predictor.shopper_note("UNVALIDATED"),
                    "note": "No trained model yet. This is a keyword and price context alert only.",
                }
                for m in matches
            ]

        relevant_context = {p: contexts[p] for p in products if p in contexts}

        # Rank and recommend. A direction with a probability is not a decision.
        for prediction in predictions:
            rec = action.recommend(
                prediction["product"],
                prediction.get("direction", "UNCERTAIN"),
                prediction.get("confidence"),
                article.get("published_at"),
                relevant_context.get(prediction["product"]),
            )
            prediction.update(rec.to_dict())

        max_urgency = max((p.get("urgency_score", 0) for p in predictions), default=0)
        top = max(predictions, key=lambda p: p.get("urgency_score", 0), default=None)

        alerts.append(
            {
                "max_urgency": max_urgency,
                "priority": top["priority"] if top else "NO ACTION",
                "headline_action": top["recommended_action"] if top else None,
                "article": {
                    "title": article["title"],
                    "source": article["source"],
                    "url": article["url"],
                    "published_at": article["published_at"],
                    "date_estimated": bool(article["date_estimated"]),
                },
                "products": predictions,
                "matched_terms": {m.product: m.matched_terms for m in matches},
                "price_context": relevant_context,
                "explanation": explainer.explain(
                    {**article, "horizon_days": config.settings().horizon_days},
                    predictions,
                    relevant_context,
                    lang=lang,
                ),
            }
        )

    alerts = action.rank(alerts)

    moving = sorted({
        p["product"]
        for a in alerts
        for p in a["products"]
        if p.get("direction") in {"UP", "DOWN"}
    })
    act_now = [a for a in alerts if a["priority"] == "ACT NOW"]
    watch = [a for a in alerts if a["priority"] == "WATCH"]

    return {
        "scanned": scanned,
        "alerts": len(alerts),
        "hours": hours,
        "model_trained": model_ready,
        "action_summary": {
            "act_now": len(act_now),
            "watch": len(watch),
            "top_priority": alerts[0]["priority"] if alerts else "NO ACTION",
            "what_to_do_first": alerts[0]["headline_action"] if alerts else
                "Nothing to act on. No recent news points to a staple price move.",
            "alerts_are_sorted_by": "urgency, most urgent first",
        },
        "domain_summary": {
            "domain": config.DOMAIN,
            "products_with_a_predicted_move": moving,
            "note": (
                "Awareness signal over a 14 day horizon, not a forecast."
                if model_ready
                else "No trained model is loaded, so no direction is predicted. "
                "These are keyword matches with official price context."
            ),
        },
        "disclaimer": config.DISCLAIMER,
        "alerts_list": alerts,
    }


@mcp.tool()
@tool_errors
def backtest_report(product: str | None = None) -> dict[str, Any]:
    """Report how accurate the model was on the held out test set.

    Includes the baselines it must beat (majority class, price momentum,
    keyword rules) and a 95 percent confidence interval. Read this before
    trusting any alert.
    """
    if not config.METRICS_FILE.exists():
        return {
            "error": "models/metrics.json not found",
            "hint": (
                "No model has been evaluated yet. Run scripts/05_build_examples.py, "
                "train, then scripts/07_evaluate.py."
            ),
        }

    metrics = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))

    # Metrics are computed and stored on the raw price directions, because
    # changing them would invalidate every comparison. Translate them on the
    # way out so a reader sees one vocabulary everywhere, not two.
    metrics = dict(metrics)
    metrics["label_legend"] = {
        impact.IMPACT[d]["impact"]: impact.IMPACT[d]["meaning"]
        for d in ("UP", "DOWN", "STABLE")
    }
    if isinstance(metrics.get("per_product"), dict):
        translated = {}
        for name, stats in metrics["per_product"].items():
            stats = dict(stats)
            counts = stats.pop("label_counts", {}) or {}
            stats["outcome_counts"] = {
                impact.IMPACT.get(k, {}).get("impact", k): v for k, v in counts.items()
            }
            translated[name] = stats
        metrics["per_product"] = translated

    if product:
        per_product = (metrics.get("per_product") or {}).get(product)
        if per_product is None:
            return {
                "error": f"no metrics for product {product!r}",
                "hint": f"products in this report: {', '.join((metrics.get('per_product') or {}))}",
            }
        return {"product": product, "metrics": per_product, "generated_at": metrics.get("generated_at")}
    return metrics


@mcp.tool()
@tool_errors
def forward_validation(product: str | None = None) -> dict[str, Any]:
    """Report how live predictions have scored against real prices since launch.

    Each prediction is logged when made and scored 14 days later against the
    official price index. Early on there will be few matured rows.
    """
    with db.session() as conn:
        rows = [dict(r) for r in db.scored_predictions(conn, product)]
        pending = len(db.unscored_predictions(conn))

    if not rows:
        return {
            "matured": 0,
            "pending": pending,
            "note": (
                "No prediction has matured yet. Each one needs "
                f"{config.settings().horizon_days} days before it can be scored. "
                "Schedule scripts/daily_refresh.py (manual step M8) to build this up."
            ),
        }

    correct = sum(1 for r in rows if r["direction"] == r["realized_label"])
    by_product: dict[str, dict[str, int]] = {}
    for row in rows:
        stats = by_product.setdefault(row["product"], {"n": 0, "correct": 0})
        stats["n"] += 1
        stats["correct"] += int(row["direction"] == row["realized_label"])

    return {
        "matured": len(rows),
        "pending": pending,
        "running_accuracy": round(correct / len(rows), 4),
        "per_product": {
            k: {**v, "accuracy": round(v["correct"] / v["n"], 4)} for k, v in by_product.items()
        },
        "caution": (
            "A small number of matured predictions says very little. "
            "Read this alongside backtest_report."
        ),
        "recent": rows[:20],
    }


@mcp.tool()
@tool_errors
def demo_cases(right: int = 2, wrong: int = 1) -> dict[str, Any]:
    """Show past predictions whose real price outcome is already known.

    Use this to check whether the model is actually any good. Live news takes
    28 days to verify, so these come from the held out test set instead: real
    headlines, what the model said at the time, and the official PriceCatcher
    move that followed. Incorrect cases are included on purpose.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "demo_cases_script", config.ROOT / "scripts" / "08_demo_cases.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cases = module.build(right, wrong)
    for case in cases:
        reader = impact.describe(case["predicted"], case["confidence"])
        case["model_said"] = reader["impact"]
        case["meaning"] = reader["meaning"]
        case["confidence_band"] = reader["confidence_band"]
        case["what_actually_happened"] = impact.describe(case["actual"])["impact"]

        # Show the recommendation the user would have been given at the time,
        # so the demo can ask whether following it would have been right.
        rec = action.recommend(
            case["product"],
            case["predicted"],
            case["confidence"],
            case["published"],
            {"change_7d_pct": None},
        )
        case["priority_at_the_time"] = rec.priority
        case["advice_at_the_time"] = rec.action
        case["would_have_been"] = (
            "good advice" if case["predicted"] == case["actual"] else "bad advice"
        )

    return {
        "note": (
            "Held out test cases. The model never saw these during training, and "
            "the real price move is already known, so each call can be checked."
        ),
        "horizon_days": cases[0]["horizon_days"] if cases else None,
        "cases": cases,
        "disclaimer": config.DISCLAIMER,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MyMarket MCP server")
    parser.add_argument("--http", action="store_true", help="serve over streamable HTTP")
    args = parser.parse_args()

    log.info("starting MyMarket MCP server, model_trained=%s", predictor.available())
    mcp.run(transport="streamable-http" if args.http else "stdio")


if __name__ == "__main__":
    main()
