# MyMarket: Product Requirements Document

Version 0.3. Status: hackathon MVP, backend only.

**v0.3 changes from v0.2:** renamed from HargaRadar to MyMarket. One domain only,
Food Staples (rice, egg, cooking oil, chicken). Consumer electronics is removed,
and with it the DOSM CPI pipeline, the monthly label path, the electronics price
cue rule, and flags F2, F14 and F15. Every product now has one price truth:
daily PriceCatcher.

One line: an MCP server that reads live Malaysian news, finds stories tied to
four food staples, predicts whether they point to price UP, DOWN or STABLE,
checks that call against official price data, and explains it in plain words.

---

## 0. Flags

Each is a known weakness with the mitigation that is actually built.

**F1. "Impact" is vague.** Positive for whom? Fix: direction only, UP, DOWN,
STABLE, plus a plain tag "bad for shoppers" (UP) or "good for shoppers" (DOWN).

**F3. News years and price years do not fully overlap.** PriceCatcher begins
2022-01 (verified by probe). Fix: only news inside the price window is labelled,
and `05_build_examples.py` reports how many articles that leaves.

**F4. News rarely moves grocery prices within two weeks.** The signal is weak.
Fix: compare against majority, momentum and keyword baselines, report macro F1
with a confidence interval, let the model abstain, and position the product as
an awareness alert rather than a forecast.

**F5. Policy moves staple prices in steps.** Subsidies and ceiling prices cause
step changes, not smooth moves. Fix: policy words are model features.

**F6. A random split would cheat.** One event produces many articles sharing one
label. Fix: split by time with a 14 day purge gap, plus a rolling origin
backtest by quarter. The proof is printed.

**F7. Heavy class imbalance.** Most windows are STABLE. Fix: class weights,
macro F1 as the headline metric, never plain accuracy alone.

**F8. Source mismatch.** Training news is mostly thestar.com.my; live news is
Malay Mail, Bernama and FMT. Fix: treat live results as an out of source test
and say so.

**F9. Scrapling caution.** Fix: plain `Fetcher` only, never a stealth fetcher,
robots.txt obeyed, 1 request per second, everything cached. In practice Malay
Mail and FMT disallow their article paths, so their bodies are never fetched.

**F10. PriceCatcher has no API.** Over a million rows a month. Fix: reduced
offline to a small daily table; the MCP server never opens a raw file.

**F11. The price index is noisy.** Premises change daily. Fix: outlier trim,
per item normalisation, 7 day smoothing.

**F12. An LLM could invent numbers.** Fix: the explainer receives only computed
facts under a strict schema, and by default the chat host writes the prose from
those facts while the server writes none.

**F13. Not financial advice.** Every alert carries a disclaimer.

**F16 (new). Price controls flatten the labels.** The egg index is flat until
mid 2024 because eggs were administratively priced. A flat training period
drives theta towards zero. Fix: a 0.5 percent floor (`MIN_THETA`), and any
product sitting on that floor is reported, because there the floor and not the
data decides the labels.

**F17 (new). Plain embedding similarity does not discriminate.** The e5 model
scores anything topical at 0.82 to 0.88, and "Best fried chicken restaurants"
outscores "Chicken prices rise as feed costs climb". Fix: stage two of the
filter is contrastive, product description similarity minus distractor
similarity.

**F18 (new). Survey cadence degrades over time.** By 2026 cooking oil items are
surveyed about weekly while eggs are near daily, which silently truncated two of
the four indexes. Fix: carry each item forward up to 10 days before averaging,
since a survey price is a step function.

---

## 1. Goal and non-goals

**Goals:** pull Malaysian news live; keep only articles tied to tracked products;
predict price direction; validate against official price data and report honest
accuracy; explain in simple sentences; expose it all as MCP tools.

**Non-goals:** no UI, no price level forecast, no trading signals, no paid APIs.

---

## 2. Domain and products

One domain, `food_staples`. Price truth PriceCatcher, daily.

| Product | EN aliases | BM aliases | Exclude phrases |
|---|---|---|---|
| rice | rice, local white rice, imported rice | beras, beras tempatan, padi | fried rice, chicken rice, rice cooker, rice bowl |
| egg | egg, eggs | telur | Easter egg, nest egg, egg roll, egg tart |
| cooking_oil | cooking oil, vegetable oil | minyak masak | engine oil, crude oil price, oil and gas |
| chicken | chicken, poultry, broiler | ayam | chicken rice, Ayam Brand, KFC, chicken chop |

Palm oil is an event word for cooking oil, not an alias: it is a crude price story.

Event words: subsidy, subsidi, ceiling price, harga siling, SHMMP, shortage,
kekurangan, flood, banjir, import ban, export ban, tariff, levy, SST, price
control, kawalan harga, festive, Hari Raya, Ramadan, Chinese New Year, ringgit,
drought, bird flu, feed cost, minimum wage, palm oil.

All of it lives in `config/products.yaml`.

---

## 3. Data sources

| Source | What | Access |
|---|---|---|
| Kaggle `azraimohamad/news-article-weekly-updated` | Training news | Login, manual step M1 |
| PriceCatcher records | Daily item prices per premise | `storage.data.gov.my/pricecatcher/pricecatcher_YYYY-MM.parquet` |
| PriceCatcher item lookup | item_code to name | `.../lookup_item.parquet`, 797 items |
| PriceCatcher premise lookup | premise_code to place | `.../lookup_premise.parquet` |
| Malay Mail RSS | Live news, full text in `content:encoded` | `malaymail.com/feed/rss/malaysia` |
| Bernama RSS | Live news, summary only, no pubDate | `bernama.com/en/rssfeed.php` |
| FMT RSS | Live news, 301 redirects to `cms.freemalaysiatoday.com/feed` | `freemalaysiatoday.com/feed/` |
| Scrapling | Article body, plain `Fetcher` only | `page.markdown()` |
| LLM explainer | Optional | Any OpenAI compatible endpoint, or none |

PriceCatcher is a price surveillance survey. For inflation analysis the right
source is DOSM CPI. Say this in the demo.

---

## 4. Functional requirements

### 4.1 News ingestion
Read every enabled feed with `feedparser`. Dedupe by URL, then by near duplicate
title. Use `content:encoded` when present rather than fetching. Fetch the body
with Scrapling only when robots.txt allows it and the feed gave none. A missing
`pubDate` means the fetch time is used and `date_estimated` is set. Rate limit 1
request per second per host. Cache in SQLite. Fall back to the RSS summary when
a body is under 300 characters.

### 4.2 Relevance filter
Stage 1: word boundary alias match on title, summary and the first 1500
characters of body. Exclude phrases are masked out of the text rather than used
to reject the whole article, so "chicken rice stalls hit as chicken supply
tightens" still matches.

Stage 2: contrastive embedding score with `intfloat/multilingual-e5-small`, the
best similarity to the product description sentences minus the best similarity
to shared distractor sentences. Dropped below the tuned threshold, which manual
step M5 sets from labelled data.

### 4.3 Impact model
Input: `title. summary. first 1500 chars of body`, product one hot, event word
flags. Output: calibrated probabilities for UP, DOWN, STABLE. Abstains below
`min_confidence` (0.55). v0 is TF-IDF plus logistic regression; v1 is e5
embeddings into logistic regression or LightGBM, whichever wins on validation,
calibrated on validation.

### 4.4 Label construction
Price index for product p:
1. Daily median per item across premises, dropping non positive prices.
2. Drop item-days outside [1/3, 3] times that item's rolling 30 day median.
3. Normalise each item by its own median over its first 90 days.
4. Carry each item forward up to 10 days (F18), average across the product's
   items, then a 7 day rolling mean.

Label for an article published on date d:
```
base = I_p(d - 1)
fut  = I_p(d + H)                              # H = 14 days, config
ret  = fut / base - 1
theta_p = max(MIN_THETA, k * std(ret over TRAIN only))   # k = 0.5, MIN_THETA = 0.005
UP     if ret >  theta_p
DOWN   if ret < -theta_p
STABLE otherwise
```
An example with no price window is dropped, never guessed.

### 4.5 Validation
Time split: train 70 percent, val 15 percent, test 15 percent, with a 14 day
purge gap at each boundary. Rolling origin by quarter. Metrics: macro F1, per
class precision and recall, confusion matrix, directional accuracy on non STABLE
rows only, bootstrap 95 percent interval. Baselines: majority, momentum,
keyword. Alert precision at p >= 0.7. Reported per product.

Forward validation: every live prediction is logged when made and scored H days
later against the index, by `scripts/daily_refresh.py`.

### 4.6 Explainer
Three modes: `host` (default, the chat host writes the prose from grounded
facts), `llm` (OpenAI compatible, one retry, template fallback), `template`
(deterministic). Output validated by pydantic. The disclaimer cannot be
rewritten by a model.

---

## 5. MCP tools

Server name `mymarket`, stdio by default, `--http` optional.

| Tool | Inputs | Returns |
|---|---|---|
| `list_domains` | none | domain, products, aliases, horizon |
| `fetch_news` | hours, sources, limit | recent articles |
| `filter_relevant` | domain, hours, limit | articles with matched products and scores |
| `predict_impact` | url or text or title, products | direction, probabilities, triggers, abstained |
| `price_context` | product, days | latest index, 7d and 30d change, series tail |
| `scan_alerts` | domain, hours, min_confidence, lang, limit | alerts with price context and explanation |
| `backtest_report` | product | test metrics, baselines, interval |
| `forward_validation` | product | matured live predictions, running accuracy |

All return JSON. On failure `{"error": ..., "hint": ...}`. Never a raw exception.
Logging to stderr only: stdout is the protocol.

---

## 6. Data model (SQLite)

`articles(id, url UNIQUE, title, source, published_at, summary, body, lang,
date_estimated, fetched_at)`,
`article_products(article_id, product, match_score, matched_terms)`,
`predictions(id, article_id, product, direction, p_up, p_down, p_stable,
model_version, created_at, horizon_days)`,
`outcomes(prediction_id, realized_ret, realized_label, scored_at)`,
`fetch_cache(url, body, status, fetched_at)`.

---

## 7. Non-functional

Everything free. Runs on a laptop, 8 GB RAM, CPU only for inference.
`scan_alerts` under 60 seconds for 48 hours of news (measured at about 15
seconds). Secrets in `.env`, never committed. Model files versioned beside
`metrics.json`. Unit tests for the filter, the price index, the label builder,
the explanation schema and the tool error shape.

---

## 8. Success metrics

| Metric | Target |
|---|---|
| Relevance precision on the audit set | at least 80 percent |
| Macro F1 on the time split test set | beat majority and momentum |
| Alert precision at p >= 0.7 | report it, no fixed target |
| `scan_alerts` end to end | under 60 seconds |
| Tools working in the Inspector | 8 of 8 |
| Forward validation | running, even with few matured rows |

If the model does not beat momentum, say so, and present it as a news context
layer with keyword triggers and price context. That is still a valid demo.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| A feed dies or changes | three sources, each independently skippable |
| A site blocks scraping | robots.txt respected up front, RSS summary fallback |
| Too few examples per product | widen the horizon, merge products |
| Labels nearly all STABLE | per product theta, class weights, lower k |
| Price file schema changes | schema check on load with a clear error |
| Price controls flatten a product | MIN_THETA floor, and it is reported |
| LLM quota or no key | host mode and the template need neither |
