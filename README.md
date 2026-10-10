# MyMarket

An MCP server that reads live Malaysian news, finds the stories tied to four food
staples, and says whether they point to prices going UP, DOWN or STABLE over the
next 14 days. Every call is checked against official daily price data, and the
accuracy is reported with the baselines it has to beat.

Products: **rice, egg, cooking oil, chicken**. Price truth: **PriceCatcher**
(KPDN, via data.gov.my), daily, CC BY 4.0.

This is an awareness tool. It is not a forecast and not financial advice.

---

## Status

| Area | State |
|---|---|
| Price pipeline | Working on real data. 2022-01-01 to 2026-09-24, 80,437 daily rows, 122 items |
| Price index | Built for all four products, about 1,725 days each, plots in `data/processed/plots/` |
| Live news | 3 feeds working, about 108 unique articles per scan |
| Relevance filter | Two stages working on live news |
| MCP server | All 8 tools respond over stdio |
| Training news | 153,981 articles, 2020-10 to 2026-10, all from thestar.com.my |
| Trained model | Trained. Test macro F1 **0.350** against a majority baseline of 0.261 |
| Tests | 106 passing |

### Model result, read honestly

| | macro F1 |
|---|---|
| **MyMarket** | **0.333** |
| momentum baseline | 0.315 |
| keyword baseline | 0.310 |
| majority baseline | 0.263 |

95 percent interval [0.218, 0.449] on 40 test rows. It is ahead of all three
baselines but the margins are small and the interval is wide. Rolling origin
beats majority in 6 of 9 quarters.

**Alerts at confidence 0.70 or above: 20, precision 0.700.** This is the PRD
metric that matters most for an alerting product, and it is the reason this
configuration was chosen over a higher scoring one. See below.

### Why a lower macro F1 was the right choice

An earlier configuration, without the price cue rule, scored macro F1 0.406 on
1,074 examples. It was discarded. Side by side:

| | no price cue | with price cue |
|---|---|---|
| training examples | 1,074 | 305 |
| macro F1 | 0.406 | 0.333 |
| alerts at p >= 0.70 | **0** | **20** |
| alert precision | not measurable | **0.700** |

The higher scoring model never produced a single confident prediction, so as an
alert product it did nothing. Worse, inspecting its training rows showed what
it was actually learning. Real examples it was trained on:

- "Labuan's tourism potential largely untapped" -> rice, UP
- "Yummy Korean delights for sharing" -> chicken, UP
- "Curry laksa with a modern twist hits the market" -> chicken, UP

For an article like that the label depends only on the date, so the model could
only learn seasonality. That explains why it merely tied the momentum baseline:
it had partly become a momentum model wearing a text model's clothes.

The price cue rule requires a word such as price, harga, subsidy, tariff,
shortage or inflation in the headline or summary. On the held out teammate
audit it lifts precision from 0.283 (alias only) to 0.800, and to 0.875 when
combined with the embedding score. The PRD applied this rule to electronics
only; the audits showed food needs it just as badly.

The cost is honest and real: 305 examples instead of 1,074, and a wider
confidence interval. A smaller set of articles that are genuinely about prices
beats a larger set that is mostly recipes.

### Why the numbers are small

153,981 articles produce only 717 labelled examples. The alias stage finds
3,470 candidates, and the relevance filter keeps about a fifth of them. The
Star does not publish many staple price stories, and that volume, not the
model, is the binding constraint.

### Relevance filter, measured twice

Target for this MVP is **0.70 precision**, agreed down from the 0.80 in the PRD
because 0.80 was not reachable at any threshold. **That target is not met.**

| set | labelled by | rows | relevant | precision | recall |
|---|---|---|---|---|---|
| alias stage alone | n/a | 120 | 21 | 0.175 | 1.000 |
| round 1 | in house | 120 | 21 | 0.607 | 0.810 |
| round 2, held out | **an AI assistant, not human-validated** | 99 | 28 | 0.552 | 0.571 |

An earlier version of this README claimed 0.700. That number came from round 1
after the distractor sentences had been tuned on those same 21 positives, and
it did not survive contact with an independently labelled set: the same
configuration scored 0.500 there. **Treat real precision as roughly 0.55.**

The threshold is 0.015, chosen because it is the more stable of the candidates
across both sets and keeps much more recall, not because it maximises either
one.

What the audits did buy, and this part is solid: they identified the dominant
false positive as crude palm oil futures, plantation output and biodiesel
stories, which are upstream commodity news rather than a retail shelf price.
Distractor sentences for those are in `config/products.yaml`.

**Important caveat on round 2.** Every one of its 120 rows carries a note
beginning "Assistant review (not human-validated)". Those labels were produced
by an AI assistant, not by a person. They are a useful second opinion and they
were generated independently of this pipeline, which is why they caught the
round 1 overfit, but **they are not ground truth**. The only human labelled set
is round 1, and the distractor sentences were tuned on it, so no set is both
human labelled and genuinely held out.

Round 2 also came from a different pipeline with a different schema, so its
rows are not a random sample of this project's candidate pool: 99 of its 120
rows match our alias stage.

The honest summary: precision is somewhere around 0.55 to 0.61, the lower
figure resting on AI labels and the higher on human labels that the filter was
tuned against. A third audit, human labelled on this pipeline's own candidate
pool, is the top open task.

## What is already done for you

Everything that needed no login, no browser and no Colab has been run:

- PriceCatcher lookups and all 57 monthly files downloaded, aggregated to a
  daily table, raw files deleted (peak disk stayed near one month).
- `config/item_mapping.csv` written as a **provisional** mapping of 4 items per
  product, ranked by history length times survey coverage. Replace it at M3.
- `data/processed/price_index.parquet` and four sanity plots built.
- All three RSS feeds tested live, including FMT, which the original spec
  recorded as failing. It works, it just 301 redirects to
  `cms.freemalaysiatoday.com/feed`.
- The whole offline training path proven on synthetic data, so the code is known
  good before your real data arrives (`tests/test_pipeline_e2e.py`).

## What still needs you

| Step | Why it needs a human | Time |
|---|---|---|
| M1 Kaggle news | Needs your Kaggle login | 15 min |
| M3 Item mapping review | A judgement call about which items represent a product | 20 min |
| M5 Relevance audit | Someone has to read 120 headlines and say yes or no | 45 min |
| M6 Training | Needs M1 first. Local CPU or Colab | 30 to 90 min |
| M7 Claude Desktop | Editing a config file on your machine | 15 min |
| M8 Scheduling | Task Scheduler or cron on your machine | 15 min |

Full instructions are in [MANUAL_STEPS.md](MANUAL_STEPS.md).

---

## Install

Python 3.11 and git. From the project root:

```bash
pip install uv
uv venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Windows
# uv pip install --python .venv/bin/python -r requirements.txt         # Mac or Linux
.venv/Scripts/python.exe -m pytest
```

No `.env` is needed to run the server. Copy `.env.example` to `.env` only if you
want the optional LLM explainer.

---

## Run

```bash
# Price data (already done, resumes and skips finished months)
python scripts/01_download_prices.py
python scripts/04_map_items.py --write-default
python scripts/03_build_price_index.py

# After M1: news, examples, training
python scripts/02_prepare_news.py
python scripts/02_prepare_news.py --audit 120     # then label it, then --tune-filter
python scripts/05_build_examples.py
python scripts/06_train_local.py
python scripts/07_evaluate.py

# The server
python -m mymarket.mcp_server
npx @modelcontextprotocol/inspector uv run python -m mymarket.mcp_server
```

---

## The eight MCP tools

| Tool | What it answers |
|---|---|
| `list_domains` | What can you track? |
| `fetch_news` | What has been published recently? |
| `filter_relevant` | Which of those stories are really about a staple price? |
| `predict_impact` | For this story, which way do prices go? |
| `price_context` | What has this product's official price actually done? |
| `scan_alerts` | The main one. News plus direction plus price context plus explanation |
| `backtest_report` | How accurate is this, against which baselines? |
| `forward_validation` | How have live predictions scored since launch? |

Every tool returns JSON. On failure it returns `{"error": ..., "hint": ...}` and
never raises at the client.

---

## Why there is usually no LLM key

When MyMarket runs inside a chat host such as Claude Desktop, the host model is
the explainer. The tools return an `explanation.facts` block of grounded values
(direction, probabilities, trigger words, price changes) and the host writes the
sentences. That satisfies PRD flag F12 structurally: the server cannot invent a
number because it never writes prose about one.

`EXPLAIN_MODE` in `.env` controls this:

- `host` (default) return grounded facts plus a deterministic fallback.
- `llm` call an OpenAI compatible endpoint. One retry, then the template.
  Works with Kimi, Groq or Ollama by changing `LLM_BASE_URL`.
- `template` deterministic sentences only, never calls out.

The server works with no key in all three cases.

---

## How the price index works

Raw PriceCatcher is over a million rows a month, so it is reduced offline and
the MCP server never touches a raw file (PRD F10).

1. Daily median price per item across premises, dropping non positive prices.
2. Drop any item-day outside [1/3, 3] times that item's rolling 30 day median,
   so a data glitch goes but a real subsidy step stays.
3. Normalise each item by its own median over its first 90 days, which is what
   lets a 10 kg rice sack and a 1 kg oil packet be averaged together.
4. Carry each item forward up to 10 days, then average across the product's
   items and take a 7 day rolling mean.

Step 4 is not in the original spec and was added because the survey cadence
degrades badly over time: by 2026 cooking oil items are surveyed about weekly
while eggs are still near daily. Without the carry forward the cooking oil and
rice indexes simply stopped in 2025. A survey price is a step function, so the
last observed price is the honest estimate until the next survey.

The index is unitless. 1.0 is the item's own early median, not a ringgit price.

---

## Honest limits

These carry over from the PRD flags. F2, F14 and F15 are gone because consumer
electronics was dropped in v0.3.

- **F1** Direction only, UP, DOWN or STABLE, plus a plain "bad for shoppers" tag.
  No claim about magnitude.
- **F3** News and price data do not fully overlap. PriceCatcher starts 2022-01;
  only news inside that window can be labelled.
- **F4** News rarely moves grocery prices inside two weeks. The signal is weak
  and accuracy will be modest. If the model does not beat momentum, say so and
  present the keyword triggers and price context as the value instead.
- **F5** Policy moves staple prices in steps, not smooth curves. Subsidy and
  ceiling price words are model features, not noise.
- **F6** The split is by time with a 14 day purge gap, never random, and
  `05_build_examples.py` prints the proof.
- **F7** Most windows are STABLE. Macro F1 is the metric. Plain accuracy is
  never reported alone.
- **F8** Training news is mostly The Star, live news is Malay Mail, Bernama and
  FMT. That is a source shift and live results are an out of source test.
- **F9** Plain Scrapling `Fetcher` only, robots.txt obeyed, 1 request per second,
  everything cached. Malay Mail and FMT disallow their article paths, so their
  bodies are never fetched; Malay Mail ships full text in the feed anyway.
- **F10** The server reads only the processed parquet.
- **F11** The index is a trimmed, normalised, smoothed median, not a raw price.
- **F12** The explainer gets only computed facts and a strict schema.
- **F13** Awareness only. Not financial advice.

Further limits found while building:

- **Confidence barely tracks accuracy.** On validation, raising the confidence
  threshold from 0.42 to 0.60 made accuracy *worse*, not better. So the
  abstention mechanism is weak, `min_confidence` is set to 0.42 from the
  validation distribution rather than the usual 0.55, and a high confidence
  alert should not be read as a more reliable one.
- **Calibration was rejected.** Sigmoid calibration on 63 validation rows
  collapsed the model onto STABLE and dropped validation macro F1 from 0.484 to
  0.232, so the shipped model is uncalibrated and `metrics.json` records that.
  Probabilities are therefore ordinal, not literal.
- **The test set was read twice.** Once on the first configuration and once on
  the one selected here. Configuration selection itself was done on validation,
  but two looks is two looks, so treat 0.350 as mildly optimistic.
- **RSS feeds are a rolling window of about 50 items.** Asking a feed for 48
  hours of news does not work once it has moved on. A scan now saves everything
  it sees to SQLite and reads the window back from there, so history accumulates
  from first run rather than being limited by the feed.

Two more from the price data:

- **Price controls flatten the labels.** The egg index is a flat line until
  mid 2024 because eggs were administratively priced. A flat training period
  drives the label threshold to zero, so a 0.5 percent floor (`MIN_THETA`)
  applies. `05_build_examples.py` reports any product sitting on that floor,
  because there the floor and not the data is deciding the labels.
- **The relevance filter is contrastive, not a plain similarity.** A raw cosine
  does not work: "Best fried chicken restaurants in Kuala Lumpur" scores *higher*
  than "Chicken prices rise as feed costs climb". Stage two scores an article as
  similarity to the product description minus similarity to distractor sentences.
  On live news that keeps a paddy subsidy story (0.083) and drops a story about
  turtles laying eggs (0.053).

---

## Layout

```
config/        products.yaml, sources.yaml, item_mapping.csv (provisional)
data/raw/      downloaded files, gitignored, emptied after aggregation
data/processed/daily prices, price index, plots, sqlite, gitignored
models/        model.joblib and friends, written by training
scripts/       the numbered offline pipeline, plus daily_refresh.py
src/mymarket/  ingest, filter, prices, model, llm, store, mcp_server
tests/         102 tests, including a synthetic end to end pipeline run
```

## Notes

- `.claude/context/` holds OAC standards. They are written for TypeScript, so the
  governing spec for this Python project is `PRD.md` plus the hard rules in
  `AGENT_PROMPT.md`.
- Never commit `.env` or `kaggle.json`. Both are gitignored.
- Do not scrape Shopee or Lazada, and never enable a Scrapling stealth fetcher.
