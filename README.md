# MyMarket

An MCP server that links Malaysian news to food staple price movements.

MyMarket reads Malaysian news from public RSS feeds, filters for stories about
four food staples, predicts whether the retail price will move over the next 28
days, validates that prediction against official daily price data, and returns
a ranked alert with a recommended action.

**Products:** rice, egg, cooking oil, chicken
**Price source:** PriceCatcher (KPDN via data.gov.my), daily, CC BY 4.0
**Interface:** Model Context Protocol, 9 tools, stdio and HTTP

MyMarket is an awareness tool, not a price forecast, and not financial advice.
Alerts carry a disclaimer, the model abstains below a confidence floor, and
accuracy is reported alongside the baselines it is measured against.

---

## Installation

Requires Python 3.11.

```bash
pip install uv
uv venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Windows
uv pip install --python .venv/bin/python -r requirements.txt           # macOS, Linux
```

Run the test suite:

```bash
.venv/Scripts/python.exe -m pytest        # 150 tests
```

The trained model and processed price data are committed, so a fresh clone
runs without retraining. No API key is required: when the server runs inside a
chat host, the host model writes explanations from the facts the tools return.

## Usage

```bash
# stdio transport, for Claude Desktop and other local clients
python -m mymarket.mcp_server

# streamable HTTP, for remote clients
python -m mymarket.mcp_server --http --port 8765
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mymarket": {
      "command": "C:\\path\\to\\MyMarket\\.venv\\Scripts\\python.exe",
      "args": ["-m", "mymarket.mcp_server"],
      "cwd": "C:\\path\\to\\MyMarket",
      "env": { "PYTHONPATH": "C:\\path\\to\\MyMarket\\src" }
    }
  }
}
```

### Verifying the model

```bash
python scripts/10_prove_it.py
```

Prints artefact hashes, unpacks the classifier, shows that different inputs
produce different outputs, retrains the model from scratch in about six
seconds, and displays the time split with its purge gaps.

---

## MCP tools

| Tool | Returns |
|---|---|
| `list_domains` | tracked products, aliases, price source, horizon |
| `fetch_news` | recent headlines from the configured feeds |
| `filter_relevant` | articles that pass all three filter stages, with scores |
| `predict_impact` | direction, probabilities, confidence band, triggers, reasoning |
| `price_context` | price index for any window from 2022 onward |
| `scan_alerts` | ranked alerts with priority, action, reasoning, price context |
| `backtest_report` | test metrics, baselines, intervals, per product, rolling origin |
| `forward_validation` | live predictions that have matured, scored against real prices |
| `demo_cases` | held out predictions with known outcomes, including incorrect ones |

All tools return JSON. Failures return `{"error": ..., "hint": ...}` rather
than raising. Requests outside the tracked domain are declined rather than
answered with a guess.

---

## How it works

```
live RSS  ->  relevance filter  ->  impact model  ->  urgency + action
(3 feeds)     3 stages              LightGBM          ranked alerts
                  |                     |                   |
                  |              PriceCatcher          reasoning chain
                  |              daily index           26 mechanisms
                  +-- audited, measured                     |
                                                       MCP server
```

### Relevance filter

Three stages run in order, cheapest first. Most articles mentioning a staple
are not about its price: food coverage is dominated by recipes, restaurant
reviews and festive features.

| Stage | Rule | Precision |
|---|---|---|
| Alias match | word boundary match on English and Malay aliases. Exclude phrases are masked out of the text rather than rejecting the article, so "chicken rice stalls hit as chicken supply tightens" still matches | 0.175 |
| Price cue | the headline or summary must contain one of 28 cue words such as price, harga, subsidy, tariff, shortage. The body is not searched, because a review mentions cost in passing while a price story says so in the headline | 0.800 |
| Embedding | contrastive score: similarity to the product description sentences minus similarity to 11 distractor sentences | 0.875 combined |

A plain cosine similarity does not discriminate here. Measured on the shipped
model, "Best fried chicken restaurants in Kuala Lumpur" scores 0.8828 while
"Chicken prices rise as feed costs climb" scores 0.8819, so the noise outranks
the signal. Subtracting distractor similarity separates them.

### Price index

A unitless index per product, so items of different pack sizes can be averaged.

```
1. daily median price per item across premises, dropping price <= 0
2. drop any item-day outside [1/3, 3] x its rolling 30 day median
3. normalise each item by its own median over its first 90 days
4. carry each item forward up to 10 days, average across items,
   then take a 7 day rolling mean
```

Step 4 handles an irregular survey cadence. By 2026 cooking oil items are
surveyed roughly weekly while eggs remain near daily; without the carry
forward, the cooking oil and rice indexes terminate in 2025.

### Labels

```
base    = I_p(d - 1)
fut     = I_p(d + 28)
ret     = fut / base - 1
theta_p = max(0.005, 0.5 * std(ret over TRAIN only))

ret >  theta_p  ->  NEGATIVELY AFFECTED   price rises
ret < -theta_p  ->  POSITIVELY AFFECTED   price eases
otherwise       ->  NEUTRAL
```

The threshold is computed on the training period only. The floor of 0.005
prevents a price controlled product, whose training period is almost flat,
from having every later rounding movement classified as a real move.

### Model

`multilingual-e5-small` embeddings (384 dimensions) plus a product one hot and
26 binary event word flags, 411 features total, into a LightGBM multiclass
classifier. Candidates are selected on validation; the test set is scored once.
Products are merged into a single group, which outperformed per product
features on validation given the available data volume.

Calibration was attempted and rejected: on a 35 row validation split it cost
0.06 macro F1, so the shipped probabilities are uncalibrated.

### Decision support

Urgency is scored out of 100 from four components: model confidence (40),
whether the direction costs the buyer money (20), whether official prices
already agree (25), and news recency (15). Scores map to ACT NOW, WATCH,
MONITOR or NO ACTION.

Two caps apply. ACT NOW requires the confidence band that measured 70 percent
correct in testing. NEUTRAL never exceeds NO ACTION.

### Explainability

26 documented mechanisms turn detected event words into a causal chain:

```
1. The article talks about chicken (it mentions chicken).
2. It mentions "ceiling price": a maximum price has been set. Sellers cannot
   legally charge above the ceiling, which pushes prices downward.
3. It mentions "Hari Raya": household buying rises sharply before the holiday,
   which pushes prices upward.
4. Official PriceCatcher data shows chicken prices already falling (-1.5% / 7d).
5. Any change usually reaches the shelf in 1 to 2 weeks.
```

Each output states that this describes the established domain relationship and
is not a readout of the model's internal weights. Where the mechanism
contradicts the model's prediction, the output flags the disagreement.

Tools return computed facts rather than prose, so the server cannot emit a
figure it did not calculate.

---

## Results

| | macro F1 |
|---|---|
| MyMarket | **0.333** |
| momentum baseline | 0.315 |
| keyword baseline | 0.310 |
| majority baseline | 0.263 |

95 percent bootstrap interval [0.218, 0.449] on 40 test rows. The model beats
all three baselines, but the 0.018 margin over momentum falls inside the noise.

Of 20 predictions made at 70 percent confidence or above, 70 percent were
correct. Overall accuracy across all predictions is 50 percent.

### Metric choice

Macro F1 is the unweighted mean of the per class F1 scores, so each of the
three outcomes counts equally regardless of frequency. Accuracy is unsuitable
here because a model that always predicts NEUTRAL scores 65 percent accuracy on
this test set while never issuing a warning:

```
MyMarket             accuracy 50.0%    macro F1 0.333
Always predict NEUTRAL  accuracy 65.0%    macro F1 0.263
```

Macro F1 is a score between 0 and 1, not a success rate.

### Per product

| Product | Test rows | Macro F1 |
|---|---|---|
| chicken | 12 | 0.429 |
| cooking oil | 8 | 0.256 |
| rice | 12 | 0.246 |
| egg | 8 | 0.083 |

Rolling origin validation, which retrains for each quarter, beats the majority
baseline in 5 of 9 quarters.

### Validation method

The split is by time with a 28 day purge gap at each boundary, never random,
because a single news event produces many articles sharing one label. Metrics
include per class precision and recall, confusion matrices, directional
accuracy on moved rows only, bootstrap confidence intervals, three baselines,
and a rolling origin backtest.

---

## Project structure

```
config/           products.yaml, sources.yaml, item_mapping.csv
data/audit/       hand labelled relevance audit sets
data/processed/   price index and daily prices (committed), training scratch (ignored)
models/           trained model, thresholds, label map, metrics
scripts/          numbered offline pipeline, daily refresh, verification, docs
src/mymarket/     ingest, filter, prices, model, llm, store, mcp_server
tests/            150 tests
```

### Pipeline

| Script | Purpose |
|---|---|
| `01_download_prices.py` | download and aggregate PriceCatcher monthly files |
| `02_prepare_news.py` | clean the news corpus, write audit samples, tune the filter |
| `03_build_price_index.py` | build the per product price index and plots |
| `04_map_items.py` | map PriceCatcher items to products |
| `05_build_examples.py` | join news to prices, label, split by time |
| `06_train_local.py` | train and evaluate on CPU |
| `07_evaluate.py` | per product metrics, baselines, rolling origin |
| `08_demo_cases.py` | select held out cases with known outcomes |
| `09_build_project_doc.py` | generate the project PDF from the repository |
| `10_prove_it.py` | verify the model is trained and reproducible |
| `daily_refresh.py` | refresh prices, log predictions, score matured ones |

`06_train_local.py` rewrites `metrics.json`, so `07_evaluate.py` must run after
it to restore the per product and rolling origin sections.

### Documentation

| File | Contents |
|---|---|
| [PRD.md](PRD.md) | product requirements and design decisions |
| [MyMarket_Project_Document.pdf](MyMarket_Project_Document.pdf) | 12 page write up, generated from the repository |
| [DEMO_QA.md](DEMO_QA.md) | example questions and their verified outputs |
| [CONFIDENCE_EXPLAINED.md](CONFIDENCE_EXPLAINED.md) | derivation of the confidence score |
| [PUBLIC_ACCESS.md](PUBLIC_ACCESS.md) | serving over HTTP and tunnels |
| [MANUAL_STEPS.md](MANUAL_STEPS.md) | steps requiring manual intervention |

---

## Limitations

The binding constraint is training data volume: 153,981 articles yield 305
labelled examples. The alias stage finds 3,470 candidates and the filter
retains roughly a fifth. Most limitations below follow from this.

- The 0.018 margin over the momentum baseline is within the confidence
  interval. Momentum alone is competitive on accuracy; what it cannot provide
  is the article, the mechanism, or same day notice of a policy change.
- The test set has been evaluated across several configurations. Selection was
  performed on validation, but repeated evaluation makes the headline figure
  optimistic by an unquantified margin. Rolling origin results are more
  reliable.
- Egg scores 0.083 macro F1 on 8 test rows. Rice has the most examples and
  among the worst directional accuracy, which is unexplained.
- The model over-predicts NEUTRAL and can be confidently incorrect. One
  included demo case is a 95 percent confidence NEUTRAL that was wrong.
- Probabilities are uncalibrated. A confidence of 0.95 indicates strong
  agreement among trees, not a 95 percent likelihood of correctness. Measured
  empirically: 70 percent correct above 0.70, 75 percent above 0.85.
- Filter precision is 0.55 to 0.61 rather than 0.80. Two audits of 120 rows
  were conducted. The first was human labelled but the distractor sentences
  were tuned against it, making its 0.700 an overfit. The second was held out
  and scored 0.500, but its labels were AI generated and marked "not
  human-validated". No audit set is both human labelled and genuinely held out.
- Alias matching is exact, so misspellings are not matched. Fuzzy matching is
  unimplemented.
- Training data comes from a single publisher. Live operation against other
  outlets is an out of source test.
- The item mapping is provisional, ranked by history length and survey
  coverage rather than expert review.
- Forward validation has no matured rows. The first scoreable prediction falls
  28 days after the daily job begins running.
- Consumer electronics was removed from scope. No free Malaysian dataset
  provides product level electronics prices, and the CPI fallback yields
  roughly 60 independent labels, insufficient for a defensible claim.

### Roadmap

1. Broaden the news corpus beyond a single publisher.
2. Conduct a human labelled audit on this pipeline's own candidate pool.
3. Add fuzzy alias matching (rapidfuzz is already a dependency).
4. Investigate the rice underperformance.
5. Restore per product models once data volume allows.

---

## Data and attribution

Price data is PriceCatcher, published by KPDN via data.gov.my under CC BY 4.0.
Training news is the Kaggle dataset
`azraimohamad/news-article-weekly-updated`. Live news is read from public RSS
feeds at one request per second, respecting robots.txt, with no stealth
fetching. Malay Mail and Free Malaysia Today disallow automated access to
article pages, so only their feed content is used.

PriceCatcher is a price surveillance survey. For inflation analysis the
appropriate source is DOSM CPI.

`.claude/context/` contains OAC standards files written for TypeScript
projects; they do not govern this codebase.
