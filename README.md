# MyMarket

**A kopitiam owner buys chicken every week. If prices are about to jump they
want to buy early. If prices are about to fall they want to wait. Today they
find out when the invoice arrives.**

MyMarket reads Malaysian news as it is published, keeps only the stories
genuinely about four food staples, predicts whether the shelf price will move
over the next 28 days, checks that call against the government's own daily
price survey, ranks it by urgency, and says what to do about it.

It is exposed as an MCP server, so the whole thing is usable from a chat
client.

Products: **rice, egg, cooking oil, chicken**.
Price truth: **PriceCatcher** (KPDN via data.gov.my), daily, CC BY 4.0.

> This is an awareness tool. It is not a forecast and not financial advice.
> That framing is in the product, not just the pitch: every alert carries the
> disclaimer, the model abstains when unsure, and the accuracy is published
> next to the baselines it has to beat.

---

## For judges: the three things worth your time

**1. It does not just score, it recommends.** Every alert carries a priority
(ACT NOW, WATCH, MONITOR, NO ACTION), a concrete next step, and the reasoning
behind it.

**2. You can check whether it is actually right, live.** Run `demo_cases` and
it returns held out predictions where the real price outcome is already known,
**including the ones it got wrong**. There is no curated highlight reel.

**3. Every weakness is written down.** The limitations section is long and
specific on purpose. If you find a problem that is not listed, we want to know.

**Verify it yourself in about 15 seconds:**

```bash
python scripts/10_prove_it.py
```

That deletes the model, shows the tools break, retrains from the training data
in about six seconds, and the rebuilt file comes back **bit identical**. It
also prints artefact hashes, unpacks the classifier, and shows the time split
with its purge gaps.

---

## Results, stated honestly

| | macro F1 |
|---|---|
| **MyMarket** | **0.333** |
| momentum baseline | 0.315 |
| keyword baseline | 0.310 |
| majority baseline | 0.263 |

95 percent bootstrap interval **[0.218, 0.449]** on 40 test rows. It beats all
three baselines, but it is 0.018 ahead of momentum, which is **inside the
noise**. Call it a tie trending the right way, not a win.

**The number that matters operationally:** of the 20 predictions made at 70
percent confidence or above, **70 percent were correct**. Overall accuracy
across every prediction is 50 percent.

### Why macro F1 and not accuracy

Accuracy rewards doing nothing. A model that always says "no change" scores
**65 percent accuracy** here, because 26 of 40 test cases genuinely are no
change, while never warning anyone about anything. Macro F1 counts all three
outcomes equally and scores that model 0.263, correctly exposing it as useless.

```
MyMarket             accuracy 50.0%    macro F1 0.333
Always say NEUTRAL   accuracy 65.0%    macro F1 0.263
```

Macro F1 is a 0 to 1 score, not a hit rate. Reading 0.333 as "right a third of
the time" is wrong in both directions.

### Per product, including where it is weakest

| product | test rows | macro F1 |
|---|---|---|
| chicken | 12 | 0.429 |
| cooking oil | 8 | 0.256 |
| rice | 12 | 0.246 |
| egg | 8 | 0.083 |

Rolling origin, which retrains for each quarter rather than relying on one
split, beats the majority baseline in **5 of 9 quarters**.

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
                                                       9 tools
```

### The relevance filter is the hard part

Not prediction. Deciding which articles are even about a staple price. The
Star publishes far more recipes and restaurant reviews mentioning chicken than
it publishes chicken price stories.

| stage | rule | precision |
|---|---|---|
| 1. Alias match | word boundary match, English and Malay. Exclude phrases are masked out of the text rather than rejecting the whole article, so "chicken rice stalls hit as chicken supply tightens" still matches | 0.175 |
| 2. Price cue | the headline must contain one of 28 cue words: price, harga, subsidy, tariff, shortage, inflation. The body is deliberately not searched, because a review mentions cost in paragraph nine while a price story says so in the headline | 0.800 |
| 3. Embedding | contrastive: similarity to the product descriptions minus similarity to 11 distractor sentences | 0.875 combined |

**Why contrastive and not a plain cosine.** Measured directly: "Best fried
chicken restaurants in Kuala Lumpur" scored **0.8828** while "Chicken prices
rise as feed costs climb" scored **0.8819**. A plain similarity ranks the noise
above the signal. Subtracting the distractors separates them.

**What the audit found.** The dominant false positive was crude palm oil
futures and plantation output stories: upstream commodity news, not a retail
shelf price. Distractor sentences for those are now in the config.

### The price index

A unitless index per product, so a 10 kg rice sack and a 1 kg oil packet can be
averaged without the sack dominating.

```
1. daily median price per item across premises, dropping price <= 0
2. drop any item-day outside [1/3, 3] x its rolling 30 day median
3. normalise each item by its own median over its first 90 days
4. carry each item forward up to 10 days, average across items,
   then a 7 day rolling mean
```

Step 4 was added after the survey cadence was found to degrade: by 2026 cooking
oil items are surveyed about weekly while eggs remain near daily. Without it
the cooking oil and rice indexes simply stopped in 2025.

**It finds real events.** Rice rises about 45 percent in September 2023, the
import crisis following India's export ban. Egg is a flat line until mid 2024
because eggs were administratively priced.

### Labels

```
base    = I_p(d - 1)
fut     = I_p(d + 28)
ret     = fut / base - 1
theta_p = max(0.005, 0.5 * std(ret over TRAIN only))

ret >  theta_p  ->  NEGATIVELY AFFECTED   price rises, bad for shoppers
ret < -theta_p  ->  POSITIVELY AFFECTED   price eases, good for shoppers
otherwise       ->  NEUTRAL
```

theta is computed on the training period only. Computing it on all data would
leak test volatility into the labels. The floor exists because a price
controlled product has an almost flat training period, which drives the
threshold towards zero.

### Decision support

A direction and a probability is not a decision. Urgency is scored out of 100
from four components: model confidence (40), whether the direction costs the
buyer money (20), whether official prices already agree (25), and how fresh the
news is (15).

Two caps stop a coin flip reaching the top of the list. **ACT NOW requires the
confidence band that actually measured 70 percent correct.** NEUTRAL never
exceeds NO ACTION, because "carry on as normal" is not an alert.

### Explainability

26 documented mechanisms turn a detected event word into a causal chain:

```
1. The article talks about chicken (it mentions chicken).
2. It mentions "ceiling price": a maximum price has been set. Sellers cannot
   legally charge above the ceiling, which pushes prices downward.
3. It mentions "Hari Raya": household buying rises sharply before the holiday,
   which pushes prices upward.
4. Official PriceCatcher data shows chicken prices already falling (-1.5% / 7d).
5. Any change usually reaches the shelf in 1 to 2 weeks.
```

Every output states that this is the established domain relationship and **not
a readout of what the model weighted internally**. Claiming otherwise would be
inventing an explanation. When the mechanism disagrees with the model, the
output says to trust the alert less.

The server cannot invent a number: tools return computed facts and the chat
host writes the prose, so the server never writes a sentence containing a
figure it did not compute.

---

## The nine MCP tools

| tool | what it answers |
|---|---|
| `list_domains` | what can you track? |
| `fetch_news` | what has been published recently? |
| `filter_relevant` | which of those are really about a staple price? |
| `predict_impact` | for this story, which way do prices go, and why? |
| `price_context` | what has this product's official price actually done? any window from 2022 |
| `scan_alerts` | the main view: ranked alerts, priority, action, price context |
| `backtest_report` | how accurate is this, against which baselines? |
| `forward_validation` | how have live predictions scored since launch? |
| `demo_cases` | past predictions with known outcomes, including the misses |

All return JSON. On failure they return `{"error": ..., "hint": ...}` and never
raise to the client. Out of domain requests are declined, not guessed: ask
about petrol or Bitcoin and it says no tracked food staple is discussed.

---

## Run it

```bash
pip install uv
uv venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Windows
.venv/Scripts/python.exe -m pytest                                      # 150 tests

# the server
python -m mymarket.mcp_server                      # stdio, for Claude Desktop
python -m mymarket.mcp_server --http --port 8765   # HTTP, for remote clients
```

The trained model and price data are committed, so a fresh clone works without
retraining. **No API key is needed:** running inside a chat host, the host
model writes the explanations from the grounded facts the tools return.

| document | what it is |
|---|---|
| [DEMO_QA.md](DEMO_QA.md) | the exact questions to ask and the answers they produce |
| [CONFIDENCE_EXPLAINED.md](CONFIDENCE_EXPLAINED.md) | where the confidence number comes from, and what it is not |
| [MyMarket_Project_Document.pdf](MyMarket_Project_Document.pdf) | the full 12 page write up, generated from the repository so its numbers cannot drift |
| [PRD.md](PRD.md) | the governing spec |
| [PUBLIC_ACCESS.md](PUBLIC_ACCESS.md) | serving it over HTTP and a tunnel |

---

## Limitations

The binding constraint, stated first: **153,981 articles produce only 305
labelled examples.** The alias stage finds 3,470 candidates and the filter
keeps about a fifth. Malaysian papers print far more recipes than rice price
stories. Almost everything below follows from that.

- **The margin over momentum is 0.018**, inside the noise. For raw accuracy you
  could use momentum alone. What momentum cannot do is tell you why, surface
  the article, or flag a subsidy announcement the day it lands.
- **The test set has been evaluated several times** across configurations.
  Selection was done on validation, but repeated looks make the headline figure
  optimistic by an unknown margin. The rolling origin numbers are more
  trustworthy.
- **Egg scores 0.083** on 8 test rows. Rice has the most examples and among the
  worst directional accuracy, and we do not know why.
- **The model over-predicts NEUTRAL** and can be confidently wrong. One demo
  case is a 95 percent confident NEUTRAL that was wrong.
- **Probabilities are uncalibrated.** Calibration was attempted and rejected
  because it cost 0.06 macro F1 on a 35 row validation split. So 0.95
  confidence means the trees agreed strongly, not a 95 percent chance of being
  right. What the number is worth was measured instead: 70 percent correct
  above 0.70, 75 percent above 0.85.
- **Filter precision is 0.55 to 0.61, not 0.80.** Two audits of 120 rows were
  run. Round 1 was human labelled, but the distractor sentences were tuned on
  it, so its 0.700 was an overfit. Round 2 was held out and scored 0.500, but
  its labels were produced by an AI assistant and are self marked "not
  human-validated". **No set is both human labelled and genuinely held out.** A
  third audit is the top open task.
- **Typos defeat the alias stage.** Matching is exact, so "chiken" does not
  match. Fuzzy matching is a known, unimplemented fix.
- **Trained on one source.** The Star only, run live against Malay Mail,
  Bernama and FMT, so live performance is an out of source test.
- **The item mapping is provisional**, ranked by history length and survey
  coverage rather than reviewed by a domain expert.
- **Forward validation has no matured rows yet.** The mechanism works; the
  first scoreable prediction is 28 days after the daily job starts.
- **Consumer electronics was removed** rather than validated badly. No free
  Malaysian dataset carries product level electronics prices, and the CPI
  fallback gives roughly 60 independent labels, which cannot support an honest
  claim.

### What would actually improve it

1. A broader news corpus beyond one outlet. Everything else is downstream of
   example count.
2. A third labelled audit, human, on this pipeline's own candidate pool.
3. Fuzzy alias matching. Cheap, and rapidfuzz is already a dependency.
4. Investigate rice specifically.
5. Per product models, once the data allows. Merging was forced by data volume,
   not chosen on principle.

---

## Provenance

Price data is PriceCatcher via data.gov.my under CC BY 4.0. Training news is
the Kaggle dataset `azraimohamad/news-article-weekly-updated`. Live news is
read from public RSS feeds, respecting robots.txt at one request per second,
with no stealth fetching. Malay Mail and FMT disallow their article paths, so
their bodies are never fetched and only the feed content is used.

`.claude/context/` holds OAC standards files written for TypeScript, which do
not govern this Python project. The governing spec is [PRD.md](PRD.md).
