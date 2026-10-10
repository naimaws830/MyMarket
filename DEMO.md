# MyMarket demo runbook

Five minutes, inside Claude Desktop, using the MCP server.

---

## Before you walk in

1. **Connect the server.** The config is already written to
   `%APPDATA%\Claude\claude_desktop_config.json`. Quit Claude Desktop
   completely (close it from the system tray, not just the window) and reopen
   it. Click the tools icon in a new chat. You should see **mymarket** with 9
   tools.

2. **Warm the model.** The first tool call loads a 120 MB embedding model and
   takes about 20 seconds. Run one throwaway question before the audience is
   watching, so the live demo is fast.

3. **Pre-generate the verifiable cases:**
   ```
   .venv\Scripts\python.exe scripts\08_demo_cases.py
   ```

4. **Screenshot everything** as a backup. If the venue Wi-Fi dies, the RSS
   fetch dies with it. The price data and the model are local and will still
   work, but `scan_alerts` needs the internet.

---

## The five minutes

### 1. What is this? (30 seconds)

> **Ask Claude:** What can MyMarket track?

Calls `list_domains`. Shows four Malaysian food staples and the price source.
Say: the price truth is PriceCatcher, the government's own daily price survey,
2022 to today, 80,437 daily observations.

### 2. It reads real news right now (60 seconds)

> **Ask Claude:** What Malaysian news in the last 3 days could affect food
> staple prices?

Calls `scan_alerts`. This fetches live RSS from Malay Mail, Bernama and FMT,
filters about a hundred headlines down to the ones genuinely about staple
prices, predicts a direction, and attaches the official price trend.

**If it returns zero alerts, that is a feature, not a failure.** Say so out
loud: "most days the news has nothing to say about rice prices, and a tool that
invents a signal every day would be lying to you." Then move to step 3, which
does not depend on today's news.

### 3. Does it actually work? (120 seconds, the important one)

> **Ask Claude:** Show me past predictions where we already know the outcome.

Calls `demo_cases`. These come from the held out test set, so the model never
saw them in training and the real price move is already known.

Work one case at a time:

1. Read the headline. Ask the room which way they think prices went.
2. Show what the model said, and how confident it was.
3. Show the actual PriceCatcher move.

The strongest case at the moment:

| | |
|---|---|
| headline | State to boost food security amid supply disruptions |
| product | egg |
| model said | price UP, 68 percent confidence |
| what happened | **UP, plus 14.1 percent over 28 days** |
| price index | 0.7998 on 28 Apr 2026 to 0.9125 on 27 May 2026 |

**Then show the one it got wrong, and do not skip it:**

| | |
|---|---|
| headline | Bulog warns of limited supply of subsidised cooking oil |
| model said | NEUTRAL, 95 percent confidence |
| what happened | DOWN, minus 0.69 percent |

Say plainly: it was confident and it was wrong. That is why the tool reports a
confidence band and why we publish the accuracy numbers.

### 4. How accurate, honestly? (60 seconds)

> **Ask Claude:** How accurate is this model, and what are the baselines?

Calls `backtest_report`. Use the numbers in the table below. The key line to
say out loud: **a model has to beat the dumb baselines or it is not worth
anything**, and ours beats all three, narrowly.

### 5. Close on the limits (30 seconds)

> **Ask Claude:** Show me the price trend for rice over the last 90 days.

Calls `price_context`. Finish on this slide:

- Awareness alert, not a forecast.
- News is a weak signal for grocery prices. We measured that rather than hid it.
- PriceCatcher is a price surveillance survey. Inflation analysis belongs to
  DOSM CPI.
- The best story in the data is real: rice rose about 45 percent in September
  2023 after India's export ban, and our index shows it.

---

## The numbers, as of today

| metric | value |
|---|---|
| test macro F1 | **0.333** |
| 95 percent interval | [0.218, 0.449] on 40 test rows |
| majority baseline | 0.263 |
| momentum baseline | 0.315 |
| keyword baseline | 0.310 |
| beats all three | yes, narrowly |
| alerts at confidence 0.70+ | 20, at **70 percent precision** |
| relevance filter precision | 0.80 with the price cue rule, held out |
| training examples | 305 |

If asked **"why is macro F1 only 0.33?"**: three classes, so random is 0.33 by
coincidence but the baselines are the real comparison, and we beat them. The
binding constraint is 305 training examples out of 153,981 articles, because
The Star publishes very few staple price stories.

If asked **"why not just use momentum then?"**: honest answer, for pure
accuracy you could, and we say so in the README. What momentum cannot do is
tell you *why*, or surface the article, or flag a subsidy announcement the day
it happens.

---

## Questions you should expect

**"Did you test on data the model saw in training?"**
No. Split by time, oldest 70 percent to train, newest 15 percent to test, with
a 28 day gap purged at each boundary because one news event produces many
articles that share a label. The build prints proof of the gap every run.

**"How do you know an article is really about rice?"**
Three stages: word boundary alias match in English and Malay, a required price
cue word in the headline, then an embedding check against product descriptions
minus distractor sentences. Measured on 120 hand labelled rows: alias alone is
17.5 percent precise, the full filter is 80 percent.

**"Is the AI making up the numbers?"**
It cannot. The tools return computed facts and Claude writes the sentences from
those facts. The server never generates a number it did not compute.

**"What happens when it does not know?"**
It abstains. Below the confidence floor it returns UNCLEAR rather than guessing.
