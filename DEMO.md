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

## The three minutes

The rubric caps the pitch at 3 minutes. This is the tight version. Everything
in the previous five minute version still works if you get questions.

### 0:00 to 0:20 - the decision, not the model

Say this, do not read a slide:

> "A kopitiam owner buys chicken every week. If prices are about to jump, they
> want to buy early. If prices are about to fall, they want to wait. Today they
> find out when the invoice arrives. MyMarket reads the news and tells them
> which way staple prices are heading, and whether to act."

### 0:20 to 1:00 - it works on live news

> **Ask Claude:** What Malaysian news could affect food staple prices, and what
> should I do about it?

Calls `scan_alerts`. Point at **three** things in the output:
1. it fetched live RSS and filtered about a hundred headlines down
2. every alert has a **priority** and a **recommended action**, sorted by urgency
3. the official PriceCatcher trend is attached to each one

**If today returns nothing, say it out loud and keep moving:** "most days the
news says nothing about rice, and a tool that manufactures a signal every day
would be lying to you. Here is the summary telling me there is nothing to act
on." Then go straight to the next step.

### 1:00 to 2:15 - but is it right? (the part that wins points)

> **Ask Claude:** Show me past predictions where we already know the outcome.

Calls `demo_cases`. Held out test cases, so the model never saw them and the
real price move is already known.

| | |
|---|---|
| headline | State to boost food security amid supply disruptions |
| advice at the time | **"Buy your next egg stock now, before the increase reaches the shelf"** |
| what happened | egg prices rose **14.1 percent** in 28 days |
| verdict | good advice |

Then immediately:

> "And here is one it got wrong."

| | |
|---|---|
| headline | Bulog warns of limited supply of subsidised cooking oil |
| advice at the time | "Buy cooking oil as normal" |
| what happened | prices fell 0.69 percent |
| verdict | bad advice, and it was confident |

Say: "that is why every alert carries a confidence band, and why we publish the
accuracy rather than the highlight reel."

### 2:15 to 2:45 - how accurate, honestly

> **Ask Claude:** How accurate is this model compared to the baselines?

Calls `backtest_report`. One sentence: **"a model has to beat the dumb
baselines or it is worth nothing, and ours beats all three."** Macro F1 0.333
against majority 0.263, momentum 0.315, keyword 0.310. Alerts above 70 percent
confidence run at 70 percent precision.

### 2:45 to 3:00 - the limit, stated first

> "This is an awareness tool, not a forecast, and we say so in the product.
> News is a weak signal for grocery prices. We measured that instead of hiding
> it."

Do not let the clock run out before you say this. Volunteering the limitation
is worth more in Q and A than one more feature.

---

## If you get a spare minute

> **Ask Claude:** Show me the rice price trend over the last two years.

Calls `price_context`. Rice rose about 45 percent in September 2023 after
India's export ban, and the index shows it clearly. It is the best story in the
data and it is real.

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
