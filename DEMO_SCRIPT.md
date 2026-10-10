# MyMarket demo script

Every expected answer below was produced by running the real tools. Nothing
here is a mockup. Where the model gives a weak answer, that is written down
too, so nothing surprises you on stage.

Ask these questions in Claude Desktop, in this order.

---

## Before you start

1. Quit Claude Desktop from the **system tray**, reopen, confirm **mymarket**
   shows **9 tools**.
2. Ask one throwaway question. The first call loads a 120 MB model, about 20
   seconds. Do this before anyone is watching.
3. Screenshot the four answers below as a backup. Only step 2 needs internet.

**The single biggest risk:** the model over-predicts NEUTRAL. If you improvise
a headline on stage you will probably get NEUTRAL back. Stick to the script.

---

## 1. Frame the decision, 20 seconds, no tool

Say this before touching the laptop:

> "A kopitiam owner buys chicken every week. If prices are about to jump they
> want to buy early. If prices are about to fall they want to wait. Today they
> find out when the invoice arrives. MyMarket reads the news and tells them
> which way staple prices are heading, and what to do about it."

---

## 2. "What can MyMarket track, and where does the price data come from?"

**Tool:** `list_domains`

**Expect:** four products (rice, egg, cooking oil, chicken), a 28 day horizon,
and PriceCatcher named as the price source.

**Say:** "The price truth is PriceCatcher, the government's own daily price
survey. 80,437 daily observations from January 2022 to September 2026. We did
not invent a price series, we used the official one."

---

## 3. "What Malaysian news in the last 3 days could affect food staple prices?"

**Tool:** `scan_alerts`

**This is the live pipeline:** it fetches RSS from Malay Mail, Bernama and FMT,
filters about 108 headlines down, predicts, ranks by urgency and attaches the
official price trend.

**Expect, honestly: often zero alerts.** The price cue filter is strict. The
answer will carry an action summary saying there is nothing to act on.

**If zero, say this and move on, do not apologise:**

> "Most days the news says nothing about rice prices. A tool that manufactured
> a signal every day would be lying to you. The summary is telling me there is
> nothing to act on, which is the correct answer today. Let me show you what it
> does when there is something."

Then go straight to step 4. **Do not retry or refresh.** You lose the room.

---

## 4. "Show me a news story and explain how it would affect prices"

**Tool:** `predict_impact`

Give it exactly this headline. It is the strongest live example and it has been
tested:

> **Ceiling price for chicken set ahead of Hari Raya**

**Expect, verbatim from a real run:**

```
impact      POSITIVELY AFFECTED
meaning     prices look likely to ease, which is good for shoppers
confidence  0.63  (medium band)
triggers    ceiling price, Hari Raya

1. The article talks about chicken (it mentions chicken).
2. It mentions "ceiling price": a maximum price has been set. Sellers cannot
   legally charge above the ceiling while it is in force, which pushes prices
   downward.
3. It mentions "Hari Raya": Hari Raya demand is building. Household buying
   rises sharply in the weeks before the holiday, which pushes prices upward.
4. Official PriceCatcher data shows chicken prices already falling
   (-1.5 percent over the last 7 days).
5. Any change usually reaches the shelf in 1 to 2 weeks.
```

**This is the moment to slow down.** Point at steps 2 and 3:

> "It found two forces pulling in opposite directions. A ceiling price pushes
> down, festive demand pushes up. It weighed them, came down on prices easing,
> and then checked itself against the official data, which is already falling.
> That is the reasoning, in words a shopper can read."

---

## 5. "But is it actually right? Show me past predictions with known outcomes"

**Tool:** `demo_cases`

**This is the part that wins points.** Held out test cases, so the model never
saw them in training and the real price move is already known.

**Expect, from a real run:**

| | case 1, RIGHT | case 3, WRONG |
|---|---|---|
| headline | State to boost food security amid supply disruptions | Bulog warns of limited supply of subsidised cooking oil |
| product | egg | cooking oil |
| model said | NEGATIVELY AFFECTED, 0.69 | NEUTRAL, 0.95 |
| advice given | "Buy your next egg stock now, before the increase reaches the shelf" | "Buy cooking oil as normal" |
| what happened | **+14.10 percent** in 28 days | fell 0.69 percent |
| verdict | good advice | **bad advice** |

**Do the right one first.** Read the headline, ask the room which way they
think egg prices went, then reveal plus 14.1 percent.

**Then do the wrong one, and do not skip it:**

> "And here is one it got wrong. It said no change, at 95 percent confidence,
> and it was wrong. That is why every alert carries a confidence band, and why
> we publish the accuracy instead of the highlight reel."

---

## 6. "How accurate is it, compared to a dumb baseline?"

**Tool:** `backtest_report`

**Expect:**

| | macro F1 |
|---|---|
| MyMarket | **0.333** |
| momentum | 0.315 |
| keyword rules | 0.310 |
| majority class | 0.263 |

Plus alert precision **0.70** on 20 alerts above 0.70 confidence.

**Say, in this order:**

> "A model has to beat the dumb baselines or it is worth nothing. Ours beats
> all three. The single number worth remembering: when it is confident enough
> to raise an alert, it is right about 70 percent of the time."

**Do not say "33 percent accurate".** Macro F1 is a 0 to 1 score, not a hit
rate. If someone asks for accuracy, see the Q and A below.

---

## 7. Optional, if you have a spare 30 seconds

> "Show me what rice prices did between August and November 2023"

**Tool:** `price_context` with `start=2023-08-01`, `end=2023-11-30`

**Expect:** `change_pct: 32.04`, index 1.0736 rising to 1.4176.

**Say:** "India banned rice exports in 2023. Malaysian rice went up 32 percent,
and it is right there in the government's own data. That is the kind of event
this is built to catch early."

---

## 8. Close on the limits, 20 seconds

Do not let the clock run out before you say this:

> "This is an awareness tool, not a forecast. News is a weak signal for grocery
> prices, and we measured that rather than hiding it. Our binding constraint is
> 305 training examples out of 153,981 articles, because Malaysian papers print
> far more recipes than rice price stories."

Volunteering the limitation is worth more in Q and A than one more feature.

---

## Questions you will get, with answers

**"What is your accuracy?"**
> "Plain accuracy is 50 percent, but that is the wrong metric here and I will
> say why. A model that always says no change scores 65 percent accuracy on our
> test set and never warns anyone about anything. We report macro F1, which
> counts all three outcomes equally. The number that matters operationally:
> alerts above 70 percent confidence are right 70 percent of the time."

**"Did you test on data the model trained on?"**
> "No. Split by time, oldest 70 percent to train, newest to test, with a 28 day
> gap purged at each boundary because one event produces many articles sharing
> a label. The build prints proof of that gap every run."

**"How do you know an article is really about rice?"**
> "Three stages. Word boundary alias matching in English and Malay, a required
> price cue word in the headline, then a contrastive embedding check. We hand
> labelled 120 articles to measure it: alias matching alone is 17.5 percent
> precise, the full filter is 80 percent."

**"Is the AI making up the numbers?"**
> "It cannot. The tools return computed facts and Claude writes sentences from
> those facts. The server never generates a number it did not compute."

**"Why only 0.018 better than momentum?"**
> "Honest answer: for raw accuracy you could use momentum alone, and the README
> says so. What momentum cannot do is tell you why, surface the article, or
> flag a subsidy announcement the day it lands."

**"What happens if I ask about something you do not track?"**
> Let them try it. Petrol, Bitcoin, nonsense: it declines cleanly every time
> and names the four products it does cover. It never guesses.

**"Why is rice your worst product?"**
> "It has the most examples and the worst directional accuracy, and we do not
> yet know why. It is written down as the top open question. Either the labels
> or the filter are wrong for rice specifically."

---

## Contingencies

| If | Do |
|---|---|
| `scan_alerts` returns nothing | Say the line in step 3 and move to step 4. Expected, not a failure |
| A tool call hangs over 30 seconds | First call loads the model. Keep talking, it will come back |
| Claude Desktop shows no tools | Quit from the system tray, not the window, and reopen |
| Someone types their own headline and gets NEUTRAL | "It over-predicts no change, because two thirds of its training windows were no change. That is the cost of only having 305 examples" |
| Wi-Fi dies | Everything except `scan_alerts` is local and still works. Use screenshots for that one |
