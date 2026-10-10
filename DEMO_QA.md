# Demo questions, word for word

Type these sentences into Claude Desktop exactly as written. Each one is
phrased so Claude picks the right tool without you having to steer it.

Every expected outcome below was produced by running the tool. Where the
answer is weak, that is written down too.

---

## Q1. Opening

> **Type:**
> `What products can MyMarket track, and where does its price data come from?`

**Claude calls:** `list_domains`

**You will see:**

| field | value |
|---|---|
| domain | `food_staples` (Food Staples) |
| products | rice, egg, cooking_oil, chicken, 4 mapped price items each |
| price source | PriceCatcher (KPDN via data.gov.my), daily |
| horizon | 28 days |
| event words tracked | 26 |

**Say:** "Four Malaysian food staples. The price truth is PriceCatcher, the
government's own daily price survey, not something we invented. 80,437 daily
observations from January 2022 to September 2026."

---

## Q2. Prove it reads live news

> **Type:**
> `Scan the latest Malaysian news and tell me if anything could affect food staple prices in the next few weeks.`

**Claude calls:** `scan_alerts`

**You will see:** a scan of about 108 live headlines from Malay Mail, Bernama
and Free Malaysia Today, and then, most days, **zero alerts** with an action
summary reading "Nothing to act on. No recent news points to a staple price
move."

**If zero, say this and keep moving. Do not retry.**

> "Most days the Malaysian press says nothing about rice prices. A tool that
> manufactured a signal every day would be lying to you. It is telling me there
> is nothing to act on, which today is the correct answer. Let me show you what
> happens when there is something."

**If it does return an alert,** read out the priority, the recommended action
and the price context, then go to Q3 anyway.

---

## Q3. The explainability moment

> **Type:**
> `A Malaysian news headline says "Ceiling price for chicken set ahead of Hari Raya". How would that affect chicken prices, and why?`

**Claude calls:** `predict_impact`

**You will see:**

```
impact       POSITIVELY AFFECTED
meaning      prices look likely to ease, which is good for shoppers
confidence   0.63  (medium band)
triggers     ceiling price, Hari Raya

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

**Slow down here. Point at steps 2 and 3:**

> "It found two forces pulling in opposite directions. A price ceiling pushes
> down, festive demand pushes up. It came down on prices easing, then checked
> itself against the official data, which is already falling. That is the
> reasoning, in words a shopper can read, and it is not a black box score."

**Use this exact headline.** Nine out of ten improvised headlines come back as
no change, because the model over-predicts no change.

---

## Q4. The part that earns trust

> **Type:**
> `Show me past predictions where the real price outcome is already known, including ones the model got wrong.`

**Claude calls:** `demo_cases`

**You will see three held out test cases.** Do them in this order.

**First, the win:**

| | |
|---|---|
| headline | State to boost food security amid supply disruptions |
| product | egg |
| model said | NEGATIVELY AFFECTED, confidence 0.69 |
| advice given | "Buy your next egg stock now, before the increase reaches the shelf" |
| what actually happened | **egg prices rose 14.10 percent in 28 days** |
| index | 0.7998 to 0.9125 |

Read the headline, pause, ask the room which way they think egg prices went,
then reveal plus 14.1 percent.

**Then the miss. Do not skip it:**

| | |
|---|---|
| headline | Bulog warns of limited supply of subsidised cooking oil |
| model said | NEUTRAL, confidence **0.95** |
| advice given | "Buy cooking oil as normal" |
| what actually happened | prices fell 0.69 percent |
| verdict | bad advice |

> "And here is one it got wrong. It said no change, at 95 percent confidence,
> and it was wrong. That is why every alert carries a confidence band, and why
> we publish the accuracy instead of a highlight reel."

---

## Q5. The numbers

> **Type:**
> `How accurate is this model on the test set, and how does it compare to the baselines?`

**Claude calls:** `backtest_report`

**You will see:**

| | macro F1 |
|---|---|
| **MyMarket** | **0.333** |
| momentum baseline | 0.315 |
| keyword baseline | 0.310 |
| majority baseline | 0.263 |

Plus `beats_all_baselines: true`, alert precision **0.70** on 20 alerts, 40
test rows, model `v1-lightgbm`, horizon 28 days.

**Say, in this order:**

> "A model has to beat the dumb baselines or it is worth nothing. Ours beats
> all three. The one number worth remembering: when it is confident enough to
> raise an alert, it is right about 70 percent of the time."

**Never say "33 percent accurate."** Macro F1 is a 0 to 1 score, not a hit
rate. The answer to "what is your accuracy" is in the Q and A below.

---

## Q6. Optional, if you have 30 seconds

> **Type:**
> `What did rice prices actually do between August and November 2023?`

**Claude calls:** `price_context` with a date range

**You will see:** index 1.0736 rising to 1.4176, **change 32.04 percent** over
122 days.

> "India banned rice exports in 2023. Malaysian rice went up 32 percent, and it
> is in the government's own data. That is the kind of event this is built to
> catch early."

---

## Q7. Close, no tool

> "This is an awareness tool, not a forecast. News is a weak signal for grocery
> prices and we measured that rather than hiding it. Our binding constraint is
> 305 training examples out of 153,981 articles, because Malaysian papers print
> far more recipes than rice price stories."

---

## If a judge wants to type something themselves

Let them. It declines cleanly rather than guessing:

| they type | what happens |
|---|---|
| anything about petrol, Bitcoin, property | "No tracked food staple is discussed in this article" |
| an unknown product in `price_context` | error naming the four valid products |
| nonsense | same clean decline |
| a plausible food headline | **probably NEUTRAL.** Have the answer ready, below |

---

## Answers to the questions you will get

**"What is your accuracy?"**
> "Plain accuracy is 50 percent, and I will tell you why that is the wrong
> number. A model that always says no change scores 65 percent on our test set
> and never warns anybody about anything. We report macro F1, which counts all
> three outcomes equally. Operationally, alerts above 70 percent confidence are
> right 70 percent of the time."

**"Why did it say no change for my headline?"**
> "It over-predicts no change, because about two thirds of the training windows
> were no change. That is the cost of having only 305 examples, and it is the
> first thing more data would fix."

**"Did you test on data it trained on?"**
> "No. Split by time, oldest 70 percent to train, newest to test, with a 28 day
> gap purged at each boundary, because one news event produces many articles
> that share a label. The build prints proof of that gap on every run."

**"How do you know an article is really about rice?"**
> "Three stages: word boundary alias matching in English and Malay, a required
> price cue word in the headline, then a contrastive embedding check. We hand
> labelled 120 articles to measure it. Alias matching alone is 17.5 percent
> precise. The full filter is 80 percent."

**"Is the AI inventing the numbers?"**
> "It cannot. The tools return computed facts and Claude writes the sentences
> from those facts. The server never generates a number it did not compute."

**"Why is it only 0.018 better than momentum?"**
> "Honest answer: for raw accuracy you could use momentum alone, and our README
> says so. What momentum cannot do is tell you why, show you the article, or
> flag a subsidy announcement the day it lands."

**"What is your weakest product?"**
> "Rice. It has the most examples and the worst directional accuracy, and we do
> not yet know why. It is written down as the top open question."
