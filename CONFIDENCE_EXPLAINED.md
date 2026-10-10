# Where the confidence score comes from

Everything below was traced through the running code and verified with a real
prediction. Use this to answer "how do you get that number".

---

## The one sentence answer

> "It is the model's own probability for the class it picked. The model outputs
> three probabilities that sum to one, we take the largest, and that is the
> confidence. We then tell you empirically what that number is worth, because
> the raw probability is not calibrated."

---

## The full chain, with a real example

Headline: **"Ceiling price for chicken set ahead of Hari Raya"**

### Step 1. The article becomes 411 numbers

| part | columns | what it is |
|---|---|---|
| text embedding | 384 | `multilingual-e5-small` turns the title, summary and first 1500 characters of body into a 384 dimension vector. Handles English and Malay in one model |
| product | 1 | one hot. Products are merged into one group, so this is a single column |
| event flags | 26 | binary: does the text contain subsidy, shortage, flood, ceiling price, Hari Raya, ringgit and so on |
| **total** | **411** | |

### Step 2. LightGBM turns those into three probabilities

The classifier is a `LGBMClassifier` with `objective=multiclass` and
`class_weight=balanced`, trained on 210 examples. Its `predict_proba` returns
one probability per class, summing to 1:

```
DOWN     0.6291
STABLE   0.2441
UP       0.1268
         ------
total    1.0000
```

### Step 3. Confidence is the largest of the three

```python
probs      = {cls: p for cls, p in zip(estimator.classes_, proba_row)}
best       = max(probs, key=probs.get)      # "DOWN"
confidence = probs[best]                    # 0.6291
abstained  = confidence < min_confidence    # 0.6291 < 0.55 -> False
```

`src/mymarket/model/predict.py`, in `predict()`.

So **confidence 0.63, direction DOWN**, which the reader sees as POSITIVELY
AFFECTED, prices likely to ease.

### Step 4. Below the floor, it refuses to answer

If the largest probability is under `min_confidence` (0.55), the direction is
replaced with UNCLEAR and the prediction is marked `abstained`. An abstention
is not logged as a prediction, because it is not a claim.

---

## The honest part: that number is not a probability of being right

**Calibration was attempted and rejected.** The log in `metrics.json` says:

> `rejected, it dropped validation macro F1 from 0.466 to 0.403 on 35 rows, so
> probabilities are uncalibrated`

Sigmoid calibration on a small, imbalanced validation split collapsed the model
towards the majority class, so it was discarded and the raw LightGBM
probabilities ship instead.

**This means 0.95 confidence does not mean a 95 percent chance of being right.**
It means the trees strongly agreed. Those are different claims, and conflating
them is the mistake to avoid tomorrow.

So we measured what the number is actually worth on the held out test set:

| confidence band | predictions | actually right |
|---|---|---|
| below 0.55 | 7 | 43 percent |
| 0.55 to 0.70 | 13 | 23 percent |
| 0.70 to 0.85 | 8 | 62 percent |
| 0.85 and up | 12 | **75 percent** |
| **0.70 and above** | **20** | **70 percent** |
| every prediction | 40 | 50 percent |

**That last column is the honest reading of the score**, and it is what the
`confidence_note` field reports back to the user:

> "about 70 percent of calls at this confidence were right in testing"

---

## Why the bands are drawn where they are

| band | rule | why |
|---|---|---|
| high | 0.70 and above | the band that measured 70 percent correct |
| medium | 0.55 to 0.70 | above the abstention floor but no better than a coin flip |
| low | below 0.55 | the model abstains here |

`src/mymarket/model/impact.py`. The same 0.70 is the threshold for an ACT NOW
priority, so the strongest recommendation is gated on the only band with
evidence behind it.

---

## What to say if challenged

**"Is 0.63 confidence a 63 percent chance of being right?"**
> "No, and I want to be precise about that. It is the model's internal
> probability, not a calibrated one. We tried calibrating it and it made the
> model worse on validation, so we did not ship it. What we did instead is
> measure what the number is worth: predictions above 0.70 were right 70
> percent of the time on the held out set. That is the number on the alert."

**"One of your demo cases was 95 percent confident and wrong."**
> "Yes, and we show it deliberately. High confidence means the trees agreed
> strongly, not that the answer is certain. The 0.85 and up band is right about
> 75 percent of the time, so one in four is wrong. We publish that rather than
> only showing the hits."

**"Why is the 0.55 to 0.70 band worse than the band below it?"**
> "Small numbers. That band has 13 test cases and the one below it has 7. With
> 40 test rows in total, any single band is noisy. The pattern that holds is
> that above 0.70 is clearly better than below it, which is why that is where
> we set the alert threshold."

**"Could you make it calibrated?"**
> "Yes, with more data. Calibration needs a validation set big enough to fit a
> reliable mapping, and ours has 35 rows. That is the same root cause as every
> other limitation here: 305 training examples."
