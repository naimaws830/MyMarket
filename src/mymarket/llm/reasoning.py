"""Plain language reasoning: how a news event reaches a shelf price.

This explains the DOMAIN MECHANISM, not the model's internals. It is the
difference between:

    "the article mentions chicken, so the price may rise"          (useless)
    "feed is the biggest cost in raising a chicken, so when feed
     costs rise farmers pass that on, usually within 1 to 2 weeks"  (useful)

An honest boundary, stated in the output itself: these chains are established
domain relationships triggered by words actually found in the article. They are
not a claim about which features the model weighted. Saying otherwise would be
inventing an explanation, which is the thing PRD F12 exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Each mechanism: what the event does, the chain to a shelf price, which way
# prices usually go, and how fast. Keyed by the event words in products.yaml.
MECHANISMS: dict[str, dict[str, Any]] = {
    "shortage": {
        "cause": "less of the product is reaching the market",
        "chain": "when supply falls short of demand, sellers can charge more",
        "pushes": "UP",
    },
    "kekurangan": {
        "cause": "a shortage is reported",
        "chain": "when supply falls short of demand, sellers can charge more",
        "pushes": "UP",
    },
    "flood": {
        "cause": "flooding damages farms and blocks transport",
        "chain": "less produce reaches market and what does costs more to move",
        "pushes": "UP",
    },
    "banjir": {
        "cause": "flooding damages farms and blocks transport",
        "chain": "less produce reaches market and what does costs more to move",
        "pushes": "UP",
    },
    "drought": {
        "cause": "dry weather cuts crop yields",
        "chain": "a smaller harvest means less supply at the same demand",
        "pushes": "UP",
    },
    "bird flu": {
        "cause": "disease forces culling of flocks",
        "chain": "fewer birds means fewer chickens and eggs for weeks afterwards",
        "pushes": "UP",
    },
    "feed cost": {
        "cause": "the cost of animal feed is rising",
        "chain": "feed is the largest single cost in raising poultry, so farmers pass it on",
        "pushes": "UP",
    },
    "import ban": {
        "cause": "imports are being restricted",
        "chain": "local supply has to cover demand on its own, which tightens the market",
        "pushes": "UP",
    },
    "export ban": {
        "cause": "a producing country is holding back exports",
        "chain": "importing countries compete for a smaller pool, so landed cost rises",
        "pushes": "UP",
    },
    "tariff": {
        "cause": "a tariff adds a charge at the border",
        "chain": "importers pay more and generally pass it down the chain",
        "pushes": "UP",
    },
    "levy": {
        "cause": "a levy adds a charge",
        "chain": "the cost is usually passed along to the buyer",
        "pushes": "UP",
    },
    "SST": {
        "cause": "sales and service tax applies",
        "chain": "tax raises the final price unless the item is exempted",
        "pushes": "UP",
    },
    "ringgit": {
        "cause": "the ringgit exchange rate is in play",
        "chain": "a weaker ringgit makes imported food and imported feed cost more",
        "pushes": "UP",
    },
    "minimum wage": {
        "cause": "labour costs are rising",
        "chain": "farms, processors and retailers all carry higher wage bills",
        "pushes": "UP",
    },
    "festive": {
        "cause": "a festive period is approaching",
        "chain": "household demand spikes while supply stays flat",
        "pushes": "UP",
    },
    "Hari Raya": {
        "cause": "Hari Raya demand is building",
        "chain": "household buying rises sharply in the weeks before the holiday",
        "pushes": "UP",
    },
    "Ramadan": {
        "cause": "Ramadan buying is building",
        "chain": "household demand rises through the fasting month",
        "pushes": "UP",
    },
    "Chinese New Year": {
        "cause": "Chinese New Year demand is building",
        "chain": "household buying rises sharply before the holiday",
        "pushes": "UP",
    },
    "subsidy": {
        "cause": "the government is paying part of the cost",
        "chain": "a subsidy lowers what the shopper pays, or stops a rise reaching them",
        "pushes": "DOWN",
    },
    "subsidi": {
        "cause": "the government is paying part of the cost",
        "chain": "a subsidy lowers what the shopper pays, or stops a rise reaching them",
        "pushes": "DOWN",
    },
    "ceiling price": {
        "cause": "a maximum price has been set",
        "chain": "sellers cannot legally charge above the ceiling while it is in force",
        "pushes": "DOWN",
    },
    "harga siling": {
        "cause": "a maximum price has been set",
        "chain": "sellers cannot legally charge above the ceiling while it is in force",
        "pushes": "DOWN",
    },
    "price control": {
        "cause": "price controls are being applied",
        "chain": "controlled items are held at a set price regardless of market pressure",
        "pushes": "DOWN",
    },
    "kawalan harga": {
        "cause": "price controls are being applied",
        "chain": "controlled items are held at a set price regardless of market pressure",
        "pushes": "DOWN",
    },
    "SHMMP": {
        "cause": "the festive maximum price scheme is in force",
        "chain": "selected items are capped for the duration of the scheme",
        "pushes": "DOWN",
    },
    "palm oil": {
        "cause": "crude palm oil prices are moving",
        "chain": "palm oil is the raw material for most Malaysian cooking oil",
        "pushes": "UP",
    },
}

# How fast a change typically reaches the shelf, per product.
SPEED = {
    "rice": "2 to 4 weeks, faster if shoppers start stockpiling",
    "egg": "1 to 3 weeks, eggs move through the chain quickly",
    "cooking_oil": "2 to 6 weeks, slowed by the subsidised packet price",
    "chicken": "1 to 2 weeks, the fastest moving of the four",
}


@dataclass(frozen=True)
class Reasoning:
    """A plain language causal chain for one (article, product)."""

    product: str
    steps: list[str]
    summary: str
    mechanisms_found: list[str]
    agrees_with_model: bool | None
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_plain_words": self.summary,
            "step_by_step": self.steps,
            "mechanisms_detected": self.mechanisms_found,
            "mechanism_agrees_with_model": self.agrees_with_model,
            "how_fast": SPEED.get(self.product, "typically a few weeks"),
            "note": self.caveat,
        }


def explain_chain(
    product: str,
    direction: str,
    event_words: list[str] | None = None,
    matched_terms: list[str] | None = None,
    price_context: dict[str, Any] | None = None,
) -> Reasoning:
    """Build the causal chain for one product from the words actually present."""
    label = product.replace("_", " ")
    found = [w for w in (event_words or []) if w in MECHANISMS]

    steps: list[str] = []
    terms = ", ".join((matched_terms or [])[:3]) or label
    steps.append(f"The article talks about {label} (it mentions {terms}).")

    pushes: list[str] = []
    for word in found[:3]:
        spec = MECHANISMS[word]
        pushes.append(spec["pushes"])
        arrow = "upward" if spec["pushes"] == "UP" else "downward"
        steps.append(
            f'It mentions "{word}": {spec["cause"]}. {spec["chain"].capitalize()}, '
            f"which pushes prices {arrow}."
        )

    if not found:
        steps.append(
            "No specific cause such as a shortage, a subsidy or a tariff was found "
            "in the text, so there is no clear mechanism to point to. The model is "
            "going on the wording of the article alone, which is weaker ground."
        )

    change = (price_context or {}).get("change_7d_pct")
    if change is not None:
        movement = "rising" if change > 0 else "falling" if change < 0 else "flat"
        steps.append(
            f"Official PriceCatcher data shows {label} prices already {movement} "
            f"({change:+.1f} percent over the last 7 days)."
        )

    steps.append(f"Any change usually reaches the shelf in {SPEED.get(product, 'a few weeks')}.")

    agrees: bool | None = None
    if found and direction in {"UP", "DOWN"}:
        agrees = direction in pushes

    if direction == "UP":
        summary = (
            f"Expect {label} to get more expensive. "
            + (f'The article points to {found[0]}, and ' if found else "")
            + "that normally feeds through to the shelf price."
        )
    elif direction == "DOWN":
        summary = (
            f"Expect {label} to get cheaper or at least stop rising. "
            + (f'The article points to {found[0]}, which eases prices.' if found else "")
        )
    elif direction == "STABLE":
        if found:
            # Do not claim there is no cause when the steps above just named
            # two. The tension is the useful thing to report.
            pushed = "up" if pushes.count("UP") >= pushes.count("DOWN") else "down"
            summary = (
                f"The model expects no clear change for {label}, even though the "
                f"article mentions {' and '.join(found[:2])}, which usually pushes "
                f"prices {pushed}. Treat that disagreement as a reason for caution "
                "rather than reassurance."
            )
        else:
            summary = (
                f"No clear change expected for {label}. Nothing in the article points "
                "to a cause strong enough to move the shelf price."
            )
    else:
        summary = (
            f"Not enough signal to call {label} either way. Read this as background, "
            "not as a prediction."
        )

    caveat = (
        "This explains the usual relationship between this kind of news and prices. "
        "It is not a readout of what the model calculated internally."
    )
    if direction == "STABLE" and found:
        caveat = (
            "The article names a cause that usually moves prices, but the model "
            "called no change. One of the two is wrong. " + caveat
        )
    if agrees is False:
        caveat = (
            "Careful: the usual mechanism points the opposite way to the model's call. "
            "That disagreement is a reason to trust this alert less. " + caveat
        )

    return Reasoning(product, steps, summary, found, agrees, caveat)
