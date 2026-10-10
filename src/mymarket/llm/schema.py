"""Strict schema for an explanation. PRD 6.6 and F12."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from mymarket.config import DISCLAIMER

# UNCERTAIN means the model abstained below its confidence floor.
# UNVALIDATED means no model is trained yet, so the alert is a keyword and
# price context match only. Both are real states and must round trip.
DIRECTIONS = ("UP", "DOWN", "STABLE", "UNCERTAIN", "UNVALIDATED")


class ProductImpact(BaseModel):
    """What one article means for one product."""

    product: str
    direction: Literal["UP", "DOWN", "STABLE", "UNCERTAIN", "UNVALIDATED"]
    why: str = Field(min_length=1, max_length=400)


class Explanation(BaseModel):
    """The explanation object every alert carries."""

    headline: str = Field(min_length=1, max_length=160)
    domain_impact: str = Field(min_length=1, max_length=800)
    product_impacts: list[ProductImpact] = Field(default_factory=list)
    confidence_note: str = Field(min_length=1, max_length=400)
    disclaimer: str = DISCLAIMER

    @field_validator("headline", mode="before")
    @classmethod
    def at_most_twelve_words(cls, value: str) -> str:
        """PRD 6.6 caps the headline at 12 words.

        Runs before the length check so an over long headline is trimmed rather
        than rejected. Rejecting it would throw away an otherwise good
        explanation and drop the caller to the template.
        """
        if not isinstance(value, str):
            return value
        return " ".join(value.split()[:12])

    @field_validator("disclaimer")
    @classmethod
    def always_disclaim(cls, value: str) -> str:
        return value.strip() or DISCLAIMER
