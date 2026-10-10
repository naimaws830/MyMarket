"""Plain language causal reasoning about how news reaches a shelf price."""

from __future__ import annotations

from mymarket.llm import reasoning


def chain(product="chicken", direction="UP", events=None, terms=None, context=None):
    return reasoning.explain_chain(product, direction, events or [], terms or [], context)


def test_named_mechanism_is_explained_not_just_listed():
    out = chain(events=["feed cost"])

    body = " ".join(out.steps)
    assert "feed" in body.lower()
    assert "pass it on" in body or "largest single cost" in body
    assert out.mechanisms_found == ["feed cost"]


def test_subsidy_pushes_prices_down():
    out = chain(direction="DOWN", events=["subsidy"])

    assert out.agrees_with_model is True
    assert "cheaper" in out.summary or "stop rising" in out.summary


def test_disagreement_with_the_model_is_flagged_as_a_caution():
    """A subsidy eases prices, so an UP call contradicts the mechanism."""
    out = chain(direction="UP", events=["subsidy"])

    assert out.agrees_with_model is False
    assert "opposite way" in out.caveat
    assert "trust this alert less" in out.caveat


def test_no_mechanism_is_admitted_rather_than_invented():
    out = chain(events=[])

    assert out.mechanisms_found == []
    assert any("no clear mechanism" in s or "weaker ground" in s for s in out.steps)
    assert out.agrees_with_model is None


def test_unknown_event_words_are_ignored_not_guessed():
    out = chain(events=["badminton", "monsoon cup"])

    assert out.mechanisms_found == []


def test_price_context_is_quoted_exactly_once_and_accurately():
    out = chain(events=["shortage"], context={"change_7d_pct": -2.4})

    quoted = [s for s in out.steps if "PriceCatcher" in s]
    assert len(quoted) == 1
    assert "-2.4" in quoted[0]
    assert "falling" in quoted[0]


def test_every_chain_states_it_is_not_a_model_readout():
    """Claiming to explain the model's internals would be inventing an explanation."""
    for direction in ("UP", "DOWN", "STABLE", "UNCERTAIN"):
        out = chain(direction=direction, events=["shortage"])
        assert "not a readout of what the model calculated" in out.caveat


def test_stable_and_uncertain_read_differently():
    assert "No clear change" in chain(direction="STABLE").summary
    assert "background" in chain(direction="UNCERTAIN").summary


def test_pass_through_speed_is_product_specific():
    assert "1 to 2 weeks" in chain(product="chicken").to_dict()["how_fast"]
    assert "2 to 4 weeks" in chain(product="rice").to_dict()["how_fast"]


def test_to_dict_shape():
    out = chain(events=["shortage"]).to_dict()

    assert set(out) == {
        "in_plain_words", "step_by_step", "mechanisms_detected",
        "mechanism_agrees_with_model", "how_fast", "note",
    }


def test_stable_does_not_claim_no_cause_when_it_just_named_one():
    """The summary must not contradict the steps above it."""
    out = chain(direction="STABLE", events=["shortage", "ringgit"])

    assert "Nothing in the article points to" not in out.summary
    assert "shortage" in out.summary
    assert "caution" in out.summary
    assert "One of the two is wrong" in out.caveat


def test_stable_without_mechanisms_still_reads_simply():
    out = chain(direction="STABLE", events=[])

    assert "Nothing in the article points to" in out.summary
