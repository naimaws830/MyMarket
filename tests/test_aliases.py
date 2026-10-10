"""Alias filter: word boundaries, exclude phrases, Malay aliases, event flags."""

from __future__ import annotations

from mymarket.filter.aliases import (
    build_text,
    event_flags,
    find_terms,
    match_all,
)


def products(title: str, summary: str = "", body: str = "") -> set[str]:
    return {hit.product for hit in match_all(title, summary, body)}


def test_plain_alias_matches():
    assert "chicken" in products("Chicken prices climb in Selangor")
    assert "rice" in products("Rice supply steady, says ministry")


def test_malay_aliases_match():
    assert "egg" in products("Harga telur naik di Johor")
    assert "rice" in products("Pengeluaran padi terjejas banjir")
    assert "cooking_oil" in products("Subsidi minyak masak dikekalkan")


def test_word_boundary_blocks_substrings():
    # "ricefield" and "eggplant" must not trigger rice or egg.
    assert products("Ricefield museum opens") == set()
    assert "egg" not in products("Eggplant recipe of the week")


def test_exclude_phrase_kills_false_positive():
    assert "chicken" not in products("New chicken rice shop opens in Ipoh")
    assert "chicken" not in products("Ayam Brand launches new campaign")
    assert "rice" not in products("Best fried rice in Penang")
    assert "egg" not in products("An Easter egg hidden in the game")


def test_exclude_phrase_only_masks_its_own_span():
    # The phrase "chicken rice" is masked, but the second mention is real signal.
    title = "Chicken rice stalls hit as chicken supply tightens"
    assert "chicken" in products(title)


def test_case_insensitive():
    assert "chicken" in products("AYAM PRICES UP")


def test_body_is_read_but_truncated():
    body = ("filler " * 400) + "chicken shortage"
    # Default body_chars is 1500, "filler " * 400 is 2800 chars, so the tail is cut.
    assert "chicken" not in products("Market report", "", body)
    assert "chicken" in products("Market report", "", "chicken shortage " + body)


def test_build_text_joins_available_fields():
    assert build_text("T", "S", "B") == "T. S. B"
    assert build_text("T", "", "") == "T"


def test_event_flags_detect_configured_words():
    flags = event_flags("Government keeps subsidy after flood damage")
    assert flags["ev_subsidy"] == 1
    assert flags["ev_flood"] == 1
    assert flags["ev_drought"] == 0


def test_event_flags_cover_every_configured_word():
    flags = event_flags("nothing relevant here")
    assert len(flags) > 20
    assert set(flags.values()) == {0}


def test_find_terms_returns_requested_order_deduped():
    assert find_terms("rice and rice and beras", ["rice", "beras"]) == ["rice", "beras"]


def test_multiple_products_can_match_one_article():
    found = products("Chicken and egg prices rise together")
    assert {"chicken", "egg"} <= found
