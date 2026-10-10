"""Two stage relevance filter, with the embedding stage stubbed.

The real model is not loaded here: these tests check the wiring and the
degradation path, not the quality of the embeddings.
"""

from __future__ import annotations

from mymarket.filter import embed_filter, relevance


def stub_scores(monkeypatch, scores: dict[str, float]) -> None:
    monkeypatch.setattr(embed_filter, "available", lambda: True)
    monkeypatch.setattr(embed_filter, "score_many", lambda text, products: scores)


def test_embedding_stage_drops_a_low_scoring_alias_hit(test_settings, monkeypatch):
    stub_scores(monkeypatch, {"egg": 0.01})

    assert relevance.match_article("Turtle lays eggs on the beach") == []


def test_embedding_stage_keeps_a_high_scoring_alias_hit(test_settings, monkeypatch):
    stub_scores(monkeypatch, {"rice": 0.09})

    matches = relevance.match_article("Rice prices climb after export ban")

    assert [m.product for m in matches] == ["rice"]
    assert matches[0].embed_score == 0.09
    assert matches[0].passed_embedding


def test_no_alias_hit_means_no_embedding_call(test_settings, monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("the embedding stage must not run without an alias hit")

    monkeypatch.setattr(embed_filter, "available", explode)

    assert relevance.match_article("Government announces new highway") == []


def test_filter_degrades_to_alias_only_when_the_model_is_missing(test_settings, monkeypatch):
    monkeypatch.setattr(embed_filter, "available", lambda: False)

    matches = relevance.match_article("Rice prices climb after export ban")

    assert [m.product for m in matches] == ["rice"]
    assert matches[0].embed_score is None


def test_use_embeddings_false_skips_the_second_stage(test_settings, monkeypatch):
    monkeypatch.setattr(
        embed_filter, "available", lambda: (_ for _ in ()).throw(AssertionError("not called"))
    )

    matches = relevance.match_article("Rice prices climb", use_embeddings=False)

    assert [m.product for m in matches] == ["rice"]


def test_threshold_argument_overrides_the_configured_one(test_settings, monkeypatch):
    stub_scores(monkeypatch, {"rice": 0.02})

    assert relevance.match_article("Rice prices climb") == []
    assert relevance.match_article("Rice prices climb", threshold=0.01)


def test_match_many_keeps_only_matching_articles(test_settings, monkeypatch):
    stub_scores(monkeypatch, {"rice": 0.09, "egg": 0.09})
    articles = [
        {"title": "Rice prices climb", "summary": "", "body": ""},
        {"title": "New highway opens", "summary": "", "body": ""},
    ]

    out = relevance.match_many(articles)

    assert len(out) == 1
    assert out[0][0]["title"] == "Rice prices climb"


def test_match_carries_event_words_through(test_settings, monkeypatch):
    stub_scores(monkeypatch, {"rice": 0.09})

    matches = relevance.match_article("Rice subsidy stays after flood damage")

    assert set(matches[0].event_words) == {"subsidy", "flood"}


def test_price_cue_rule_rejects_a_lifestyle_article(test_settings, monkeypatch):
    """The audits showed food coverage is dominated by recipes and buffets."""
    stub_scores(monkeypatch, {"rice": 0.09})

    assert relevance.match_article("Curry laksa with a modern twist", require_price_cue=True) == []


def test_price_cue_rule_keeps_a_real_price_story(test_settings, monkeypatch):
    stub_scores(monkeypatch, {"rice": 0.09})

    matches = relevance.match_article("Rice prices climb after export ban", require_price_cue=True)

    assert [m.product for m in matches] == ["rice"]


def test_price_cue_looks_only_at_headline_and_summary(test_settings):
    """A review mentions cost in paragraph nine; a price story says so up front."""
    body = "The dish is excellent. The cost of chicken has risen, the owner said."

    assert relevance.match_article("Fiery comfort of creamy chicken", "", body, require_price_cue=True) == []


def test_both_filter_stages_read_the_same_settings(test_settings, monkeypatch):
    """A regression guard.

    price_cues_in once imported `settings` directly at module load, so it kept
    reading the real config while match_article read the patched one. The two
    stages of one filter must never disagree about their configuration.
    """
    stub_scores(monkeypatch, {"rice": 0.09})

    assert relevance.price_cues_in("Rice prices climb") == ["prices"]
    assert relevance.price_cues_in("Curry laksa with a twist") == []
