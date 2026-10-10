"""MCP tool contract: JSON out, the error shape in, never a raw exception."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

import mymarket.mcp_server as server
from mymarket.model import predict as predictor
from mymarket.prices import index as price_index

TOOL_NAMES = {
    "list_domains",
    "fetch_news",
    "filter_relevant",
    "predict_impact",
    "price_context",
    "scan_alerts",
    "backtest_report",
    "forward_validation",
}


def call(name: str, *args, **kwargs):
    """Call a tool through its undecorated function, whatever the SDK wraps it in."""
    tool = getattr(server, name)
    func = getattr(tool, "fn", None) or getattr(tool, "func", None) or tool
    return func(*args, **kwargs)


def test_all_eight_tools_are_exposed():
    for name in TOOL_NAMES:
        assert hasattr(server, name), f"{name} is not defined"


def test_every_tool_result_is_json_serialisable():
    for name in ("list_domains", "backtest_report", "forward_validation"):
        json.dumps(call(name))


def test_tool_errors_wraps_an_unexpected_exception():
    @server.tool_errors
    def boom():
        raise ValueError("something broke")

    out = boom()

    assert out["error"] == "ValueError: something broke"
    assert "hint" in out
    assert "stderr" in out["hint"]


def test_tool_errors_gives_a_specific_hint_for_a_known_state():
    @server.tool_errors
    def no_model():
        raise predictor.ModelMissing("no model")

    out = no_model()

    assert "06_train_local" in out["hint"]


def test_tool_errors_hint_for_missing_price_data():
    @server.tool_errors
    def no_prices():
        raise price_index.PriceDataMissing("no index")

    assert "03_build_price_index" in no_prices()["hint"]


def test_price_context_rejects_an_unknown_product():
    out = call("price_context", "durian")

    assert "error" in out
    assert "durian" in out["error"]
    assert "rice" in out["hint"]


def test_filter_relevant_rejects_an_unknown_domain():
    out = call("filter_relevant", domain="consumer_electronics")

    assert "error" in out
    assert "food_staples" in out["hint"]


def test_scan_alerts_rejects_an_unknown_domain():
    out = call("scan_alerts", domain="fuel")

    assert "error" in out


def test_predict_impact_needs_an_input():
    out = call("predict_impact")

    assert out["error"] == "nothing to predict on"


def test_predict_impact_returns_the_error_shape_without_a_model(monkeypatch):
    monkeypatch.setattr(predictor, "available", lambda: False)
    predictor._bundle.cache_clear()

    out = call("predict_impact", title="Chicken prices climb as feed costs rise")

    assert "error" in out
    assert "ModelMissing" in out["error"]
    predictor._bundle.cache_clear()


def test_list_domains_describes_food_staples_only():
    out = call("list_domains")

    assert len(out["domains"]) == 1
    assert out["domains"][0]["key"] == "food_staples"
    products = {p["key"] for p in out["domains"][0]["products"]}
    assert products == {"rice", "egg", "cooking_oil", "chicken"}


def test_backtest_report_without_metrics_explains_itself(monkeypatch, tmp_path):
    monkeypatch.setattr(server.config, "METRICS_FILE", tmp_path / "missing.json")

    out = call("backtest_report")

    assert "error" in out
    assert "05_build_examples" in out["hint"]


def test_no_tool_writes_to_stdout():
    """stdout is the MCP stdio channel. One stray print corrupts the protocol."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        call("list_domains")
        call("price_context", "durian")
        call("backtest_report")
        call("forward_validation")

    assert buffer.getvalue() == ""


@pytest.mark.parametrize("name", sorted(TOOL_NAMES))
def test_tool_has_a_docstring_for_the_client(name):
    tool = getattr(server, name)
    func = getattr(tool, "fn", None) or getattr(tool, "func", None) or tool

    assert func.__doc__ and len(func.__doc__.strip()) > 40, f"{name} needs a usable description"
