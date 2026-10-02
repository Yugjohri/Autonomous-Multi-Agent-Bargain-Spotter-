"""End-to-end fixture run: saved posts -> scanner -> valuation -> planning, fully offline."""

import socket

import pytest

import agents.inr_store as inr_store
from deal_agent_framework import DealAgentFramework
from tests.test_valuation import FakeEncoder

SETTINGS = {
    "pricer_mode": "inr",
    "ensemble": {"inr": {"frontier": 0.6, "market": 0.3, "mrp": 0.1, "mrp_factor": 0.75}},
    "thresholds": {"min_discount_inr": 500, "min_discount_pct": 20},
    "planning": {"top_n": 3, "category_cooldown_minutes": 60, "alert_min_confidence": "low"},
    "scan": {"freshness_hours": 6, "max_llm_candidates": 30, "picks_per_scan": 8},
}


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"network access attempted: {args}")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


def test_fixture_run_produces_inr_deals_offline(tmp_path, monkeypatch, no_network):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(inr_store, "_encoder", FakeEncoder())
    framework = DealAgentFramework(settings=SETTINGS, fixture=True)
    memory = framework.run()
    assert memory, "fixture run should surface at least one deal"
    for opp in memory:
        assert opp.currency == "INR" and opp.deal.currency == "INR"
        assert opp.discount >= 500 and opp.discount_pct >= 20
        assert opp.confidence in ("low", "medium", "high")
    assert framework.planner.messenger.dry_run
    assert (tmp_path / "data" / "fixture_run" / "memory.json").exists()
    assert not (tmp_path / "memory.json").exists(), "fixture runs must not touch the real memory.json"
