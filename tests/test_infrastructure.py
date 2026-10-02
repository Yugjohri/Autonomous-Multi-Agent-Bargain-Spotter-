import json

import pytest

from agents.config import deep_merge, load_settings, load_sources_config
from agents.http import HttpClient, OfflineError
from agents.observations import ObservationLog, observation_row
from agents.normalize import normalize_link
from agents.sources.base import DealSource, RawDeal


class StubResponse:
    def __init__(self, status, text=""):
        self.status_code, self.text = status, text


class StubSession:
    def __init__(self, robots_status, robots_text=""):
        self.robots = StubResponse(robots_status, robots_text)
        self.headers = {}

    def get(self, url, **kwargs):
        return self.robots


@pytest.mark.parametrize(
    "status, text, url, allowed",
    [
        (200, "User-agent: *\nDisallow: /", "https://ddime.in/abc", False),
        (200, "User-agent: *\nDisallow: /goto/", "https://www.desidime.com/new", True),
        (200, "User-agent: *\nDisallow: /goto/", "https://www.desidime.com/goto/123", False),
        (200, "User-agent: Bingbot\nDisallow: /", "https://amzlinks.in/x", True),
        (403, "", "https://fkrt.cc/x", True),  # RFC 9309: 4xx means no restrictions
        (404, "", "https://grbn.in/x", True),
        (503, "", "https://example.com/x", False),  # 5xx means assume disallowed
    ],
)
def test_robots_rules(status, text, url, allowed):
    client = HttpClient(min_interval=0)
    client.session = StubSession(status, text)
    assert client.allowed(url) is allowed


def test_offline_client_never_requests():
    client = HttpClient(offline=True)
    with pytest.raises(OfflineError):
        client.get("https://t.me/s/OMGDeals")


def test_failing_source_is_isolated():
    class Broken(DealSource):
        def _fetch(self, limit):
            raise ConnectionError("down")

    source = Broken("broken")
    assert source.fetch(10) == []
    assert "ConnectionError" in source.last_error


def test_local_sources_override_base(tmp_path):
    base = tmp_path / "sources.yaml"
    local = tmp_path / "sources.local.yaml"
    base.write_text("telegram:\n  live: true\n  channels:\n    - username: example\n", encoding="utf-8")
    local.write_text("telegram:\n  channels:\n    - username: OMGDeals\n    - id: -100123\n", encoding="utf-8")
    config = load_sources_config(base, local)
    assert config["telegram"]["live"] is True
    assert [c.get("username") or c.get("id") for c in config["telegram"]["channels"]] == ["OMGDeals", -100123]
    assert load_sources_config(base, tmp_path / "missing.yaml")["telegram"]["channels"] == [{"username": "example"}]


def test_settings_env_overrides(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("pricer_mode: inr\nthresholds:\n  min_discount_inr: 500\n", encoding="utf-8")
    settings = load_settings(path, env={"MIN_DISCOUNT_INR": "750", "PRICER_MODE": "usd_legacy"})
    assert settings["thresholds"]["min_discount_inr"] == 750.0
    assert settings["pricer_mode"] == "usd_legacy"
    with pytest.raises(ValueError):
        load_settings(path, env={"PRICER_MODE": "eur"})


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1}}
    merged = deep_merge(base, {"a": {"c": 2}})
    assert merged == {"a": {"b": 1, "c": 2}} and base == {"a": {"b": 1}}


def test_observation_log_is_append_only_and_idempotent(tmp_path):
    path = tmp_path / "observations.jsonl"
    raw = RawDeal(title="Mivi Play speaker", text="Mivi Play speaker @264", url="https://www.amazon.in/dp/B08FTB3CCK?tag=x",
                  price_hint=264, source="telegram:DCLootsOffers", external_id="telegram:DCLootsOffers/5")
    row = observation_row(raw, normalize_link(raw.url))
    log = ObservationLog(path)
    assert log.append([row]) == 1
    assert ObservationLog(path).append([row]) == 0  # a fresh instance reads existing ids from disk
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["canonical_id"] == "amazon_in:B08FTB3CCK"
    assert rows[0]["price_inr"] == 264
    assert rows[0]["brand"] == "Mivi" and rows[0]["category"] == "Audio"
    assert rows[0]["is_deal"] is True
