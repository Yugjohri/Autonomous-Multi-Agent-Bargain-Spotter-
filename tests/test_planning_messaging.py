import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from agents.deals import Deal, DealSelection, Opportunity
from agents.messaging_agent import MessagingAgent, format_alert
from agents.planning_agent import PlanningAgent
from agents.price_signal import Valuation

SETTINGS = {
    "pricer_mode": "inr",
    "thresholds": {"min_discount_inr": 500, "min_discount_pct": 20, "min_discount_usd": 50},
    "planning": {"top_n": 3, "category_cooldown_minutes": 60, "alert_min_confidence": "medium"},
}
NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


def opp(price, estimate, category="Audio", confidence="high", currency="INR", found_at=None, title="x"):
    deal = Deal(product_description=title, title=title, price=price, url=f"https://www.amazon.in/dp/B0{price:08d}",
                category=category, currency=currency, store="amazon_in")
    discount = estimate - price
    return Opportunity(deal=deal, estimate=estimate, discount=discount, discount_pct=round(100 * discount / estimate, 1),
                       confidence=confidence, found_at=found_at)


class Recorder:
    def __init__(self):
        self.alerts = []

    def alert(self, o):
        self.alerts.append(o)


def planner(settings=SETTINGS, scanner=None, ensemble=None):
    return PlanningAgent(settings=settings, scanner=scanner or object(), ensemble=ensemble or object(), messenger=Recorder())


@pytest.mark.parametrize(
    "price, estimate, ok",
    [
        (1000, 2000, True),  # ₹1,000 and 50%
        (100, 400, False),  # 75% but only ₹300
        (9000, 10000, False),  # ₹1,000 but only 10%
        (1600, 2000, False),  # 20% but only ₹400
    ],
)
def test_inr_threshold_needs_both_rupees_and_percent(price, estimate, ok):
    assert planner().qualifies(opp(price, estimate)) is ok


def test_threshold_boundaries():
    p = planner()
    assert p.qualifies(opp(2000, 2500))  # exactly ₹500 and 20%
    assert not p.qualifies(opp(2001, 2500))


def test_legacy_usd_rule():
    p = planner(settings={**SETTINGS, "pricer_mode": "usd_legacy"})
    assert p.qualifies(opp(100, 151, currency="USD"))
    assert not p.qualifies(opp(100, 150, currency="USD"))
    assert p.top_n() == 1


def test_top_n_sorted_by_percent_then_amount_with_one_per_category():
    opps = [
        opp(1000, 2000, "Audio", title="a50"),  # 50%
        opp(5000, 10000, "Mobiles", title="m50"),  # 50%, bigger amount
        opp(1000, 4000, "Audio", title="a75"),  # 75%
        opp(3000, 5000, "Kitchen", title="k40"),
        opp(2000, 3000, "Fashion", title="f33"),
    ]
    chosen = planner().select(opps, memory=[], now=NOW)
    assert [o.deal.title for o in chosen] == ["a75", "m50", "k40"]


def test_category_cooldown_uses_memory():
    recent = opp(500, 2000, "Audio", found_at=NOW - timedelta(minutes=20))
    old = opp(500, 2000, "Mobiles", found_at=NOW - timedelta(hours=3))
    chosen = planner().select([opp(1000, 3000, "Audio", title="a"), opp(1000, 3000, "Mobiles", title="m")], [recent, old], now=NOW)
    assert [o.deal.title for o in chosen] == ["m"]


def test_plan_alerts_only_confident_deals_and_returns_list():
    deals = [Deal(product_description=t, title=t, price=1000, url=f"https://www.amazon.in/dp/B0000000{i}{i}", category=c)
             for i, (t, c) in enumerate([("hi", "Audio"), ("lo", "Mobiles")])]
    scanner = SimpleNamespace(scan=lambda memory, extra=None, fetch=True: DealSelection(deals=deals), last_candidates=[])
    ensemble = SimpleNamespace(value=lambda d: Valuation(3000, 2000, 66.7, "high" if d.title == "hi" else "low", "r"))
    p = planner(scanner=scanner, ensemble=ensemble)
    result = p.plan(memory=[])
    assert [o.deal.title for o in result] == ["hi", "lo"]
    assert [o.deal.title for o in p.messenger.alerts] == ["hi"]
    assert all(o.found_at is not None and o.confidence for o in result)


# ------------------------------------------------------------------ messaging


def test_alert_format_inr():
    o = opp(129999, 164999, confidence="medium", title="Apple iPhone 15 Pro 128GB")
    o.deal.mrp = 179900
    o.deal.posted_at = NOW - timedelta(minutes=12)
    o.deal.seen_in = ["telegram:a", "telegram:b"]
    o.deal.coupon_note = "Extra ₹3,000 off with HDFC cards"
    text = format_alert(o, now=NOW)
    assert "₹1,29,999 (MRP ₹1,79,900), worth about ₹1,64,999" in text
    assert "21% / ₹35,000 below typical, medium confidence" in text
    assert "Amazon, posted 12 min ago, in 2 channels" in text
    assert "HDFC" in text and text.endswith(o.deal.url)
    assert "$" not in text


def test_alert_format_legacy_usd_unchanged():
    o = opp(350, 773, currency="USD", title="Galaxy Watch")
    assert format_alert(o).startswith("Deal Alert! Price=$350.00, Estimate=$773.00, Discount=$423.00")


def test_dry_run_sends_nothing(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("agents.messaging_agent.requests.post", lambda *a, **k: calls.append(a))
    monkeypatch.setenv("PUSHOVER_USER", "u")
    monkeypatch.setenv("PUSHOVER_TOKEN", "t")
    agent = MessagingAgent(dry_run=True, sent_path=tmp_path / 'sent.json')
    agent.alert(opp(1000, 2000))
    assert calls == [] and len(agent.sent) == 1


def test_sends_to_pushover_and_telegram_bot(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("agents.messaging_agent.requests.post", lambda url, data=None, timeout=None: calls.append((url, data)))
    for key, value in {"PUSHOVER_USER": "u", "PUSHOVER_TOKEN": "t", "TG_BOT_TOKEN": "123:abc", "TG_CHAT_ID": "42"}.items():
        monkeypatch.setenv(key, value)
    MessagingAgent(sent_path=tmp_path / 'sent.json').alert(opp(1000, 2000))
    assert calls[0][0] == "https://api.pushover.net/1/messages.json"
    assert calls[1][0] == "https://api.telegram.org/bot123:abc/sendMessage" and calls[1][1]["chat_id"] == "42"


def test_unconfigured_targets_are_skipped(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("agents.messaging_agent.requests.post", lambda *a, **k: calls.append(a))
    for key in ("PUSHOVER_USER", "PUSHOVER_TOKEN", "TG_BOT_TOKEN", "TG_CHAT_ID"):
        monkeypatch.delenv(key, raising=False)
    MessagingAgent(sent_path=tmp_path / 'sent.json').alert(opp(1000, 2000))
    assert calls == []


# ------------------------------------------------------------ autonomous agent


def test_autonomous_agent_tool_loop_inr():
    from agents.autonomous_planning_agent import AutonomousPlanningAgent

    deal = Deal(product_description="Portable speaker", title="Mivi Play", price=999, url="https://www.amazon.in/dp/B08FTB3CCK")
    scanner = SimpleNamespace(scan=lambda memory: DealSelection(deals=[deal]))
    ensemble = SimpleNamespace(value=lambda d: Valuation(1999, 1000, 50.0, "high", "GPT and market agree"))
    notified = []
    messenger = SimpleNamespace(notify=lambda *a: notified.append(a))

    def call(name, args):
        return SimpleNamespace(id=name, function=SimpleNamespace(name=name, arguments=json.dumps(args)))

    script = [
        [call("scan_the_internet_for_bargains", {})],
        [call("estimate_true_value", {"deal_number": 0})],
        [call("notify_user_of_deal", {"deal_number": 0})],
        None,
    ]
    tool_outputs = []

    def create(model, messages, tools):
        tool_outputs.extend(m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "tool")
        step = script.pop(0)
        if step is None:
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="OK"))])
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(tool_calls=step, content=None))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    agent = AutonomousPlanningAgent(settings=SETTINGS, scanner=scanner, ensemble=ensemble, messenger=messenger, client=client)
    result = agent.plan(memory=[])
    assert result.discount_pct == 50.0 and result.currency == "INR"
    assert notified and notified[0][-1] == "INR"
    outputs = " ".join(tool_outputs)
    assert '"price": "₹999"' in outputs and "estimated typical price ₹1,999, 50% below, high confidence" in outputs


# ------------------------------------------------------------------------- UI


def test_ui_rows_and_store_filter():
    from price_is_right import ALL_STORES, HEADERS, table_for

    a = opp(1099, 1999, title="Mivi Play")
    a.deal.store, a.deal.mrp, a.deal.source = "amazon_in", 4490, "telegram:DCLootsOffers"
    a.deal.seen_in = ["telegram:DCLootsOffers", "telegram:OMGDeals"]
    b = opp(8988, 15000, title="Air fryer")
    b.deal.store = "flipkart"
    rows = table_for([a, b], ALL_STORES)
    assert len(rows[0]) == len(HEADERS)
    assert rows[1][:3] == ["Mivi Play", "₹1,099", "₹4,490"]
    assert rows[1][8] == "telegram:DCLootsOffers +1"
    assert [r[0] for r in table_for([a, b], "Flipkart")] == ["Air fryer"]


def test_plot_colors_have_other_bucket():
    from deal_agent_framework import OTHER_COLOR, color_for

    assert color_for("Audio") != OTHER_COLOR
    assert color_for("Something new") == OTHER_COLOR
    assert color_for(None) == OTHER_COLOR
    assert color_for("Electronics", mode="usd_legacy") == "orange"
    assert color_for("Mobiles", mode="usd_legacy") == OTHER_COLOR


def test_same_deal_is_not_alerted_twice(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("agents.messaging_agent.requests.post", lambda url, data=None, timeout=None: calls.append(url))
    monkeypatch.setenv("PUSHOVER_USER", "u")
    monkeypatch.setenv("PUSHOVER_TOKEN", "t")
    for key in ("TG_BOT_TOKEN", "TG_CHAT_ID"):
        monkeypatch.delenv(key, raising=False)
    sent = tmp_path / "sent.json"
    deal = opp(1000, 2000)
    assert MessagingAgent(sent_path=sent).alert(deal) is True
    # A burst of repeats (e.g. UI events) and a restarted bot both stay silent.
    assert [MessagingAgent(sent_path=sent).alert(deal) for _ in range(20)] == [False] * 20
    assert len(calls) == 1
    cheaper = opp(900, 2000)
    cheaper.deal.url = deal.deal.url
    assert MessagingAgent(sent_path=sent).alert(cheaper) is True, "a price drop is a new alert"
    assert MessagingAgent(sent_path=sent).alert(deal, force=True) is True, "the UI button can resend"


def test_dry_run_does_not_silence_later_live_alerts(tmp_path):
    sent = tmp_path / "sent.json"
    dry = MessagingAgent(dry_run=True, sent_path=sent)
    deal = opp(1000, 2000)
    assert dry.alert(deal) is True and dry.alert(deal) is False
    assert not sent.exists()


def test_ui_hides_other_currency():
    from price_is_right import table_for

    rows = table_for([opp(350, 773, currency="USD", title="old"), opp(1000, 2000, title="new")], currency="INR")
    assert [r[0] for r in rows] == ["new"]
