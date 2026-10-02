from datetime import datetime, timedelta, timezone
from typing import List, Optional

from agents.agent import Agent
from agents.config import get, load_settings
from agents.deals import Deal, Opportunity
from agents.money import format_money
from agents.sources.base import RawDeal

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


class PlanningAgent(Agent):

    name = "Planning Agent"
    color = Agent.GREEN

    def __init__(self, collection=None, settings: Optional[dict] = None, scanner=None, ensemble=None,
                 messenger=None, offline: bool = False, dry_run: bool = False, remember: bool = True):
        """
        Create instances of the 3 Agents that this planner coordinates across.
        Any of them can be injected (tests, fixture mode).
        """
        self.log("Planning Agent is initializing")
        self.settings = settings if settings is not None else load_settings()
        self.mode = self.settings.get("pricer_mode", "inr")
        self.remember = remember
        if scanner is None:
            from agents.observations import ObservationLog
            from agents.scanner_agent import ScannerAgent

            scanner = ScannerAgent(settings=self.settings, offline=offline, observations=ObservationLog())
        if ensemble is None:
            from agents.ensemble_agent import EnsembleAgent

            ensemble = EnsembleAgent(collection, settings=self.settings, offline=offline)
        if messenger is None:
            from agents.messaging_agent import MessagingAgent

            messenger = MessagingAgent(settings=self.settings, dry_run=dry_run)
        self.scanner, self.ensemble, self.messenger = scanner, ensemble, messenger
        self.log("Planning Agent is ready")

    # ------------------------------------------------------------- thresholds

    def qualifies(self, opp: Opportunity) -> bool:
        """INR: at least MIN_DISCOUNT_INR off AND at least MIN_DISCOUNT_PCT off. USD: legacy $50 rule."""
        if opp.currency == "INR":
            min_inr = float(get(self.settings, "thresholds.min_discount_inr", 500))
            min_pct = float(get(self.settings, "thresholds.min_discount_pct", 20))
            return opp.discount >= min_inr and (opp.discount_pct or 0) >= min_pct
        return opp.discount > float(get(self.settings, "thresholds.min_discount_usd", 50))

    def top_n(self) -> int:
        return 1 if self.mode == "usd_legacy" else int(get(self.settings, "planning.top_n", 3))

    def select(self, opportunities: List[Opportunity], memory: List[Opportunity], now: Optional[datetime] = None) -> List[Opportunity]:
        """
        Best qualifying opportunities, sorted by discount percent then absolute discount,
        at most one per category, skipping categories alerted within the cooldown window.
        """
        now = now or datetime.now(timezone.utc)
        cooldown = timedelta(minutes=float(get(self.settings, "planning.category_cooldown_minutes", 60)))
        cooling = {
            o.deal.category
            for o in memory
            if o.found_at and o.deal.category != "Other" and now - _aware(o.found_at) < cooldown
        }
        ranked = sorted(
            (o for o in opportunities if self.qualifies(o)),
            key=lambda o: (o.discount_pct or 0, o.discount),
            reverse=True,
        )
        chosen, used = [], set()
        for opp in ranked:
            category = opp.deal.category
            if category != "Other" and (category in cooling or category in used):
                self.log(f"Planning Agent skips a {category} deal (category cooldown)")
                continue
            chosen.append(opp)
            used.add(category)
            if len(chosen) >= self.top_n():
                break
        return chosen

    # ---------------------------------------------------------------- running

    def run(self, deal: Deal) -> Opportunity:
        """
        Run the workflow for a particular deal
        :param deal: the deal, summarized from a scan
        :returns: an opportunity including the discount and a confidence label
        """
        self.log("Planning Agent is pricing up a potential deal")
        valuation = self.ensemble.value(deal)
        self.log(
            f"Planning Agent has processed a deal with discount {format_money(valuation.discount, deal.currency)} "
            f"({valuation.discount_pct:.0f}%)"
        )
        return Opportunity(
            deal=deal,
            estimate=valuation.estimate,
            discount=valuation.discount,
            discount_pct=valuation.discount_pct,
            confidence=valuation.confidence,
            confidence_reason=valuation.reason,
            found_at=datetime.now(timezone.utc),
        )

    def plan(self, memory: List[Opportunity] = [], extra: Optional[List[RawDeal]] = None) -> List[Opportunity]:
        """
        Run the full workflow:
        1. Use the ScannerAgent to find deals from all sources
        2. Use the EnsembleAgent to estimate them
        3. Use the MessagingAgent to send notifications for the best few
        :param memory: Opportunities surfaced in the past
        :param extra: posts that arrived live since the last scan
        :return: the Opportunities surfaced in this run (possibly empty)
        """
        self.log("Planning Agent is kicking off a run")
        selection = self.scanner.scan(memory=memory, extra=extra)
        if self.remember and self.mode == "inr":
            candidates = getattr(self.scanner, "last_candidates", [])
            if candidates:
                added = self.ensemble.inr_store.add([c.to_inr_item() for c in candidates])
                self.log(f"Planning Agent added {added} priced items to the INR price store")
        if not selection or not selection.deals:
            return []
        opportunities = [self.run(deal) for deal in selection.deals]
        chosen = self.select(opportunities, memory)
        best = max(opportunities, key=lambda o: o.discount_pct or 0)
        self.log(f"Planning Agent has identified the best deal has discount {best.discount_pct:.0f}%")
        min_confidence = CONFIDENCE_RANK.get(get(self.settings, "planning.alert_min_confidence", "low"), 0)
        for opp in chosen:
            if CONFIDENCE_RANK.get(opp.confidence or "low", 0) >= min_confidence:
                self.messenger.alert(opp)
            else:
                self.log(f"Planning Agent shows but does not alert a {opp.confidence}-confidence deal")
        self.log(f"Planning Agent has completed a run with {len(chosen)} deal(s) surfaced")
        return chosen


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
