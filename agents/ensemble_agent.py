from typing import Dict, Optional

from agents.agent import Agent
from agents.config import get, load_settings
from agents.deals import Deal
from agents.inr_store import InrItem, InrProductStore
from agents.money import format_money
from agents.price_signal import Valuation, value_inr

LEGACY_WEIGHTS = {"frontier": 0.8, "specialist": 0.1, "neural_network": 0.1}


class EnsembleAgent(Agent):
    """
    Values deals. Two modes, chosen by PRICER_MODE (settings.yaml):

    inr         the Frontier Agent estimates the typical Indian selling price in INR with RAG
                over products_inr, and the price-signal module blends it with the market median
                and the MRP. The USD-trained Specialist and Neural Network are never used here.
    usd_legacy  the original blend of Specialist (fine-tuned Llama on Modal), Frontier
                (GPT + RAG over the USD "products" collection) and the local Neural Network.

    Weights come from settings.yaml, so trained INR models can be added as new signals later.
    """

    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection=None, settings: Optional[dict] = None, inr_store: Optional[InrProductStore] = None,
                 frontier=None, offline: bool = False):
        """
        Create an instance of Ensemble, by creating each of the models
        And loading the weights of the Ensemble
        """
        self.log("Initializing Ensemble Agent")
        self.settings = settings if settings is not None else load_settings()
        self.mode = self.settings.get("pricer_mode", "inr")
        self.offline = offline
        if self.mode == "usd_legacy":
            from agents.frontier_agent import FrontierAgent
            from agents.neural_network_agent import NeuralNetworkAgent
            from agents.preprocessor import Preprocessor
            from agents.specialist_agent import SpecialistAgent

            self.weights: Dict[str, float] = {**LEGACY_WEIGHTS, **(get(self.settings, "ensemble.usd_legacy", {}) or {})}
            self.specialist = SpecialistAgent()
            self.frontier = FrontierAgent(collection)
            self.neural_network = NeuralNetworkAgent()
            self.preprocessor = Preprocessor()
        else:
            self.weights = dict(get(self.settings, "ensemble.inr", {}) or {})
            self.inr_store = inr_store or InrProductStore()
            self.frontier = frontier
            if self.frontier is None and not offline:
                from agents.frontier_agent import FrontierAgent

                self.frontier = FrontierAgent(model=get(self.settings, "models.frontier", "gpt-5.1"))
        self.log(f"Ensemble Agent is ready ({self.mode} mode)")

    # ------------------------------------------------------------------ legacy

    def price(self, description: str) -> float:
        """
        Run this ensemble model
        Ask each of the models to price the product
        Then use the Linear Regression model to return the weighted price
        :param description: the description of a product
        :return: an estimate of its price
        """
        if self.mode != "usd_legacy":
            raise RuntimeError("EnsembleAgent.price() is the USD pipeline; use value() in INR mode")
        self.log("Running Ensemble Agent - preprocessing text")
        rewrite = self.preprocessor.preprocess(description)
        self.log(f"Pre-processed text using {self.preprocessor.model_name}")
        specialist = self.specialist.price(rewrite)
        frontier = self.frontier.price(rewrite)
        neural_network = self.neural_network.price(rewrite)
        w = self.weights
        combined = frontier * w["frontier"] + specialist * w["specialist"] + neural_network * w["neural_network"]
        self.log(f"Ensemble Agent complete - returning ${combined:.2f}")
        return combined

    # --------------------------------------------------------------------- INR

    @staticmethod
    def text_for(deal: Deal) -> str:
        return f"{deal.title}\n{deal.product_description}".strip()

    def value(self, deal: Deal) -> Valuation:
        """Estimate what a deal is worth and how sure we are."""
        if self.mode == "usd_legacy":
            estimate = self.price(deal.product_description)
            discount = estimate - deal.price
            pct = round(100 * discount / estimate, 1) if estimate > 0 else 0.0
            return Valuation(estimate, discount, pct, "medium", "USD ensemble (Specialist, Frontier, Neural Network)")

        text = self.text_for(deal)
        similars = self.inr_store.similar(text, n=8, exclude_canonical=deal.canonical_id)
        llm_estimate = None
        if self.frontier is not None:
            try:
                llm_estimate = self.frontier.estimate_inr(text, similars[:5])
            except Exception as exc:  # noqa: BLE001 - one failed call should not lose the deal
                self.log(f"Frontier Agent failed ({exc}); valuing without it")
        valuation = value_inr(
            price=deal.price,
            mrp=deal.mrp,
            llm_estimate=llm_estimate,
            similars=similars,
            weights={k: self.weights[k] for k in ("frontier", "market", "mrp") if k in self.weights},
            mrp_factor=float(self.weights.get("mrp_factor", 0.75)),
            market_max_distance=float(self.weights.get("market_max_distance", 0.45)),
            market_min_items=int(self.weights.get("market_min_items", 3)),
            sources_count=len(deal.seen_in) or 1,
        )
        direction = "below" if valuation.discount_pct >= 0 else "above"
        self.log(
            f"Ensemble Agent valued {format_money(deal.price)} deal at {format_money(valuation.estimate)} "
            f"({abs(valuation.discount_pct):.0f}% {direction}, {valuation.confidence} confidence: {valuation.reason})"
        )
        return valuation

    def remember(self, deals) -> int:
        """Add scraped INR deals to products_inr so future valuations have more price context."""
        if self.mode == "usd_legacy":
            return 0
        items = [
            InrItem(
                document=self.text_for(d) if d.title or d.product_description else "",
                price=d.price,
                canonical_id=d.canonical_id,
                mrp=d.mrp,
                store=d.store,
                category=d.category,
                seen_at=d.posted_at.isoformat() if d.posted_at else None,
                source=d.source,
            )
            for d in deals
            if d.currency == "INR"
        ]
        return self.inr_store.add(items)
