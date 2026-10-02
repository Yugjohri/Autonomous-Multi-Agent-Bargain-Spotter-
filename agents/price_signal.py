"""
INR valuation from several weak signals, with an honest confidence label.

Signals (weights from settings.yaml, ensemble.inr):
  frontier  GPT's estimate of the typical selling price in India, given similar INR items
  market    median price of close neighbours in products_inr (needs a few close items)
  mrp       MRP scaled down by mrp_factor; MRPs in India are often inflated, so this is weak

The estimate is the weighted mean of the signals that are available. Confidence depends
on how many independent signals exist and whether they agree, not on the discount size.
"""

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agents.inr_store import Similar
from agents.money import format_money

DEFAULT_WEIGHTS = {"frontier": 0.6, "market": 0.3, "mrp": 0.1}
AGREE = 0.25  # signals within 25% of each other agree
LOOSE = 0.40


@dataclass
class Valuation:
    estimate: float
    discount: float
    discount_pct: float
    confidence: str
    reason: str
    signals: Dict[str, float] = field(default_factory=dict)


@dataclass
class MarketEvidence:
    median: float
    count: int
    low: float
    high: float


def market_evidence(similars: List[Similar], max_distance: float = 0.45, min_items: int = 3) -> Optional[MarketEvidence]:
    close = [s.price for s in similars if s.distance <= max_distance and s.price > 0]
    if len(close) < min_items:
        return None
    close.sort()
    return MarketEvidence(statistics.median(close), len(close), close[0], close[-1])


def _gap(a: float, b: float) -> float:
    return abs(a - b) / max(a, b)


def value_inr(
    price: float,
    mrp: Optional[float] = None,
    llm_estimate: Optional[float] = None,
    similars: Optional[List[Similar]] = None,
    weights: Optional[Dict[str, float]] = None,
    mrp_factor: float = 0.75,
    market_max_distance: float = 0.45,
    market_min_items: int = 3,
    sources_count: int = 1,
) -> Valuation:
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    signals: Dict[str, float] = {}
    notes: List[str] = []

    if llm_estimate and llm_estimate > 0:
        signals["frontier"] = float(llm_estimate)
        notes.append(f"GPT {format_money(llm_estimate)}")
    market = market_evidence(similars or [], market_max_distance, market_min_items)
    if market:
        signals["market"] = float(market.median)
        notes.append(f"{market.count} similar listings median {format_money(market.median)}")
    if mrp and mrp > price:
        signals["mrp"] = float(mrp) * mrp_factor
        notes.append(f"MRP {format_money(mrp)} x{mrp_factor:g}")

    usable = {k: v for k, v in signals.items() if weights.get(k, 0) > 0}
    if not usable:
        return Valuation(price, 0.0, 0.0, "low", "no valuation signal available", signals)

    total = sum(weights[k] for k in usable)
    # Whole rupees: a blended estimate is not precise to the paisa.
    estimate = round(sum(weights[k] * v for k, v in usable.items()) / total)
    discount = estimate - price
    discount_pct = round(100 * discount / estimate, 1) if estimate > 0 else 0.0

    confidence = _confidence(usable, sources_count)
    agreement = _agreement_note(usable)
    reason = "; ".join(notes) + (f"; {agreement}" if agreement else "")
    return Valuation(round(estimate, 2), round(discount, 2), discount_pct, confidence, reason, signals)


def _confidence(signals: Dict[str, float], sources_count: int) -> str:
    strong = {k: v for k, v in signals.items() if k in ("frontier", "market")}
    if len(strong) == 2:
        gap = _gap(strong["frontier"], strong["market"])
        if gap <= AGREE:
            return "high"
        if gap <= LOOSE:
            return "medium"
        return "low"
    if len(signals) >= 2:
        values = list(signals.values())
        if _gap(values[0], values[1]) <= AGREE:
            return "medium"
        return "low"
    # A single signal: only a repeatedly posted deal earns medium.
    if "frontier" in signals and sources_count >= 3:
        return "medium"
    return "low"


def _agreement_note(signals: Dict[str, float]) -> str:
    if "frontier" in signals and "market" in signals:
        gap = _gap(signals["frontier"], signals["market"])
        return "GPT and market agree" if gap <= AGREE else f"GPT and market differ by {gap:.0%}"
    if len(signals) == 1:
        return f"only one signal ({next(iter(signals))})"
    return ""
