"""
INR valuation from several weak signals, with an honest confidence label.

Signals (weights from settings.yaml, ensemble.inr):
  frontier        GPT's estimate of the typical selling price in India, given similar INR items
  market          median price of close neighbours in products_inr (needs a few close items)
  neural_network  the INR model trained by scripts/train_nn_inr.py (off unless it has a weight)
  specialist      Llama 3.2 3B fine-tuned on INR prices (scripts/train_specialist_inr.py; off unless weighted)
  mrp             MRP scaled down by mrp_factor; MRPs in India are often inflated, so this is weak

The estimate is the weighted mean of the available signals that have a weight. Market and
MRP are always computed, so a signal with weight 0 is still a check: confidence depends on
how many signals exist and whether they agree, not on the discount size or the weights.
High confidence needs GPT and real market listings to agree; any other two strong signals
that agree (for example GPT and the neural network, when no close listings exist) give medium.
The neural network is only consulted when it has a weight.

docs/training_report.md has the evaluation behind the weights in settings.yaml: on held-out
data GPT alone gave the best estimate, and adding market, MRP or the network did not help.
"""

import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agents.inr_store import Similar
from agents.money import format_money

DEFAULT_WEIGHTS = {"frontier": 0.6, "market": 0.3, "mrp": 0.1, "neural_network": 0.0, "specialist": 0.0}
STRONG = ("frontier", "market", "neural_network", "specialist")
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


# Combos and multipacks: similar listings and GPT usually price a single unit, so the
# comparison is unreliable. "12-in-1", "3 jars" or "4GB" are not multipacks.
MULTIPACK_RE = re.compile(
    r"\bcombo\b|\bmulti-?pack\b|\bbundle\b|\b(pack|set) of\s*(?:[2-9]|\d{2,})\b|"
    r"\b(?:[2-9]|\d{2,})\s*-?\s*(?:pcs|pieces|pack|units|packs)\b|"
    # "2 x 750ml" is a multipack; "2560x1440" or "6x4 inches" is a size.
    r"\b(?:[2-9]|1\d)\s*x\s*\d+(?:\.\d+)?\s*(?:ml|l|ltr|litres?|g|gm|kg|pcs|pieces|tablets|capsules|sheets|rolls)\b|"
    r"\bbuy\s*\d+\s*get\s*\d+\b|\bb\d+g\d+\b",
    re.IGNORECASE,
)


def is_multipack(text: str) -> bool:
    return bool(MULTIPACK_RE.search(text or ""))


def cap_for_multipack(valuation: "Valuation", text: str) -> "Valuation":
    """Combos and multipacks stay visible but never above low confidence."""
    if is_multipack(text) and valuation.confidence != "low":
        valuation.confidence = "low"
        valuation.reason = f"combo or multipack, per-unit comparison unreliable; {valuation.reason}"
    elif is_multipack(text) and "multipack" not in valuation.reason:
        valuation.reason = f"combo or multipack; {valuation.reason}"
    return valuation


def cap_for_listing(valuation: "Valuation", url: str) -> "Valuation":
    """
    A deal whose link is a search, category or sale page ("Upto 88% off on DANIEL KLEIN
    watches") has no single product behind its price, so it stays visible but never above
    low confidence, like a multipack.
    """
    from agents.normalize import is_search_or_listing

    if is_search_or_listing(url or ""):
        valuation.confidence = "low"
        valuation.reason = f"links to a search or sale page, not one product; {valuation.reason}"
    return valuation


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
    nn_estimate: Optional[float] = None,
    specialist_estimate: Optional[float] = None,
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
    if nn_estimate and nn_estimate > 0 and weights.get("neural_network", 0) > 0:
        signals["neural_network"] = float(nn_estimate)
        notes.append(f"model {format_money(nn_estimate)}")
    if specialist_estimate and specialist_estimate > 0 and weights.get("specialist", 0) > 0:
        signals["specialist"] = float(specialist_estimate)
        notes.append(f"specialist {format_money(specialist_estimate)}")
    if mrp and mrp > price:
        signals["mrp"] = float(mrp) * mrp_factor
        notes.append(f"MRP {format_money(mrp)} x{mrp_factor:g}")

    if not signals:
        return Valuation(price, 0.0, 0.0, "low", "no valuation signal available", signals)
    # Signals with a weight make the estimate; the others still count as checks for confidence.
    usable = {k: v for k, v in signals.items() if weights.get(k, 0) > 0}
    used_weights = {k: weights[k] for k in usable}
    fallback = not usable
    if fallback:
        # No weighted signal (GPT failed, say): average whatever checks there are.
        usable, used_weights = dict(signals), {k: 1.0 for k in signals}

    total = sum(used_weights.values())
    # Whole rupees: a blended estimate is not precise to the paisa.
    estimate = round(sum(used_weights[k] * v for k, v in usable.items()) / total)
    discount = estimate - price
    discount_pct = round(100 * discount / estimate, 1) if estimate > 0 else 0.0

    confidence = _confidence(signals, sources_count)
    agreement = _agreement_note(signals)
    reason = "; ".join(notes) + (f"; {agreement}" if agreement else "")
    if fallback:
        reason += "; estimate from checks only"
    return Valuation(round(estimate, 2), round(discount, 2), discount_pct, confidence, reason, signals)


def _confidence(signals: Dict[str, float], sources_count: int) -> str:
    if "frontier" in signals and "market" in signals:
        gap = _gap(signals["frontier"], signals["market"])
        if gap <= AGREE:
            return "high"
        if gap <= LOOSE:
            return "medium"
        return "low"
    strong = [v for k, v in signals.items() if k in STRONG]
    if len(strong) >= 2:
        return "medium" if _gap(max(strong), min(strong)) <= AGREE else "low"
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
    if "frontier" in signals and "neural_network" in signals:
        gap = _gap(signals["frontier"], signals["neural_network"])
        return "GPT and model agree" if gap <= AGREE else f"GPT and model differ by {gap:.0%}"
    if len(signals) == 1:
        return f"only one signal ({next(iter(signals))})"
    return ""
