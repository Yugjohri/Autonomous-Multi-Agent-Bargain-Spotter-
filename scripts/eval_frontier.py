"""
Evaluate the Frontier Agent (GPT with INR RAG) and the store's market signal.

    uv run python scripts/eval_frontier.py [--test-n 200] [--val-n 300] [--max-cost 3.0] [--no-mrp-check]

Uses the app's own path: InrProductStore.similar() (8 neighbours, the item's own id
excluded) and FrontierAgent.estimate_inr() with the first 5 as context. Run it after the
store is rebuilt from the train split only (see populate_vectorstore.py), so no validation
or test product is in the context.

    frontier       fixed random samples of test (--test-n) and val (--val-n) items
    frontier_mrp   the same test items with "MRP: Rs X" added to the description, to check
                   whether GPT echoes the MRP instead of estimating the selling price
    market         median of close neighbours (agents.price_signal.market_evidence), all val and
                   test items; missing when fewer than market_min_items are close enough

Answers are cached in data/cache/frontier_eval.json by prompt, so reruns are free. Spend is
logged and the run stops before --max-cost is exceeded.
"""

import argparse
import hashlib
import json
import sys
import threading

import numpy as np

from inr_common import DATA, ROOT, load_splits, save_predictions, seed_everything

from agents.config import get, load_settings
from agents.evaluator_inr import evaluate, metrics
from agents.money import format_money
from agents.price_signal import market_evidence

CACHE = ROOT / "data" / "cache" / "frontier_eval.json"


class BudgetExceeded(RuntimeError):
    pass


class MeteredClient:
    """Wraps the OpenAI client: caches answers by request, sums the cost, enforces a cap."""

    def __init__(self, client, prices, max_cost: float):
        self.client = client
        self.prices = prices
        self.max_cost = max_cost
        self.cost = 0.0
        self.calls = 0
        self.lock = threading.Lock()
        self.cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        from types import SimpleNamespace

        key = hashlib.sha1(json.dumps([kwargs["model"], kwargs["messages"]], sort_keys=True).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                content = self.cache[key]
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
            if self.cost >= self.max_cost:
                raise BudgetExceeded(f"spent ${self.cost:.3f}, cap ${self.max_cost}")
        response = self.client.chat.completions.create(**kwargs)
        usage = response.usage
        with self.lock:
            self.calls += 1
            if usage is not None:
                self.cost += (usage.prompt_tokens * self.prices[0] + usage.completion_tokens * self.prices[1]) / 1e6
            self.cache[key] = response.choices[0].message.content
        return response

    def save(self):
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(self.cache), encoding="utf-8")


def mrp_description(row: dict) -> str:
    mrp = row.get("mrp_inr")
    return f"{row['text']}\nMRP: {format_money(mrp)}" if mrp and mrp == mrp else row["text"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-n", type=int, default=200)
    parser.add_argument("--val-n", type=int, default=300)
    parser.add_argument("--max-cost", type=float, default=3.0, help="USD cap for new API calls")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-mrp-check", action="store_true")
    args = parser.parse_args()
    seed_everything(42)

    from dotenv import load_dotenv
    from openai import OpenAI

    from agents.frontier_agent import FrontierAgent
    from agents.inr_store import InrProductStore

    load_dotenv(ROOT / ".env", override=True)
    settings = load_settings()
    model = get(settings, "models.frontier", "gpt-5.1")
    prices = get(settings, f"models.prices_per_million.{model}", [1.25, 10.0])
    weights = get(settings, "ensemble.inr", {})
    client = MeteredClient(OpenAI(), prices, args.max_cost)
    frontier = FrontierAgent(client=client, model=model)
    store = InrProductStore(path=str(ROOT / "products_vectorstore"))

    splits = load_splits()
    samples = {
        "test": splits["test"].sample(n=min(args.test_n, len(splits["test"])), random_state=42),
        "val": splits["val"].sample(n=min(args.val_n, len(splits["val"])), random_state=42),
    }
    context = {}

    def similars_for(row):
        key = row["product_key"]
        if key not in context:
            context[key] = store.similar(row["text"], n=8, exclude_canonical=key)
        return context[key]

    # Market signal for every val and test item (free).
    report = {"model": model}
    for split in ("val", "test"):
        df = splits[split]
        preds = []
        for row in df.to_dict("records"):
            evidence = market_evidence(similars_for(row), float(weights.get("market_max_distance", 0.45)),
                                       int(weights.get("market_min_items", 3)))
            preds.append(evidence.median if evidence else np.nan)
        save_predictions("market", split, df, preds)
        covered = float(np.mean(np.isfinite(preds)))
        report[f"market_coverage_{split}"] = covered
        print(f"market {split}: coverage {100 * covered:.0f}%, {metrics(df['price_inr'], preds)}")

    runs = [("frontier", "test", lambda r: frontier.estimate_inr(r["text"], similars_for(r)[:5])),
            ("frontier", "val", lambda r: frontier.estimate_inr(r["text"], similars_for(r)[:5]))]
    if not args.no_mrp_check:
        runs.append(("frontier_mrp", "test", lambda r: frontier.estimate_inr(mrp_description(r), similars_for(r)[:5])))
    try:
        for name, split, predictor in runs:
            df = samples[split]
            preds = evaluate(predictor, df, workers=args.workers, verbose=False)
            save_predictions(name, split, df, preds)
            client.save()
            print(f"{name} {split}: {metrics(df['price_inr'], preds)}  (spent ${client.cost:.3f} in {client.calls} calls)")
    except BudgetExceeded as exc:
        print(f"Stopped: {exc}")
    finally:
        client.save()

    # What the model saw: MRPs in the RAG context, for the echo analysis in the report.
    rows = []
    for split, df in samples.items():
        for row in df.to_dict("records"):
            sims = similars_for(row)[:5]
            rows.append({"split": split, "product_key": row["product_key"],
                         "context_mrps": [s.mrp for s in sims if s.mrp],
                         "context_prices": [s.price for s in sims]})
    (DATA / "frontier_context.json").write_text(json.dumps(rows), encoding="utf-8")
    report.update(new_calls=client.calls, cost_usd=round(client.cost, 4))
    (DATA / "frontier_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"New API calls: {client.calls}, cost about ${client.cost:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
