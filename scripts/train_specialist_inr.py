"""
Fine-tune Llama 3.2 3B (QLoRA) to estimate Indian prices in rupees: the INR Specialist.

    uv run python scripts/train_specialist_inr.py                      # all variants, then test
    uv run python scripts/train_specialist_inr.py --variants base --epochs 1
    uv run python scripts/train_specialist_inr.py --smoke              # a few steps, to check setup

Prompt: agents.items.inr_prompt ("What does this cost in India, in rupees?\\n\\n{text}\\n\\n
Price is Rs.") with the whole-rupee price as the completion; the loss is on the completion
only. 4-bit nf4 base, bf16 compute, LoRA on the attention and MLP projections.

Data variants (compared on validation, as asked in the addendum):
    base             the Phase A train split only (normal selling prices)
    telegram_median  base + Telegram products seen 2+ times, labelled with their median DEAL price
    telegram_p75     base + the same products labelled with their 75th percentile deal price
Telegram products come from data/labels.csv (scripts/build_labels.py) with a real Amazon or
Flipkart id; each product is one example however often it was reposted, and products in
the validation or test split are left out.

Every epoch is saved and scored on the validation split by generating prices (not by loss);
the best variant and epoch is copied to models/specialist_inr/ with specialist_meta.json,
then scored once on the test split. Progress: models/specialist_progress.txt.
"""

import argparse
import json
import math
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from inr_common import DATA, ROOT, load_splits, save_predictions, seed_everything

from agents.evaluator_inr import metrics
from agents.items import inr_completion, inr_prompt
from agents.normalize import is_product_id
from agents.titles import normalise_title

BASE_MODEL = "meta-llama/Llama-3.2-3B"
OUT = ROOT / "models" / "specialist_inr"
RUNS = ROOT / "models" / "specialist_runs"
PROGRESS = ROOT / "models" / "specialist_progress.txt"
VARIANTS = ("base", "telegram_median", "telegram_p75")
ANSWER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


# ------------------------------------------------------------------- data


def telegram_examples(label: str, held_out: pd.DataFrame, path: Path = ROOT / "data" / "labels.csv") -> pd.DataFrame:
    labels = pd.read_csv(path)
    labels = labels[labels["canonical_id"].map(lambda c: is_product_id(str(c)))]
    labels = labels[(labels["observations"] >= 2) & labels["title"].notna() & (labels[label] > 0)]
    held_ids = set(held_out["product_key"]) | {i for a in held_out["alt_ids"].fillna("") for i in a.split(";") if i}
    labels = labels[~labels["canonical_id"].isin(held_ids)]
    labels = labels[~labels["title"].map(normalise_title).isin(set(held_out["norm_title"]))]
    return pd.DataFrame({"text": labels["title"].str.strip(), "price_inr": labels[label].astype(float),
                         "source": "telegram"})


def build_examples(variant: str, splits) -> pd.DataFrame:
    train = splits["train"][["text", "price_inr"]].assign(source="phase_a")
    if variant == "base":
        return train
    label = "median_price" if variant == "telegram_median" else "p75_price"
    held_out = pd.concat([splits["val"], splits["test"]])
    return pd.concat([train, telegram_examples(label, held_out)], ignore_index=True)


def to_dataset(df: pd.DataFrame, seed: int = 42):
    from datasets import Dataset

    df = df.sample(frac=1, random_state=seed)
    return Dataset.from_dict({"prompt": [inr_prompt(t) for t in df["text"]],
                              "completion": [inr_completion(p) for p in df["price_inr"]]})


# ------------------------------------------------------------- model


def quant_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)


def load_base():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config(),
                                                 device_map="cuda:0", dtype="bfloat16")
    return model, tokenizer


def parse_price(answer: str):
    match = ANSWER_RE.search(answer or "")
    return float(match.group().replace(",", "")) if match else math.nan


def generate_prices(model, tokenizer, texts, batch_size: int = 32, desc: str = "scoring"):
    import torch
    from tqdm.auto import tqdm

    tokenizer.padding_side = "left"
    model.eval()
    prices = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        prompts = [inr_prompt(t) for t in texts[start : start + batch_size]]
        batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        answers = tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        prices.extend(parse_price(a.split("\n")[0]) for a in answers)
    return np.array(prices, dtype=float)


def score(prices, truth) -> dict:
    m = metrics(truth, prices)
    m["unparsed"] = int(np.isnan(prices).sum())
    return m


# ------------------------------------------------------------- progress


class Progress:
    """Writes models/specialist_progress.txt so a long run can be followed from an editor."""

    def __init__(self, total_runs: int):
        self.total_runs, self.done, self.started = total_runs, 0, time.perf_counter()

    def write(self, line: str) -> None:
        PROGRESS.parent.mkdir(exist_ok=True)
        elapsed = (time.perf_counter() - self.started) / 60
        PROGRESS.write_text(f"run {self.done + 1}/{self.total_runs}: {line}\nelapsed {elapsed:.1f} min\n",
                            encoding="utf-8")

    def callback(self, label: str):
        from transformers import TrainerCallback

        progress = self

        class Callback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % 10 and state.global_step != state.max_steps:
                    return
                spent = time.perf_counter() - self.start
                rate = spent / max(state.global_step, 1)
                left = (state.max_steps - state.global_step) * rate / 60
                loss = next((h["loss"] for h in reversed(state.log_history) if "loss" in h), float("nan"))
                progress.write(f"{label}, step {state.global_step}/{state.max_steps}, loss {loss:.3f}, "
                               f"about {left:.0f} min left in this run")

            def on_train_begin(self, args, state, control, **kwargs):
                self.start = time.perf_counter()

        return Callback()


# --------------------------------------------------------------- train


def train_variant(variant, examples, val, args, progress):
    import torch
    from peft import LoraConfig, PeftModel
    from trl import SFTConfig, SFTTrainer

    run_dir = RUNS / variant
    if run_dir.exists():
        shutil.rmtree(run_dir)
    model, tokenizer = load_base()
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    config = SFTConfig(
        output_dir=str(run_dir), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=1,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.03, weight_decay=0.0,
        bf16=True, gradient_checkpointing=True, optim="paged_adamw_8bit",
        max_length=256, completion_only_loss=True, logging_steps=10, report_to=[],
        save_strategy="steps" if args.max_steps > 0 else "epoch", save_steps=max(args.max_steps, 1),
        seed=42, data_seed=42,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=to_dataset(examples), processing_class=tokenizer,
                         peft_config=lora, callbacks=[progress.callback(variant)])
    started = time.perf_counter()
    trainer.train()
    minutes = (time.perf_counter() - started) / 60
    peak = round(torch.cuda.max_memory_allocated() / 2**20)
    del trainer, model
    torch.cuda.empty_cache()

    # Score every saved checkpoint (one per epoch) on validation by generating prices.
    base, tokenizer = load_base()
    results = []
    for checkpoint in sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
        progress.write(f"{variant}, scoring {checkpoint.name} on validation")
        model = PeftModel.from_pretrained(base, str(checkpoint))
        prices = generate_prices(model, tokenizer, val["text"].tolist(), args.eval_batch, desc=f"{variant} {checkpoint.name}")
        m = score(prices, val["price_inr"].to_numpy())
        results.append({"variant": variant, "checkpoint": str(checkpoint), **m})
        print(f"{variant} {checkpoint.name}: val MdAPE {100 * m['mdape']:.1f}%  within20 {100 * m['within_20']:.1f}%  "
              f"|log err| {m['mean_abs_log']:.4f}  unparsed {m['unparsed']}", flush=True)
        base = model.unload()
    del base
    torch.cuda.empty_cache()
    return results, {"train_minutes": round(minutes, 1), "peak_gpu_mb": peak, "examples": len(examples)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps (testing)")
    parser.add_argument("--val-limit", type=int, default=0, help="score on the first N validation items only")
    parser.add_argument("--smoke", action="store_true", help="base variant, 20 steps, 64 validation items, no test")
    args = parser.parse_args()
    if args.smoke:
        args.variants, args.max_steps, args.val_limit = ["base"], 20, 64

    import torch

    if not torch.cuda.is_available():
        print("CUDA is not available; run scripts/check_gpu.py first.")
        return 1
    seed_everything(42)
    splits = load_splits()
    val = splits["val"].head(args.val_limit) if args.val_limit else splits["val"]
    progress = Progress(len(args.variants))
    all_results, runs = [], {}
    for variant in args.variants:
        examples = build_examples(variant, splits)
        counts = examples["source"].value_counts().to_dict()
        print(f"== {variant}: {len(examples):,} examples {counts}", flush=True)
        results, info = train_variant(variant, examples, val, args, progress)
        runs[variant] = {**info, "sources": counts}
        all_results += results
        progress.done += 1

    best = min(all_results, key=lambda r: r["mean_abs_log"])
    print(f"Best on validation: {best['variant']} {Path(best['checkpoint']).name} (|log err| {best['mean_abs_log']:.4f})")
    report = {"base_model": BASE_MODEL, "runs": runs, "validation": all_results, "best": best,
              "settings": {k: v for k, v in vars(args).items()},
              "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if args.smoke:
        (ROOT / "models" / "specialist_smoke.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        PROGRESS.write_text("smoke test finished\n", encoding="utf-8")
        return 0

    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(best["checkpoint"], OUT, ignore=shutil.ignore_patterns("optimizer.pt", "scheduler.pt", "rng_state*"))
    from peft import PeftModel

    progress.write("scoring the chosen adapter on validation and test")
    base, tokenizer = load_base()
    model = PeftModel.from_pretrained(base, str(OUT))
    for split in ("val", "test"):
        df = splits[split]
        prices = generate_prices(model, tokenizer, df["text"].tolist(), args.eval_batch, desc=split)
        save_predictions("specialist", split, df, prices)
        report[f"{split}_full"] = score(prices, df["price_inr"].to_numpy())
        print(f"specialist {split}: {report[f'{split}_full']}", flush=True)
    (OUT / "specialist_meta.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    PROGRESS.write_text(f"finished: best {best['variant']} {Path(best['checkpoint']).name}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
