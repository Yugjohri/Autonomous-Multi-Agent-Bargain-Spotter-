"""
Train the INR neural network pricer and pick its size and inputs on the validation split.

    uv run python scripts/train_nn_inr.py [--quick] [--max-epochs 300] [--patience 20]

Reuses agents.deep_neural_network.DeepNeuralNetwork (residual MLP) with smaller sizes than
the USD model: about 11k training rows would overfit 4096 x 10 badly. The grid tries
hidden sizes 256, 512 and 1024 with 4 or 6 layers and learning rates 1e-3 and 3e-4, on two inputs:
    hashing             5000 binary word features + taxonomy one-hot
    hashing+embeddings  the same plus the 384-dim all-MiniLM-L6-v2 embedding
Target: log1p(price_inr), standardised with the train mean and std. Loss: Huber. AdamW,
learning rate halved on plateaus, early stopping on validation mean |log error|, fixed seeds.

The winner (lowest validation error) is saved to models/nn_inr.pth with its settings and
target statistics in models/nn_inr_meta.json. Every run is logged to models/nn_inr_search.json.
Requires a CUDA GPU (scripts/check_gpu.py) unless --allow-cpu is given.
"""

import argparse
import itertools
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from inr_common import ROOT, embeddings, load_splits, save_predictions, seed_everything

from agents.deep_neural_network import DeepNeuralNetwork
from agents.evaluator_inr import metrics
from agents.inr_features import build_features

MODELS = ROOT / "models"
PROGRESS = MODELS / "nn_inr_progress.txt"


class Progress:
    """
    Two progress bars (all runs, and epochs of the current run) plus a small text file,
    models/nn_inr_progress.txt, rewritten every epoch so progress can be followed from an
    editor when the script runs in the background.
    """

    def __init__(self, total_runs: int):
        self.total_runs = total_runs
        self.done = 0
        self.started = time.perf_counter()
        self.runs_bar = tqdm(total=total_runs, desc="configs", unit="run", position=0)
        self.epoch_bar = None
        self.label = ""
        MODELS.mkdir(exist_ok=True)

    def start_run(self, label: str, max_epochs: int) -> None:
        self.label = label
        self.epoch_bar = tqdm(total=max_epochs, desc=label, unit="epoch", position=1, leave=False)

    def epoch(self, epoch: int, val_error: float, best: float, stale: int, patience: int) -> None:
        self.epoch_bar.update(1)
        self.epoch_bar.set_postfix(val=f"{val_error:.3f}", best=f"{best:.3f}", stop_in=patience - stale)
        elapsed = time.perf_counter() - self.started
        eta = "after the first config"
        if self.done:
            eta = f"about {elapsed / self.done * (self.total_runs - self.done) / 60:.0f} min"
        PROGRESS.write_text(
            f"config {self.done + 1}/{self.total_runs}: {self.label}\n"
            f"epoch {epoch}, val |log error| {val_error:.4f}, best {best:.4f}, "
            f"early stop in {patience - stale} epochs without improvement\n"
            f"elapsed {elapsed / 60:.1f} min, remaining {eta}\n",
            encoding="utf-8",
        )

    def end_run(self) -> None:
        self.epoch_bar.close()
        self.done += 1
        self.runs_bar.update(1)

    def close(self, summary: str) -> None:
        self.runs_bar.close()
        PROGRESS.write_text(f"finished {self.total_runs} configs in "
                            f"{(time.perf_counter() - self.started) / 60:.1f} min\n{summary}\n", encoding="utf-8")
FEATURE_SETS = {
    "hashing": {"hashing": True, "embeddings": False, "category": True},
    "hashing+embeddings": {"hashing": True, "embeddings": True, "category": True},
}


def features(splits, spec):
    return {
        name: torch.from_numpy(build_features(df["text"].tolist(), df["category"].tolist(), spec,
                                              embeddings(name, df) if spec["embeddings"] else None))
        for name, df in splits.items()
    }


def train_one(x, y_train, val_prices, config, device, max_epochs, patience, progress=None, seed=42):
    seed_everything(seed)
    model = DeepNeuralNetwork(x["train"].shape[1], num_layers=config["num_layers"],
                              hidden_size=config["hidden_size"], dropout_prob=config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    loss_fn = nn.HuberLoss(delta=1.0)
    x_train = x["train"].to(device)
    y = y_train.to(device)
    x_val = x["val"].to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best = {"val_mean_abs_log": float("inf")}
    best_state = None
    stale = 0
    epoch_times = []
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    for epoch in range(1, max_epochs + 1):
        start = time.perf_counter()
        model.train()
        order = torch.randperm(len(x_train), generator=generator).to(device)
        for i in range(0, len(order), config["batch_size"]):
            batch = order[i : i + config["batch_size"]]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[batch]).squeeze(1), y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_std = model(x_val).squeeze(1).float().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - start)
        val_pred = np.expm1(val_std * config["y_std"] + config["y_mean"])
        m = metrics(val_prices, val_pred)
        scheduler.step(m["mean_abs_log"])
        if m["mean_abs_log"] < best["val_mean_abs_log"] - 1e-4:
            best = {"val_mean_abs_log": m["mean_abs_log"], "val_mdape": m["mdape"], "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if progress:
            progress.epoch(epoch, m["mean_abs_log"], best["val_mean_abs_log"], stale, patience)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    best.update(
        epochs_run=epoch,
        seconds_per_epoch=float(np.mean(epoch_times)),
        peak_gpu_mb=round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
        parameters=sum(p.numel() for p in model.parameters()),
    )
    return model, best


def predict(model, x, config, device):
    model.eval()
    with torch.no_grad():
        out = model(x.to(device)).squeeze(1).float().cpu().numpy()
    return np.maximum(np.expm1(out * config["y_std"] + config["y_mean"]), 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="one small config per feature set (smoke test)")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        print("CUDA is not available; run scripts/check_gpu.py (or pass --allow-cpu).")
        return 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    seed_everything(42)
    splits = load_splits()
    y_log = np.log1p(splits["train"]["price_inr"].to_numpy())
    y_mean, y_std = float(y_log.mean()), float(y_log.std())
    y_train = torch.from_numpy(((y_log - y_mean) / y_std).astype(np.float32))
    val_prices = splits["val"]["price_inr"].to_numpy()

    sizes = [(256, 4)] if args.quick else [(256, 4), (512, 4), (512, 6), (1024, 4), (1024, 6)]
    rates = [1e-3] if args.quick else [1e-3, 3e-4]
    runs = []
    best = None
    progress = Progress(len(FEATURE_SETS) * len(sizes) * len(rates))
    for feature_name, spec in FEATURE_SETS.items():
        x = features(splits, spec)
        for (hidden, layers), lr in itertools.product(sizes, rates):
            config = {"features": spec, "feature_set": feature_name, "hidden_size": hidden, "num_layers": layers,
                      "dropout": 0.3, "lr": lr, "weight_decay": 1e-2, "batch_size": 128,
                      "y_mean": y_mean, "y_std": y_std}
            progress.start_run(f"{feature_name} {hidden}x{layers} lr {lr:g}", args.max_epochs)
            model, result = train_one(x, y_train, val_prices, config, device, args.max_epochs, args.patience, progress)
            progress.end_run()
            run = {**{k: v for k, v in config.items() if k != "features"}, **result}
            runs.append(run)
            tqdm.write(f"{feature_name:20s} {hidden:5d}x{layers} lr {lr:g}: val |log err| {result['val_mean_abs_log']:.4f}  "
                  f"MdAPE {100 * result['val_mdape']:.1f}%  best epoch {result['epoch']}/{result['epochs_run']}  "
                  f"{result['seconds_per_epoch']:.2f}s/epoch  peak GPU {result['peak_gpu_mb']} MB")
            if best is None or result["val_mean_abs_log"] < best[1]["val_mean_abs_log"]:
                best = (model, result, config, x)

    model, result, config, x = best
    progress.close(f"best: {config['feature_set']} {config['hidden_size']}x{config['num_layers']}, "
                   f"val |log error| {result['val_mean_abs_log']:.4f}")
    MODELS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS / "nn_inr.pth")
    meta = {
        "features": config["features"], "feature_set": config["feature_set"],
        "hidden_size": config["hidden_size"], "num_layers": config["num_layers"], "dropout": config["dropout"],
        "y_mean": y_mean, "y_std": y_std, "target": "log1p(price_inr), standardised on train",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_rows": len(splits["train"]), "validation": result, "torch": torch.__version__,
    }
    (MODELS / "nn_inr_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (MODELS / "nn_inr_search.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    for split in ("val", "test"):
        save_predictions("nn", split, splits[split], predict(model, x[split], config, device))
    print(f"Best: {config['feature_set']} {config['hidden_size']}x{config['num_layers']} "
          f"(val |log err| {result['val_mean_abs_log']:.4f}); saved models/nn_inr.pth and models/nn_inr_meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
