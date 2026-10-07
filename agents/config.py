"""
Configuration loading.

sources.yaml    committed, example values only
sources.local.yaml   gitignored, your real channels; merged on top of sources.yaml
settings.yaml   committed, pricing mode, ensemble weights, thresholds and scan limits

Dictionaries merge key by key; lists in the local file replace lists in the base
file, so a local telegram.channels list fully replaces the placeholder list.
Settings can be overridden with environment variables named after the key path,
for example PRICER_MODE, MIN_DISCOUNT_INR, MIN_DISCOUNT_PCT or FRESHNESS_HOURS.
"""

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent

SOURCES_FILE = ROOT / "sources.yaml"
SOURCES_LOCAL_FILE = ROOT / "sources.local.yaml"
SETTINGS_FILE = ROOT / "settings.yaml"

PRICER_MODES = ("inr", "usd_legacy")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_sources_config(base: Path = SOURCES_FILE, local: Path = SOURCES_LOCAL_FILE) -> Dict[str, Any]:
    config = deep_merge(read_yaml(base), read_yaml(local))
    config["_has_local"] = local.exists()
    return config


# Environment variable -> (path in settings.yaml, type)
ENV_OVERRIDES = {
    "PRICER_MODE": (("pricer_mode",), str),
    "MIN_DISCOUNT_INR": (("thresholds", "min_discount_inr"), float),
    "MIN_DISCOUNT_PCT": (("thresholds", "min_discount_pct"), float),
    "MIN_DISCOUNT_USD": (("thresholds", "min_discount_usd"), float),
    "FRESHNESS_HOURS": (("scan", "freshness_hours"), float),
    "TOP_N_PER_RUN": (("planning", "top_n"), int),
    "SCAN_TIME_BUDGET_SECONDS": (("scan", "time_budget_seconds"), float),
}


def load_settings(path: Path = SETTINGS_FILE, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    settings = read_yaml(path)
    env = os.environ if env is None else env
    for name, (keys, cast) in ENV_OVERRIDES.items():
        raw = env.get(name)
        if raw in (None, ""):
            continue
        target = settings
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = cast(raw)
    mode = settings.get("pricer_mode", "inr")
    if mode not in PRICER_MODES:
        raise ValueError(f"PRICER_MODE must be one of {PRICER_MODES}, got {mode!r}")
    return settings


def get(settings: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = settings
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
