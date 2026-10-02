import os
import sys
import logging
import json
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
import chromadb
from agents.config import get, load_settings, load_sources_config
from agents.deals import Opportunity, load_opportunities
from agents import categorize
import numpy as np

load_dotenv(override=True)

# Colors for logging
BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

# Colors for plot (USD "products" collection categories)
CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]
COLORS = ["red", "blue", "brown", "orange", "yellow", "green", "purple", "cyan"]

# Colors for the INR categories; anything unknown falls into the grey "Other" bucket.
INR_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
    "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
]
OTHER_COLOR = "#9e9e9e"


def color_for(category: Optional[str], mode: str = "inr") -> str:
    """Plot color for a category. Unknown categories map to the Other bucket instead of raising."""
    names, palette = (CATEGORIES, COLORS) if mode == "usd_legacy" else (categorize.CATEGORIES[:-1], INR_COLORS)
    if category in names:
        return palette[names.index(category) % len(palette)]
    return OTHER_COLOR


_logging_ready = False


def init_logging():
    """Send INFO logs to stdout once per process (repeated calls used to duplicate every line)."""
    global _logging_ready
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if _logging_ready:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    for noisy in ("httpx", "httpcore", "urllib3", "telethon", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _logging_ready = True


class DealAgentFramework:
    DB = "products_vectorstore"
    MEMORY_FILENAME = "memory.json"
    FIXTURE_DIR = Path("data/fixture_run")

    def __init__(self, settings: Optional[dict] = None, fixture: bool = False, dry_run: bool = False):
        """
        :param fixture: replay saved posts offline (no network, no notifications, separate memory)
        :param dry_run: run the real pipeline but send no notifications
        """
        init_logging()
        self.settings = settings if settings is not None else load_settings()
        self.mode = self.settings.get("pricer_mode", "inr")
        self.fixture = fixture
        self.dry_run = dry_run or fixture
        if fixture:
            self._prepare_fixture_run()
        client = chromadb.PersistentClient(path=self.DB)
        self.memory = self.read_memory()
        self.collection = client.get_or_create_collection("products") if self.mode == "usd_legacy" else None
        self.planner = None
        self.live_queue: "queue.Queue" = queue.Queue()
        self._run_lock = threading.Lock()
        self._live_thread: Optional[threading.Thread] = None

    def _prepare_fixture_run(self) -> None:
        from agents.inr_store import offline_hf

        offline_hf()
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        self.FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        # A fixture run starts clean every time and never touches the real memory.json.
        self.MEMORY_FILENAME = str(self.FIXTURE_DIR / "memory.json")
        for name in ("memory.json", "seen.json"):
            path = self.FIXTURE_DIR / name
            if path.exists():
                path.unlink()

    def _fixture_planner(self):
        from agents.cache import MemoryCache
        from agents.extraction import SeenStore
        from agents.normalize import RedirectResolver
        from agents.planning_agent import PlanningAgent
        from agents.scanner_agent import ScannerAgent
        from agents.sources.fixture import FixtureSource, load_fixture_redirects

        cache = MemoryCache()
        for short, target in load_fixture_redirects().items():
            cache.set(short, target)
        scanner = ScannerAgent(
            settings=self.settings,
            sources=[FixtureSource()],
            resolver=RedirectResolver(None, cache, offline=True),
            seen_store=SeenStore(self.FIXTURE_DIR / "seen.json"),
            offline=True,
            sources_config={},
        )
        return PlanningAgent(settings=self.settings, scanner=scanner, offline=True, dry_run=True, remember=False)

    def init_agents_as_needed(self):
        if not self.planner:
            self.log("Initializing Agent Framework")
            if self.fixture:
                self.planner = self._fixture_planner()
            else:
                from agents.planning_agent import PlanningAgent

                self.planner = PlanningAgent(self.collection, settings=self.settings, dry_run=self.dry_run)
            self.log("Agent Framework is ready")

    def read_memory(self) -> List[Opportunity]:
        return self.load_memory_file(self.MEMORY_FILENAME)

    @classmethod
    def load_memory_file(cls, path: Optional[str] = None) -> List[Opportunity]:
        """Saved opportunities, with records from before currencies existed migrated to USD."""
        path = path or cls.MEMORY_FILENAME
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return load_opportunities(data)
        return []

    def write_memory(self) -> None:
        data = [opportunity.model_dump(mode="json") for opportunity in self.memory]
        with open(self.MEMORY_FILENAME, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

    @classmethod
    def reset_memory(cls) -> None:
        data = []
        if os.path.exists(cls.MEMORY_FILENAME):
            with open(cls.MEMORY_FILENAME, "r", encoding="utf-8") as file:
                data = json.load(file)
        truncated = data[:2]
        with open(cls.MEMORY_FILENAME, "w", encoding="utf-8") as file:
            json.dump(truncated, file, indent=2, ensure_ascii=False)

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def drain_live(self) -> list:
        items = []
        while True:
            try:
                items.append(self.live_queue.get_nowait())
            except queue.Empty:
                return items

    def run(self, extra: Optional[list] = None) -> List[Opportunity]:
        """One pipeline run. Serialized: the 5 minute timer and live Telegram posts share it."""
        with self._run_lock:
            self.init_agents_as_needed()
            extra = list(extra or []) + self.drain_live()
            logging.info("Kicking off Planning Agent")
            result = self.planner.plan(memory=self.memory, extra=extra) if extra else self.planner.plan(memory=self.memory)
            if isinstance(result, Opportunity):
                result = [result]
            result = result or []
            logging.info(f"Planning Agent has completed and returned {len(result)} deal(s)")
            if result:
                self.memory.extend(result)
                self.write_memory()
            return self.memory

    # ------------------------------------------------------------------- live

    def start_live(self, debounce_seconds: float = 5.0) -> bool:
        """
        Listen to Telegram (MTProto) and run the pipeline within seconds of a new post.
        Returns False if live mode is off or unavailable; the timed scan still runs.
        """
        if self.fixture or self.mode != "inr" or self._live_thread is not None:
            return False
        if not get(load_sources_config(), "telegram.live", True):
            return False
        self.init_agents_as_needed()
        telegram = next(
            (s for s in self.planner.scanner.sources if hasattr(s, "start_live")), None
        )
        if telegram is None or not telegram.start_live(self.live_queue.put):
            return False

        def worker():
            while True:
                first = self.live_queue.get()
                time.sleep(debounce_seconds)  # gather posts that arrive together
                batch = [first] + self.drain_live()
                self.log(f"{len(batch)} live Telegram post(s) arrived; running the pipeline")
                try:
                    self.run(extra=batch)
                except Exception as exc:  # noqa: BLE001
                    self.log(f"Live run failed: {exc}")

        self._live_thread = threading.Thread(target=worker, name="live-deals", daemon=True)
        self._live_thread.start()
        return True

    # ------------------------------------------------------------------- plot

    @classmethod
    def get_plot_data(cls, max_datapoints=2000, mode: str = "inr"):
        from sklearn.manifold import TSNE

        client = chromadb.PersistentClient(path=cls.DB)
        name = "products" if mode == "usd_legacy" else "products_inr"
        collection = client.get_or_create_collection(name)
        result = collection.get(
            include=["embeddings", "documents", "metadatas"], limit=max_datapoints
        )
        documents = result["documents"]
        if len(documents) < 5:
            return documents, np.zeros((len(documents), 3)), [OTHER_COLOR] * len(documents)
        vectors = np.array(result["embeddings"])
        categories = [(metadata or {}).get("category") for metadata in result["metadatas"]]
        colors = [color_for(c, mode) for c in categories]
        perplexity = min(30, len(documents) - 1)
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1, perplexity=perplexity)
        reduced_vectors = tsne.fit_transform(vectors)
        return documents, reduced_vectors, colors


if __name__ == "__main__":
    DealAgentFramework().run()
