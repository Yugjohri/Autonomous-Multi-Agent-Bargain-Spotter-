from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Optional

from agents.agent import Agent
from agents.cache import DiskCache
from agents.config import get, load_settings, load_sources_config
from agents.deals import Deal, DealSelection, InrPick, InrSelection, LegacyDealSelection, ScrapedDeal
from agents.extraction import Candidate, SeenStore, build_candidates, rank_candidates
from agents.money import format_money
from agents.normalize import RedirectResolver, SeenIndex, normalize_link
from agents.observations import ObservationLog, observation_row
from agents.sources import build_http, build_sources
from agents.sources.base import DealSource, RawDeal


class ScannerAgent(Agent):
    """
    Finds candidate deals and picks the best few.

    INR mode: every configured source (Telegram, RSS, aggregators) -> regex extraction and
    link normalization -> pre-filter (priced, recognised store, fresh, not seen) -> at most
    max_llm_candidates go to gpt-5-mini, which picks the best ones and resolves the payable
    price, conditions, category and brand.

    usd_legacy mode: the original DealNews RSS flow.
    """

    MODEL = "gpt-5-mini"

    INR_SYSTEM_PROMPT = """You pick the best Indian online shopping deals from candidates collected from Telegram deal channels and Indian deal sites.
All prices are in Indian rupees (INR). Never convert currencies.

For each pick, give the payable price: what any buyer actually pays for one unit.
- "₹X off", "save ₹X", "₹X dropped" and "₹X cashback" are savings, not prices.
- A coupon anyone can apply on the product page counts: use the price after it and mention it in coupon_note.
- Bank card offers, exchange bonuses and cashback are conditional: keep them out of price and describe them in coupon_note, including the effective price if the post states it.
- The price hint was extracted by regex and is usually right; correct it only when the text clearly says otherwise.

Skip candidates that are gift cards or vouchers, subscriptions, credit card or app promotions, multi-product sale pages, or where the product or the price is unclear.
Prefer clear single products with a strong real discount, posted by several channels or trusted channels, and recent posts.
Write product_description about the product itself (specs, model, variant), not about the deal."""

    LEGACY_SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
    Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
    Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
    Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price.
    """

    LEGACY_USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
    You should rephrase the description to be a summary of the product itself, not the terms of the deal.
    Remember to respond with a short paragraph of text in the product_description field for each of the 5 items that you select.
    Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price.

    Deals:

    """

    LEGACY_USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(
        self,
        settings: Optional[dict] = None,
        sources: Optional[List[DealSource]] = None,
        client=None,
        resolver: Optional[RedirectResolver] = None,
        seen_store: Optional[SeenStore] = None,
        offline: bool = False,
        sources_config: Optional[dict] = None,
        observations: Optional[ObservationLog] = None,
    ):
        """
        Set up this instance. Sources, the OpenAI client and the resolver can be injected
        (tests, fixture mode); otherwise they are built from sources.yaml and settings.yaml.
        """
        self.log("Scanner Agent is initializing")
        self.settings = settings if settings is not None else load_settings()
        self.mode = self.settings.get("pricer_mode", "inr")
        self.offline = offline
        self.MODEL = get(self.settings, "models.scanner", self.MODEL)
        config = sources_config if sources_config is not None else load_sources_config()
        http = build_http(config, offline=offline)
        self.sources = sources if sources is not None else build_sources(config, http)
        self.resolver = resolver or RedirectResolver(
            http,
            DiskCache("redirects", ttl_seconds=30 * 24 * 3600),
            extra_hosts=get(config, "resolve_hosts", []) or [],
            offline=offline,
        )
        self.seen_store = seen_store or SeenStore()
        self.observations = observations
        self.openai = client
        if self.openai is None and not offline:
            from openai import OpenAI

            self.openai = OpenAI()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._running: dict = {}
        self.last_raw: List[RawDeal] = []
        self.last_candidates: List[Candidate] = []
        self.last_report = None
        self.log(f"Scanner Agent is ready ({self.mode} mode, {len(self.sources)} sources)")

    # ------------------------------------------------------------------ INR

    def fetch_raw(self) -> List[RawDeal]:
        """
        All posts from every source, fetched in parallel within scan.time_budget_seconds.
        Each source isolates its own failures; a source that overruns the budget is skipped
        for this scan (and not started again while its previous fetch is still running).
        """
        limit = 60
        budget = float(get(self.settings, "scan.time_budget_seconds", 120))
        if self._executor is None:
            workers = int(get(self.settings, "scan.source_workers", 6))
            self._executor = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="source")
        futures = {}
        for source in self.sources:
            previous = self._running.get(source.name)
            if previous is not None and not previous.done():
                self.log(f"Scanner Agent skips {source.name}: its previous fetch is still running")
                continue
            future = self._executor.submit(source.fetch, limit)
            self._running[source.name] = future
            futures[future] = source
        done, late = wait(futures, timeout=budget)
        raws: List[RawDeal] = []
        for future in done:
            raws.extend(future.result())
        for future in late:
            self.log(f"Scanner Agent skipped {futures[future].name}: no answer within the {budget:.0f}s budget")
        return raws

    def log_observations(self, raws: List[RawDeal]) -> None:
        """Append every parsed post (before any filtering) to the observation log."""
        if self.observations is None or not raws:
            return
        rows = []
        for raw in raws:
            if raw.currency != "INR":
                continue
            link = None
            if raw.url:
                # Only priceable posts are worth a network resolution; the rest are normalized offline.
                link = normalize_link(raw.url, self.resolver if raw.priceable else None)
            rows.append(observation_row(raw, link))
        written = self.observations.append(rows)
        if written:
            self.log(f"Scanner Agent logged {written} new observations")

    def candidates(self, memory, extra: Optional[List[RawDeal]] = None) -> List[Candidate]:
        raws = self.fetch_raw() + list(extra or [])
        self.last_raw = raws
        try:
            self.log_observations(raws)
        except OSError as exc:
            self.log(f"Could not write observations: {exc}")
        seen = self.seen_store.index(SeenIndex.from_opportunities(memory))
        candidates, report = build_candidates(
            raws,
            self.resolver,
            seen,
            freshness_hours=float(get(self.settings, "scan.freshness_hours", 6)),
        )
        self.last_report = report
        self.last_candidates = candidates
        self.resolver.cache.flush()
        self.log(f"Scanner Agent pre-filter: {report.summary()}")
        limit = int(get(self.settings, "scan.max_llm_candidates", 30))
        return rank_candidates(candidates, limit)

    @staticmethod
    def describe(i: int, c: Candidate) -> str:
        parts = [f"[{i}] {c.title or '(untitled)'}", f"price hint {format_money(c.price)}"]
        if c.mrp:
            parts.append(f"MRP {format_money(c.mrp)}")
        if c.raw.highest_price_hint:
            parts.append(f"highest past price {format_money(c.raw.highest_price_hint)}")
        parts.append(f"store {c.store}")
        age = c.age_minutes()
        if age is not None:
            parts.append(f"posted {age} min ago")
        parts.append(f"posted by {len(c.seen_in)} source(s)")
        return " | ".join(parts) + f"\n    text: {c.clean_text()}"

    def make_inr_prompt(self, candidates: List[Candidate], picks: int) -> str:
        lines = "\n".join(self.describe(i, c) for i, c in enumerate(candidates))
        return (
            f"Pick up to {picks} of the best deals from these {len(candidates)} candidates. "
            f"Use each candidate's [id] as candidate_id.\n\nCandidates:\n\n{lines}"
        )

    def _log_usage(self, usage) -> None:
        if usage is None:
            return
        prices = get(self.settings, f"models.prices_per_million.{self.MODEL}", None)
        cost = ""
        if prices:
            usd = (usage.prompt_tokens * prices[0] + usage.completion_tokens * prices[1]) / 1_000_000
            cost = f", about ${usd:.4f}"
        self.log(f"Scanner Agent LLM usage: {usage.prompt_tokens} in, {usage.completion_tokens} out tokens{cost}")

    def llm_pick(self, candidates: List[Candidate], picks: int) -> List[InrPick]:
        self.log(f"Scanner Agent is asking {self.MODEL} to pick from {len(candidates)} candidates")
        result = self.openai.chat.completions.parse(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.INR_SYSTEM_PROMPT},
                {"role": "user", "content": self.make_inr_prompt(candidates, picks)},
            ],
            response_format=InrSelection,
            reasoning_effort="minimal",
        )
        self._log_usage(getattr(result, "usage", None))
        parsed = result.choices[0].message.parsed
        return parsed.picks if parsed else []

    def offline_pick(self, candidates: List[Candidate], picks: int) -> List[InrPick]:
        """Deterministic stand-in for the LLM (fixture mode): candidates are already ranked."""
        return [
            InrPick(
                candidate_id=i,
                title=c.title or c.clean_text(80),
                product_description=c.clean_text(300),
                price=c.price,
                coupon_note=c.raw.coupon_hint,
                category=c.category,
                brand=c.brand,
            )
            for i, c in enumerate(candidates[:picks])
        ]

    @staticmethod
    def to_deal(pick: InrPick, c: Candidate) -> Deal:
        price = pick.price
        if price <= 0 or not (c.price / 3 <= price <= c.price * 3):
            # The LLM disagrees wildly with the price in the post; trust the post.
            price = c.price
        return Deal(
            product_description=pick.product_description,
            price=price,
            url=c.link.url,
            currency="INR",
            mrp=c.mrp,
            store=c.store,
            source=c.seen_in[0],
            posted_at=c.posted_at,
            canonical_id=c.canonical_id,
            coupon_note=pick.coupon_note,
            title=pick.title or c.title,
            category=pick.category,
            brand=pick.brand or c.brand,
            seen_in=list(c.seen_in),
            highest_price=c.raw.highest_price_hint,
        )

    def scan_inr(self, memory, extra: Optional[List[RawDeal]] = None) -> Optional[DealSelection]:
        candidates = self.candidates(memory, extra)
        if not candidates:
            self.log("Scanner Agent found no new candidates")
            return None
        picks_wanted = int(get(self.settings, "scan.picks_per_scan", 5))
        if self.offline or self.openai is None:
            picks = self.offline_pick(candidates, picks_wanted)
        else:
            picks = self.llm_pick(candidates, picks_wanted)
        self.seen_store.mark(candidates)
        deals: List[Deal] = []
        used = set()
        for pick in picks:
            if not 0 <= pick.candidate_id < len(candidates) or pick.candidate_id in used:
                continue
            used.add(pick.candidate_id)
            deals.append(self.to_deal(pick, candidates[pick.candidate_id]))
        self.log(f"Scanner Agent selected {len(deals)} deals")
        return DealSelection(deals=deals[:picks_wanted]) if deals else None

    # --------------------------------------------------------------- legacy

    def fetch_deals(self, memory) -> List[ScrapedDeal]:
        """
        Look up deals published on RSS feeds
        Return any new deals that are not already in the memory provided
        """
        self.log("Scanner Agent is about to fetch deals from RSS feed")
        urls = [opp.deal.url for opp in memory]
        scraped = ScrapedDeal.fetch()
        result = [scrape for scrape in scraped if scrape.url not in urls]
        self.log(f"Scanner Agent received {len(result)} deals not already scraped")
        return result

    def make_user_prompt(self, scraped) -> str:
        """
        Create a user prompt for OpenAI based on the scraped deals provided
        """
        user_prompt = self.LEGACY_USER_PROMPT_PREFIX
        user_prompt += "\n\n".join([scrape.describe() for scrape in scraped])
        user_prompt += self.LEGACY_USER_PROMPT_SUFFIX
        return user_prompt

    def scan_legacy(self, memory) -> Optional[DealSelection]:
        scraped = self.fetch_deals(memory)
        if scraped:
            user_prompt = self.make_user_prompt(scraped)
            self.log("Scanner Agent is calling OpenAI using Structured Outputs")
            result = self.openai.chat.completions.parse(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.LEGACY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=LegacyDealSelection,
                reasoning_effort="minimal",
            )
            picks = result.choices[0].message.parsed
            result = DealSelection(deals=[pick.to_deal() for pick in picks.deals if pick.price > 0])
            self.log(
                f"Scanner Agent received {len(result.deals)} selected deals with price>0 from OpenAI"
            )
            return result
        return None

    # ---------------------------------------------------------------- entry

    def scan(self, memory: List = [], extra: Optional[List[RawDeal]] = None) -> Optional[DealSelection]:
        """
        Find new deals and return the best few, or None if there aren't any
        :param memory: Opportunities surfaced before, used to skip known deals
        :param extra: posts that arrived live (Telegram) since the last scan
        """
        if self.mode == "usd_legacy":
            return self.scan_legacy(memory)
        return self.scan_inr(memory, extra)

    def test_scan(self, memory: List = []) -> Optional[DealSelection]:
        """
        Return a test DealSelection, to be used during testing
        """
        results = {
            "deals": [
                {
                    "product_description": "The Hisense R6 Series 55R6030N is a 55-inch 4K UHD Roku Smart TV that offers stunning picture quality with 3840x2160 resolution. It features Dolby Vision HDR and HDR10 compatibility, ensuring a vibrant and dynamic viewing experience. The TV runs on Roku's operating system, allowing easy access to streaming services and voice control compatibility with Google Assistant and Alexa. With three HDMI ports available, connecting multiple devices is simple and efficient.",
                    "price": 178,
                    "url": "https://www.dealnews.com/products/Hisense/Hisense-R6-Series-55-R6030-N-55-4-K-UHD-Roku-Smart-TV/484824.html?iref=rss-c142",
                },
                {
                    "product_description": "The Poly Studio P21 is a 21.5-inch LED personal meeting display designed specifically for remote work and video conferencing. With a native resolution of 1080p, it provides crystal-clear video quality, featuring a privacy shutter and stereo speakers. This display includes a 1080p webcam with manual pan, tilt, and zoom control, along with an ambient light sensor to adjust the vanity lighting as needed. It also supports 5W wireless charging for mobile devices, making it an all-in-one solution for home offices.",
                    "price": 30,
                    "url": "https://www.dealnews.com/products/Poly-Studio-P21-21-5-1080-p-LED-Personal-Meeting-Display/378335.html?iref=rss-c39",
                },
                {
                    "product_description": "The Lenovo IdeaPad Slim 5 laptop is powered by a 7th generation AMD Ryzen 5 8645HS 6-core CPU, offering efficient performance for multitasking and demanding applications. It features a 16-inch touch display with a resolution of 1920x1080, ensuring bright and vivid visuals. Accompanied by 16GB of RAM and a 512GB SSD, the laptop provides ample speed and storage for all your files. This model is designed to handle everyday tasks with ease while delivering an enjoyable user experience.",
                    "price": 446,
                    "url": "https://www.dealnews.com/products/Lenovo/Lenovo-Idea-Pad-Slim-5-7-th-Gen-Ryzen-5-16-Touch-Laptop/485068.html?iref=rss-c39",
                },
                {
                    "product_description": "The Dell G15 gaming laptop is equipped with a 6th-generation AMD Ryzen 5 7640HS 6-Core CPU, providing powerful performance for gaming and content creation. It features a 15.6-inch 1080p display with a 120Hz refresh rate, allowing for smooth and responsive gameplay. With 16GB of RAM and a substantial 1TB NVMe M.2 SSD, this laptop ensures speedy performance and plenty of storage for games and applications. Additionally, it includes the Nvidia GeForce RTX 3050 GPU for enhanced graphics and gaming experiences.",
                    "price": 650,
                    "url": "https://www.dealnews.com/products/Dell/Dell-G15-Ryzen-5-15-6-Gaming-Laptop-w-Nvidia-RTX-3050/485067.html?iref=rss-c39",
                },
            ]
        }
        for deal in results["deals"]:
            deal["currency"] = "USD"
        return DealSelection(**results)
