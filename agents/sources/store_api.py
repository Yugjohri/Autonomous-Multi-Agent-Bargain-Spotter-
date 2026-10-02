"""
Interface stubs for official store APIs. Both are off and not implemented, because
each needs an approved affiliate account (checked October 2026, see docs/sources.md):

* Amazon Creators API (replaced Product Advertising API 5.0, retired 15 May 2026):
  OAuth client credentials from Associates Central, which reportedly requires
  10 qualifying referred sales in the last 30 days.
* Flipkart Affiliate API: Product Feed / Delta Feed APIs, which need a registered
  affiliate account, a tracking id and a private API token.

A future implementation should subclass StoreApiSource and return RawDeal objects with
store, price and MRP filled from the API, so the rest of the pipeline is unchanged.
"""

from typing import List

from agents.sources.base import DealSource, RawDeal


class StoreApiNotConfigured(Exception):
    pass


class StoreApiSource(DealSource):
    store: str = "other"
    reason = "store API sources are not implemented; they need an approved affiliate account"

    def _fetch(self, limit: int) -> List[RawDeal]:
        raise StoreApiNotConfigured(self.reason)

    def lookup(self, canonical_id: str) -> RawDeal:
        """Current price and MRP for one product, e.g. to verify a claimed deal price."""
        raise StoreApiNotConfigured(self.reason)


class AmazonCreatorsApiSource(StoreApiSource):
    store = "amazon_in"
    reason = "Amazon Creators API needs approved Associates credentials (not implemented)"

    def __init__(self, **_):
        super().__init__("amazon_creators_api")


class FlipkartAffiliateApiSource(StoreApiSource):
    store = "flipkart"
    reason = "Flipkart Affiliate API needs a registered affiliate id and token (not implemented)"

    def __init__(self, **_):
        super().__init__("flipkart_affiliate_api")
