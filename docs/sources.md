# Deal sources: what was checked and why

Every source below was checked with real requests on 2 October 2026. "Robots" means the
site's robots.txt as read by our User-Agent (`BargainSpotter/0.2`); a 4xx robots.txt is
treated as "no rules" and a 5xx as "disallow everything" (RFC 9309). Re-check before
enabling anything, because sites change their rules.

## Summary

| Source | Status | Why |
| --- | --- | --- |
| Telegram channels via MTProto (your account) | Enabled by default | Primary source. Read-only, needs your one-time login. |
| Telegram public preview (`t.me/s/<channel>`) | Automatic fallback | No login; username channels only. Use `public_only: true` for hosted demos. |
| DealsMagnet RSS (`/feed`) | Implemented, **disabled** | robots.txt allows `/feed`, but the Terms of Service forbid automated access without written permission. |
| DesiDime listing (`/new`) | **Disabled**, not implemented | robots.txt allows `/new`, but the Terms of Use forbid automated download of data. |
| Buyhatke | **Disabled**, not implemented | No compliant way to read price history (it lives behind the disallowed `/api/`). |
| DealNews RSS (`legacy_us`) | Disabled | Kept for `PRICER_MODE=usd_legacy`. |
| Amazon Creators API | Stub only | Needs an approved Associates account with recent qualifying sales. |
| Flipkart Affiliate API | Stub only | Needs a registered affiliate id and API token. |
| amazon.in / flipkart.com product pages | Never requested | Their terms prohibit bots; redirect resolution stops before any store page. |

## Telegram

**Public web preview** `https://t.me/s/<username>`

- `https://t.me/robots.txt` returns 404, so no rules apply.
- All eight username channels in `sources.local.yaml` returned a preview with their latest
  20 posts: GrabOnIndiaOfficial, MahaLOOTbyKK, TrickXpert, OMGDeals, desidime, desidimeHot,
  pricebefore, bigtricksin. Older posts are reachable with `?before=<message id>`.
- `pricebefore`'s newest public post was from May 2025, so the channel looks inactive.
  `TrickXpert`'s newest post was from 24 September 2026.
- Channels given only by numeric id (DC LOOTS) or invite link (MTD Deals) have no public
  preview; they are MTProto-only.
- Two full preview pages are saved as test fixtures in `tests/fixtures/telegram/`.

Formats seen, all handled by `agents/sources/telegram_parser.py`:

| Example | Parsed as |
| --- | --- |
| `Now only ₹4,784 (MRP ₹14,995)` | price 4784, MRP 14995 |
| `Perfume Combo (Pack of 2) @199.` | price 199 |
| `Loot 199 https://fkrt.cc/...` / a line `3399` | price |
| `₹8,988 / 63% off` | price 8988, MRP derived as 24292 |
| `... - Rs. 9500` after `Gift Card of Rs10000` | price 9500, face value 10000 as MRP |
| `₹700 dropped! Current Price: ₹20,999 Highest Price: ₹25,299` | price 20999, highest 25299 |
| `Upto 64% Off On Fossil Watches` | not priceable (percentage only) |
| `Shein Clothing Starts @60` | not priceable (category listing) |
| `Edit - Over Now` | not priceable (expired) |

**MTProto (Telethon)**: not run live yet. It needs your interactive login
(`uv run python scripts/telegram_login.py`); after that, `scripts/list_channels.py` checks
that every configured channel resolves.

## Short links and redirect resolution

Shortened links are resolved one hop at a time by reading `Location` headers, never by
loading the destination, and resolution stops as soon as the next hop is a store URL.
robots.txt of each redirect host is respected:

| Host | robots.txt | Result |
| --- | --- | --- |
| `amzn.to`, `bit.ly` (Bitly) | allows all | resolved to `amazon.in/dp/<ASIN>` |
| `link.amazon` | 403 (no rules) | HEAD returns 404; GET redirects via `amzlinks.in` to `amazon.in/dp/<ASIN>` |
| `amzlinks.in` | blocks only some search bots | resolved |
| `fkrt.cc` | 403 (no rules) | resolved to `dl.flipkart.com/dl/.../p/itm...` |
| `grbn.in` (GrabOn) | 404 (no rules) | resolved to flipkart.com |
| `links.bigtricks.in` | allows all | redirects to `go.bigtricks.in/?o=<url>`, unwrapped without a request |
| `bitli.in` | 403 (no rules) | redirects to `linkredirect.in/...&dl=<url>`, unwrapped without a request |
| `trackingv3.linkredirect.in` | disallows all | never requested (destination is in the `dl=` parameter) |
| `ddime.in`, `visit.desidime.com`, `links.ddime.in` | disallow all | **not resolved**; DesiDime posts keep their short link and take the store from the post text |
| `cutt.ly` | allows all | resolved |

Affiliate and tracking parameters (`tag`, `linkCode`, `ascsubtag`, `affid`, `affExtParam*`,
`utm_*`, `otracker*`, ...) are stripped, so alerts push clean store links, never a channel
owner's affiliate link.

## DealsMagnet: `https://www.dealsmagnet.com/feed`

- Valid RSS 2.0; 578 items on 2 October 2026.
- Each item has a title, a link to the DealsMagnet deal page, tags such as `Myntra Deal`,
  and a description like "offer price of ₹559. MRP: ₹1999, Discount: ₹1440. Offer Store:
  Myntra. Last Updated on 30th September, 2026." There is no publish time, only that date.
- robots.txt allows `/feed` and disallows `/buy`, `/Buy`, `/RedirectTo`, `/RD/`, `/RDT/`.
  The store link sits behind `/buy`, so the deal page URL is kept instead.
- **Terms of Service** (`/ToS`, "Use Restrictions"): "You will not ... (b) access, monitor or
  copy any content or information on the Service using any robot, spider, scraper or other
  automated means or any manual process for any purpose without our express written
  permission; (c) violate the restrictions in any robot exclusion headers".
- Decision: `DealsMagnetSource` is implemented and tested against a **synthetic** feed with
  the same structure (`tests/fixtures/rss/dealsmagnet_like.xml`; no DealsMagnet content is
  stored in the repo), and is **disabled** in `sources.yaml`. Enable it only with their
  written permission.

## DesiDime: `https://www.desidime.com/new`

- robots.txt allows `/new` for general crawlers. It disallows `/goto/`, `/links/`,
  `/redirector*`, `/top-deals`, `/users*`, `/search_result/*` and others.
- **Terms of Use** (`/terms`): users agree not to "Use Automated Means, Including Spiders,
  Robots, Crawlers, Data Mining Tools, Or The Like To Download Data From The Service, Unless
  Expressly Permitted By DesiDime".
- Decision: **disabled and not implemented**; `/new` was never requested. The public
  `desidimeHot` Telegram channel, which DesiDime runs to broadcast its hot deals, covers the
  useful part. `AggregatorSource` (config-driven CSS selectors) is ready if DesiDime grants
  permission; it refuses to run unless the config sets `terms_checked: true`.

## Buyhatke: `https://buyhatke.com`

- robots.txt disallows `/api/`, `/redirect/`, `/webview/` and a few others; the sitemap is
  served from `/api/sitemap/`.
- Terms (`/terms`) are about the browser extension and do not mention automated access.
- The homepage is rendered in the browser and contains no product links in its HTML, and
  price history is loaded from `/api/`. There is no compliant way to find products or
  read their price history.
- Decision: **disabled and not implemented**.
- Note: while checking, one manual HEAD request was sent to `/api/sitemap/sitemap.xml`
  by mistake. Nothing in the code calls Buyhatke.

## Official store APIs

- **Amazon Creators API.** Product Advertising API 5.0 was retired on 15 May 2026 and
  replaced by the Creators API (OAuth client credentials from Associates Central).
  Secondary sources report that access needs at least 10 qualifying referred sales in the
  previous 30 days. Not built; `agents/sources/store_api.py` has the interface only.
- **Flipkart Affiliate API.** Product Feed and Delta Feed APIs; the official docs require
  a registered affiliate account, a tracking id and a private API token. Not built; stub only.

Sources: [Creators API migration (DEV)](https://dev.to/th3nate/amazon-pa-api-v5-is-shutting-down-april-30-2026-here-is-what-changes-at-the-auth-layer-22ek),
[Creators API eligibility (FreshStore)](https://blog.freshstore.com/amazon-creators-api-credentials-eligibility/),
[Flipkart Affiliate API overview](https://affiliate.flipkart.com/api-docs/af_overview.html),
[Flipkart Affiliate API FAQ](https://affiliate.flipkart.com/api-docs/af_faq.html).

## Adding a source

1. Check robots.txt and the terms of use, and add a section here with what you found.
2. RSS or Atom feed: add an entry under `rss:` in `sources.yaml`.
3. HTML listing page: add an entry under `aggregators:` with CSS selectors and
   `terms_checked: true`.
4. Anything else: subclass `DealSource` and return `RawDeal` objects; register it in
   `agents/sources/__init__.py`.
