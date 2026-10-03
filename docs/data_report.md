# Data report: Indian product datasets for INR pricing

What is actually in the four Kaggle datasets in `data/raw/`, where they differ from their
descriptions, and how `scripts/prepare_data.py` cleans them. Numbers are from the files as
downloaded (October 2026). Credits and licences are in the README.

## Summary

| Folder | Files | Rows | Scraped | Price columns | Used for |
|---|---|---|---|---|---|
| `amazon_2025` | 1 CSV | 9,318 | 2025-10-06 (from the `qid` in the URLs) | `Price` (float, no MRP) | train / val |
| `flipkart_2025` | 3 CSVs (earphones, laptops, mobiles) | 2,460 | 2025-07-21 | `Price`, `Original Price` (text, `₹1,099`) | train / val |
| `flipkart_khanna` | 1 CSV | 12,041 | unknown (about 4 years old) | `selling_price`, `mrp` (text, `₹1,399`) | train / val, fashion, footwear, home only |
| `amazon_2026` | 1 CSV | 623 | 2026-06-05 | `price_inr`, `original_price_inr` (float) | test only |

## Per dataset

### amazon_2025 (prothomeshmistry)

- Columns: `Product_Name, Price, Rating, Review_Count, ASIN, Product_URL, Availability`. Matches the description.
- 66 missing prices, 36 missing ratings, no zero prices, no duplicate rows or ASINs. `Availability` is "In Stock" for all rows.
- Prices: min Rs 41, median Rs 616, max Rs 1,39,990. Heavily skewed to cheap accessories: about 2,300 phone accessories (cases, cables, chargers) and 2,200 computer accessories, versus about 400 phones and 140 laptops after cleaning.
- **Scrape date is inside the Amazon Great Indian Festival sale.** The `qid` search timestamp in every URL is 2025-10-06. Prices are likely festive sale prices, and there is no MRP column to tell.
- **Almost no large appliances**: 3 refrigerators, about 12 ACs, 5 washing machines. The search was "electronics and accessories", so it also includes stationery, art supplies, batteries and extended warranties.
- 159 duplicate titles: colour variants of the same product listed as separate ASINs, often at the same price (Hush Puppies sandals appear 7 times).

### flipkart_2025 (priyankamalavade)

- **Three CSVs with 2,460 rows**, not 960: `flipkart_earphones.csv` (960), `flipkart_laptops.csv` (540), `flipkart_mobile_data.csv` (960). Earphones has `Title, Product URL, Image URL, Rating, Rating Count, Price, Original Price, Discount, Offer, Timestamp`; laptops and mobiles have `Product Link`, `Key Features` (a Python list stored as text), `Exchange Offer` and `Bank Offer` instead.
- Prices are text with the rupee sign and Indian grouping (`₹1,05,590`), parsed with `agents.money.parse_amount`. 84 earphone rows have no price.
- **Titles are cut at about 60 characters with "..."** (890 of 960 earphones, 405 of 540 laptops), so title rules cannot place many rows. Each file is a single search, so the file name gives the category.
- **URLs are doubled**: `https://www.flipkart.comhttps://www.flipkart.com/...`. After fixing, the `pid` query parameter is a stable product id; only 580 of 960 earphone rows are distinct products (pages overlap), 337 of 540 laptops and 889 of 960 phones.
- All three files were scraped within 15 minutes on 2025-07-21.

### flipkart_khanna (priyankkhanna)

- **About 12,000 rows** (12,041), with `category_1/2/3, title, product_rating, selling_price, mrp, seller_name, seller_rating, description, highlights, image_links`. No URL, no product id, no date.
- Categories: Baby and Kids (2,338), Men's wear (2,360), Women's wear (2,422), Home and Furniture (2,120), Sports, Books and More (1,820), Electronics (981). Each leaf category has a round number of rows (40, 80, 120 ...), so this is a capped crawl per category, not a sample of the catalogue.
- 37 duplicate rows, 28 missing prices, 375 missing MRPs. Descriptions are missing for 7,020 rows and highlights for 5,481.
- MRPs are often far above the selling price (a T-shirt at Rs 228 with an MRP of Rs 1,399), typical of Indian fashion listings.
- The fashion category includes gold coins, gold bars and fine jewellery (Rs 15,000 to 85,000). They are removed by the outlier step.
- Kept: Men's and Women's wear (except grooming, beauty and personal care appliances), kids' clothing and footwear, and Home and Furniture (except smart home). Electronics are about four years out of date, and toys, books and sports are outside the taxonomy, so 5,853 rows are dropped.

### amazon_2026 (kulkarniparth09), test set

- 623 rows, 22 columns. Already cleaned by the author: `price_inr` equals `product_price`, `original_price_inr` equals `product_original_price`; `currency` is INR or empty.
- 12 search categories with 48 rows each, except gaming console (96) and earbuds (47). **About 23% are large appliances** (refrigerator, AC, washing machine: 144 rows) and 15% gaming consoles. There is no fashion, footwear or home.
- Titles contain HTML entities (`&quot;`, `&amp;`, `&#x27;`), unescaped during cleaning.
- 6 rows have no price; 44 have no MRP. No price is above its MRP. The median MRP is 1.75 times the price.
- The "camera" search mixes Leica and Canon cinema cameras (up to Rs 11,17,486) with CCTV cameras and kids' toy cameras (Rs 200); the "earbuds" search includes Rs 209 silicone case covers.
- **143 ASINs overlap amazon_2025.** These and 7 more matches by normalised title are removed from training (see the leakage check below).

## Differences from the prompt's table

| Prompt said | Files show |
|---|---|
| flipkart_2025: 960 rows, maybe several CSVs | 3 CSVs, 2,460 rows, different columns per file |
| flipkart_khanna: real Flipkart scrape | no URL, id or date; capped at round numbers per category |
| amazon_2026: 12 categories | 12 search categories, but they are search terms: accessories, CCTV and toy cameras appear inside them |
| amazon_2025: no category column | true; it also contains stationery, art supplies and warranty plans |

## Cleaning funnel

From `data/processed/funnel.json`:

| step | amazon_2025 | flipkart_2025 | flipkart_khanna | amazon_2026 | total |
|---|---|---|---|---|---|
| raw rows | 9,318 | 2,460 | 12,041 | 623 | 24,442 |
| price present | 9,252 | 2,376 | 12,013 | 617 | 24,258 |
| price above zero, single price (no ranges) | 9,252 | 2,376 | 12,013 | 617 | 24,258 |
| not refurbished or renewed | 9,248 | 2,376 | 12,013 | 617 | 24,254 |
| not a combo or multipack | 8,449 | 2,367 | 10,105 | 612 | 21,533 |
| not a gift card or warranty | 8,317 | 2,367 | 10,105 | 612 | 21,401 |
| flipkart_khanna: fashion, footwear, home only | 8,317 | 2,367 | 6,188 | 612 | 17,484 |
| one row per product (colour variants merged) | 7,279 | 1,085 | 4,367 | 601 | 13,332 |
| log price within 3x IQR of its category | 7,264 | 1,085 | 4,331 | 599 | 13,279 |

No dataset had zero prices or price ranges; the checks stay in the script for future data.

Notes on the steps:

- **Multipacks** use `agents.price_signal.is_multipack`, the same rule live valuation uses. Inspecting what it removed exposed a bug: its `2 x 750ml` pattern also matched screen resolutions (`2560x1440`, `3840 x 2160`) and sizes (`6x4 inches`), so monitors and TVs were treated as multipacks, here and in the live app (capped at low confidence). Fixed, with tests.
- **Colour variants**: rows are merged when they share a product id, or the same brand, normalised title (colour words and colour-only bracket parts removed, `agents/titles.py`) and RAM/storage/size. One row per group at the median price. 2,058 groups had more than one listing.
- **Outliers** (3x IQR of log price within a category) removed 53 rows: 36 gold coins, bars and fine jewellery in khanna's fashion, 15 misfiled amazon_2025 rows (AC remotes, fridge magnets and a smart plug filed as large appliances, docking stations and laptop RAM filed as laptops), and **2 test rows**: the Canon EOS R5 C kit (Rs 10,05,481) and the Canon EOS R1 (Rs 5,99,999), far above every other camera.

## Categories

A fixed taxonomy of 17 categories (`agents/taxonomy.py`), separate from the app's own list.
khanna rows use their own category tree and flipkart_2025 rows their file name; the rest
use keyword rules. Rules report whether they are sure: a title that names a device first
and an accessory later ("Spigen Optik Armor for Samsung Galaxy S24 Ultra Case") is unsure,
and unsure rows go to gpt-5-mini, 25 titles per call, answers cached in
`data/cache/taxonomy_llm.json`.

| source of the label | rows |
|---|---|
| dataset category (khanna tree, flipkart_2025 file) | 5,416 |
| keyword rules | 6,109 |
| gpt-5-mini | 1,754 |

LLM cost: about $0.09 in total over four runs while the rules were tuned, $0.06 of it for
the final cache. The first run returned a JSON list, and when the model skipped a title the
answers shifted by one (an Acer laptop came back as "fashion"); answers are now keyed by
title number.

Spot check of the final labels against amazon_2026's own search categories (cleaned test
rows): all ACs (47), fridges (48), washing machines (48), TVs (45) and phones (46) land in
the matching taxonomy category, as do 43 of 46 laptops, 43 of 48 tablets, 43 of 44
smartwatches and 91 of 93 gaming consoles (as "other"). The weakest are earbuds (41 of 47)
and headphones (38 of 46), and most of those disagreements are right: case covers and
stands filed under the device search.

## Splits and leakage

`scripts/make_splits.py`:

- **Test**: all 599 cleaned amazon_2026 rows.
- **Leakage**: 147 amazon_2025 products removed from train and validation, 140 because an
  ASIN (including the ASINs of merged colour variants) is in the test set and 7 more because
  the normalised title matches a test product.
- **Validation**: 10% of the rest, stratified by category: 1,254 rows.
- **Train**: 11,279 rows.

| category | train | val | test |
|---|---|---|---|
| fashion | 2,156 | 240 | 0 |
| mobile_accessory | 2,126 | 236 | 13 |
| computer_accessory | 2,002 | 222 | 9 |
| home_kitchen | 1,313 | 146 | 0 |
| earbuds_headphones | 745 | 83 | 79 |
| smartphone | 705 | 78 | 52 |
| footwear | 492 | 55 | 0 |
| other | 459 | 51 | 91 |
| laptop | 404 | 45 | 43 |
| audio_other | 264 | 29 | 0 |
| camera | 172 | 19 | 38 |
| smartwatch | 154 | 17 | 43 |
| tv | 135 | 15 | 45 |
| tablet | 59 | 7 | 43 |
| monitor | 52 | 6 | 0 |
| small_appliance | 34 | 4 | 0 |
| large_appliance | 7 | 1 | 143 |

**The test set does not look like the training data.** Large appliances are 24% of the
test set and 0.06% of training. Fashion, footwear and home are 35% of training and absent
from test. Median price is Rs 11,700 in test against Rs 639 (amazon_2025) and Rs 488
(khanna) in training. Validation comes from the training distribution, so validation scores
are optimistic about the test set, and both are optimistic about live Telegram deals (see
the training report).
