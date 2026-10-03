"""
The fixed product taxonomy used to train and evaluate the INR pricing models.

This is separate from the app's own category list (agents/categorize.py), which drives
UI colours, cooldowns and the scanner's LLM schema. The training taxonomy is finer for
electronics, where price levels differ a lot (a phone case and a phone are both "Mobiles"
in the app). from_app_category() maps the app's categories onto it.

classify() is cheap and offline and says whether it is sure. Titles that name both a
device and an accessory ("Spigen Optik Armor for Galaxy S24 Ultra Case") are the hard
part: an accessory noun before the first device word means an accessory, an accessory
noun after it is reported as unsure. scripts/prepare_data.py sends unsure rows to an LLM;
at inference the rule answer is used as is.
"""

import html
import re
from typing import Optional, Tuple

TAXONOMY = [
    "smartphone", "laptop", "tablet", "tv", "earbuds_headphones", "smartwatch", "camera", "monitor",
    "large_appliance", "small_appliance", "computer_accessory", "mobile_accessory", "audio_other",
    "fashion", "footwear", "home_kitchen", "other",
]


def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)


# Phrases that only ever name an accessory.
_ACCESSORY_PHRASES = _rx(
    r"\b(back cover|flip cover|case cover|bumper case|phone case|mobile cover|screen (guard|protector)|"
    r"tempered glass|power ?banks?|charging cable|otg (adapter|connector|cable)|lens protector|"
    r"camera lens protector|replacement (strap|band|ear ?tips|cushions?|ear ?pads?)|silicone case)\b"
)
# A device word directly followed by an accessory noun ("Earphone Case Cover", "Tablet Stand").
_DEVICE_ACCESSORY = _rx(
    r"\b(phones?|mobile|iphone|earphones?|earbuds?|airpods|airdopes|buds|headphones?|watch|smartwatch|tablet|"
    r"ipad|camera|tv|laptop|macbook|keyboard)\s+(?:\w+\s+){0,2}(cases?|covers?|straps?|stand|holder|mount|bags?|"
    r"backpacks?|sleeves?|skins?|protectors?|pouch|ear ?tips|cushions?|ear ?pads?|chargers?|adapters?|riser|"
    r"cooling pad|tote|briefcase|table|desk|lock|remote)\b"
)
# Accessory nouns used to break ties against a device word elsewhere in the title. "Charging
# case" (earbuds) and "aluminium case" (watches) are part of the device.
_ACCESSORY_NOUN = _rx(
    r"\b(?<!charging )(?<!aluminium )(?<!aluminum )(?<!titanium )(?<!steel )(cases?|covers?)\b",
    r"\b(straps?|chargers?|cables?|adapters?|holder|mount|sleeves?|skins?|pouch|stand|connector|hub|remote|"
    r"refill|tripod|dock|ear ?tips|ear ?pads?|cushions?|splitter|converter|cleaner|cleaning kit|mouse ?pads?|"
    r"pens?|pencil|stylus|keyboards?|mouse|mice|webcam|batter(y|ies)|thermal paste|stickers?|decals?)\b",
)
_COMPUTER_WORDS = _rx(r"\b(laptops?|macbook|notebook|pc|computer|desktop|ipad|tablet|tab|keyboard|monitor|printer)\b")

_OUT_OF_SCOPE = _rx(
    r"\b(fire ?tv stick|streaming (stick|device|media player)|chromecast|gaming consoles?|game consoles?|"
    r"video games?|playstation|xbox|nintendo|retro handheld|handheld game|handheld gaming)\b"
)

_CPU = r"\b(intel|amd|ryzen|core ?i[3579]|core ultra|celeron|pentium|athlon|snapdragon x|apple m\d|m[1-5] chip|mediatek)\b"
# A model number, unless it is a size or capacity ("Redmi 80 cm TV", "realme 10000mAh").
_MODEL_NO = r"[a-z]?\d+(?![\d\s]*(cms?|inch(es)?|mah|w|l|kg|ton|litres?)\b)\w*"

DEVICE_RULES = [
    ("tv", _rx(r"\b(smart tv|led tv|qled|oled tv|google tv|android tv|televisions?|webos|tizen|mini led tv|"
               r"smart led|projector)\b", r"\b\d{2,3} ?cms? \(\d{2,3} ?inch(es)?\)[^|]{0,60}\btv\b")),
    ("large_appliance", _rx(r"\b(refrigerators?|fridge|air conditioners?|split ac|window ac|inverter ac|"
                            r"\d(\.\d)? ton\b|washing machines?|washer dryer|dishwashers?|chimney|geysers?|"
                            r"water heater|air cooler|deep freezer|side by side refrigerator)\b(?! safe)")),
    # Many accessories say "for laptop", so the bare word (or a series name shared with other
    # products, such as Predator) counts only next to a CPU name.
    ("laptop", _rx(r"\b(macbook|chromebook|ultrabook|vivobook|zenbook|ideapad|thinkpad|thinkbook|inspiron|vostro|"
                   r"jiobook|imac)\b",
                   rf"\b(laptops?|notebook|latitude|pavilion|victus|aspire|nitro|predator|omen|legion|loq|"
                   rf"tuf gaming|rog strix|desktop pc|all[- ]in[- ]one pc|mini pc)\b[^\n]*{_CPU}",
                   rf"{_CPU}[^\n]*\b(laptops?|notebook|desktop pc|all[- ]in[- ]one pc)\b")),
    ("computer_accessory", _rx(r"\b(writing (tablet|pad)|drawing (tablet|pad)|graphics? tablet|pen tablet|"
                               r"lcd writing)\b")),
    ("tablet", _rx(r"\b(tablets?|ipad|galaxy tab|matepad|redmi pad|xiaomi pad|oneplus pad|lenovo tab|honor pad|"
                   r"realme pad|idea ?tab|tab [a-z]?\d+)\b")),
    ("smartwatch", _rx(r"\b(smart ?watch(es)?|fitness (band|tracker)|smart band|apple watch|galaxy watch\d*|"
                       r"smart ring|calling watch|watch \d+ (lite|pro|active|ultra)|watch ultra|watch fit)\b")),
    ("earbuds_headphones", _rx(r"\b(earbuds?|earphones?|headphones?|headsets?|neckbands?|tws|airpods|airdopes|"
                               r"buds|in[- ]ear|over[- ]ear|on[- ]ear|ear ?pods?)\b")),
    # Buds, watches, pads and TVs of phone brands are matched by the rules above.
    ("smartphone", _rx(r"\b(smartphones?|mobile phones?|feature phone|keypad phone|galaxy z (fold|flip)|narzo|iqoo|"
                       r"oneplus nord|nothing phone|cmf phone|dual sim)\b",
                       r"\b(iphone|galaxy|redmi( note)?|realme|poco|vivo|oppo( reno)?|oneplus|motorola|moto|tecno|"
                       r"infinix|pixel|lava|itel|hmd|nokia)\s+(pop |spark |camon |pova |hot |note |"
                       r"smart |gt |zero |edge |reno ?)?" + _MODEL_NO,
                       r"\b5g \([^)]*\d+ ?gb", r"\b\d+ ?gb ?\+ ?\d+ ?gb\b", r"\([^)]*\d+ ?gb (ram)?\s*[,+/|]\s*\d+ ?gb")),
    ("camera", _rx(r"\b(cameras?|dslr|mirrorless|gopro|action cam|cctv|dash ?cam|instax|camcorder|"
                   r"camera lens)\b")),
    # Consoles, streaming sticks and games are out of scope; accessories that mention them are not.
    ("other", _OUT_OF_SCOPE),
    ("monitor", _rx(r"\b(gaming monitor|led monitor|ips monitor|\d+(\.\d+)? ?(inch|inches|\")[^|]{0,40}\bmonitor|"
                    r"monitor[^|]{0,30}\d+ ?hz)\b")),
    ("audio_other", _rx(r"\b(speakers?|soundbars?|sound bar|home theat(re|er)|subwoofer|party speaker)\b")),
]

OTHER_RULES = [
    ("computer_accessory", _rx(r"\b(keyboards?|mouse|mice|webcam|pen ?drives?|flash drive|ssd|solid state drive|"
                               r"hard (disk|drive)|hdd|memory card|micro ?sd|sd card|routers?|wi-?fi "
                               r"(adapter|extender|range|repeater)|mesh wifi|printers?|ink|toner|cartridge|usb hub|"
                               r"docking|graphics card|gamepad|game controller|controllers?|joystick|mouse ?pad|"
                               r"ups|cpu cooler|ddr\d|nas|usb (drive|stick)|card reader|laptop)\b")),
    ("mobile_accessory", _rx(r"\b(cases?|covers?|chargers?|charging (adapter|stand|dock|pad)|cables?|adapters?|"
                             r"car mount|phone holder|mobile holder|mobile stand|phone stand|selfie stick|tripod|"
                             r"ring light|popsocket|watch strap|straps?|smart ?tag|airtag|magsafe|stylus)\b")),
    ("audio_other", _rx(r"\b(speakers?|soundbars?|sound bar|home theat(re|er)|subwoofer|microphones?|mic|"
                        r"amplifier|turntable|radio|music system|karaoke)\b")),
    ("small_appliance", _rx(r"\b(trimmer|shaver|epilator|hair dryer|straightener|curler|steam iron|dry iron|"
                            r"vacuum cleaner|robot vacuum|air purifier|water purifier|electric kettle|kettle|"
                            r"mixer grinder|mixer|juicer|hand blender|toaster|air fryer|induction|microwave|otg oven|"
                            r"oven|sandwich maker|coffee maker|ceiling fan|table fan|pedestal fan|tower fan|"
                            r"room heater|massager|weighing scale|hair clipper|electric toothbrush|chopper|"
                            r"rice cooker|garment steamer)\b")),
]

# Loose words: fine for a dataset's own fashion and home rows, unsure anywhere else.
SOFT_RULES = [
    ("footwear", _rx(r"\b(shoes?|sneakers?|sandals?|slippers?|flip[- ]?flops?|floaters?|loafers?|boots?|heels|"
                     r"crocs|clogs|mojaris?|juttis?|bellies|footwear|wedges|slides)\b")),
    ("fashion", _rx(r"\b(shirts?|t-?shirts?|tees?|jeans|trousers|shorts|kurtas?|kurtis?|sarees?|lehenga|"
                    r"dress(es)?|crop top|tank top|jackets?|hoodies?|sweatshirts?|sweaters?|blazers?|skirts?|"
                    r"leggings|innerwear|briefs?|boxers?|bras?|lingerie|nightwear|night suit|pyjamas?|socks|"
                    r"caps?|belts?|wallets?|handbags?|sling bags?|backpacks?|watch(es)?|sunglasses|jewell?ery|"
                    r"earrings?|necklace|bracelet|pendant|bangles?|dupatta|track ?pants|joggers|raincoat|"
                    r"windcheater|scarf|stole|gloves|swimwear|unstitched|dhoti|sherwani|trolley bag|luggage)\b")),
    ("home_kitchen", _rx(r"\b(cookware|kadai|kadhai|tawa|frying pan|pressure cooker|bottles?|flask|lunch box|"
                         r"dinner set|plates?|bowls?|mugs?|containers?|storage box|organi[sz]ers?|bed ?sheets?|"
                         r"pillows?|cushions?|curtains?|blankets?|comforter|mattress|towels?|sofa|chairs?|"
                         r"wardrobe|shelf|shelves|rack|lamps?|bulbs?|decor|wall clock|vase|showpiece|painting|"
                         r"doormat|rugs?|carpet|cutlery|knife|knives|spoons?|tray|jars?|baskets?|dustbin|mop|"
                         r"furniture|candles?|diya|idol|photo frame|hangers?|kitchen|dining|casserole|tiffin|"
                         r"bedding|quilt|dohar)\b")),
]

# Category names in the raw datasets, used when the rules find nothing.
HINTS = {
    "smartphone": "smartphone", "mobile": "smartphone", "mobiles": "smartphone", "laptop": "laptop",
    "laptops": "laptop", "tablet": "tablet", "television": "tv", "tv": "tv", "refrigerator": "large_appliance",
    "air conditioner": "large_appliance", "washing machine": "large_appliance", "smartwatch": "smartwatch",
    "camera": "camera", "headphones": "earbuds_headphones", "earbuds": "earbuds_headphones",
    "earphones": "earbuds_headphones", "fashion": "fashion", "footwear": "footwear",
    "home_kitchen": "home_kitchen",
}

# The app's categories (agents/categorize.py) mapped onto the training taxonomy.
APP_TO_TAXONOMY = {
    "Mobiles": "smartphone", "Laptops & Computers": "computer_accessory", "Audio": "audio_other",
    "Wearables": "smartwatch", "TV & Video": "tv", "Large Appliances": "large_appliance",
    "Kitchen": "home_kitchen", "Footwear": "footwear", "Fashion": "fashion",
    "Home & Furniture": "home_kitchen",
}


ACCESSORY_LEAD_WORDS = 8


def _accessory(text: str) -> str:
    return "computer_accessory" if _COMPUTER_WORDS.search(text) else "mobile_accessory"


def classify(text: str, hint: Optional[str] = None) -> Tuple[str, bool]:
    """Return (category, sure). The dataset's own label (hint) is used when the rules find nothing."""
    text = html.unescape(text or "")
    if _ACCESSORY_PHRASES.search(text) or _DEVICE_ACCESSORY.search(text):
        return _accessory(text), True
    # The device named first is what is sold: "Headphones ... compatible with Laptop" are headphones.
    # Ties at the same position go to the earlier rule.
    matches = [(m.start(), i, name, m) for i, (name, pattern) in enumerate(DEVICE_RULES) if (m := pattern.search(text))]
    if matches:
        _, _, name, found = min(matches, key=lambda x: (x[0], x[1]))
        accessory = _ACCESSORY_NOUN.search(text)
        if accessory is None:
            return name, True
        # Accessory listings lead with what they are; "...Bottom Mount Refrigerator" is a fridge.
        early = len(text[: accessory.start()].split()) < ACCESSORY_LEAD_WORDS
        if accessory.start() < found.start() and early:
            return _accessory(text), True
        return name, False

    for name, pattern in OTHER_RULES:
        if pattern.search(text):
            return name, True
    hinted = HINTS.get((hint or "").strip().lower())
    for name, pattern in SOFT_RULES:
        if pattern.search(text):
            return name, hinted == name
    if hinted:
        return hinted, True
    return "other", False


def rule_category(text: str, hint: Optional[str] = None) -> str:
    return classify(text, hint)[0]


def from_app_category(category: Optional[str]) -> str:
    return APP_TO_TAXONOMY.get(category or "", "other")


def taxonomy_for(text: str, app_category: Optional[str] = None) -> str:
    """Category used as a model feature at inference: rules first, then the app's category."""
    found = rule_category(text)
    return found if found != "other" else from_app_category(app_category)
