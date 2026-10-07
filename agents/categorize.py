"""
Cheap keyword heuristics for category and brand. They run on every scraped post
(before any LLM call), so they must be fast and offline. The LLM can refine the
category later for the few deals it selects.
"""

import re
from typing import List, Optional, Tuple

CATEGORY_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("Gift Cards", ["gift card", "giftcard", "voucher", "amazon pay balance"]),
    ("Mobiles", ["smartphone", "mobile", "iphone", "galaxy s", "galaxy m", "galaxy a", "galaxy f",
                 "redmi", "realme", "oneplus nord", "poco", "vivo", "oppo", "iqoo", "nothing phone", "5g phone"]),
    ("Laptops & Computers", ["laptop", "notebook", "macbook", "chromebook", "ideapad", "vivobook", "monitor",
                             "keyboard", "mouse", "ssd", "pendrive", "pen drive", "hard disk", "router", "printer",
                             "tablet", "ipad", "memory card", "microsd"]),
    ("Audio", ["earbuds", "earphone", "headphone", "neckband", "airdopes", "speaker", "soundbar", "tws",
               "airpods", "buds", "home theatre", "home theater"]),
    ("Wearables", ["smartwatch", "smart watch", "fitness band", "smart ring"]),
    ("TV & Video", [" tv", "television", "smart tv", "projector", "fire tv", "streaming stick"]),
    ("Large Appliances", [" ac ", "air conditioner", "split ac", "inverter ac", "refrigerator", "fridge",
                          "washing machine", "geyser", "water heater", "air purifier", "water purifier",
                          "chimney", "dishwasher", "inverter battery", "air cooler"]),
    ("Kitchen", ["air fryer", "mixer", "grinder", "induction", "cooktop", "kettle", "pressure cooker",
                 "cookware", "kadai", "tawa", "toaster", "microwave", "otg", "juicer", "blender", "lunch box",
                 "bottle", "flask", "dinner set"]),
    ("Beauty & Personal Care", ["trimmer", "shaver", "hair dryer", "straightener", "perfume", "deodorant",
                                "shampoo", "conditioner", "face wash", "serum", "sunscreen", "lotion", "soap",
                                "lipstick", "makeup", "moisturi", "toothbrush", "toothpaste", "body wash"]),
    ("Footwear", ["shoes", "sneakers", "sandals", "slippers", "flip flops", "running shoe", "loafers", "crocs"]),
    ("Fashion", ["t-shirt", "tshirt", "shirt", "jeans", "trousers", "kurta", "saree", "dress", "jacket",
                 "hoodie", "skirt", "watch", "sunglasses", "wallet", "backpack", "handbag", "trolley bag",
                 "luggage", "innerwear", "socks"]),
    ("Grocery", ["atta", "rice", "oil", "ghee", "dal", "coffee", "tea ", "biscuits", "chocolate", "dry fruits",
                 "almonds", "cashew", "detergent", "dishwash", "surf excel", "harpic", "ltr", "5 ltr", "kg pack"]),
    ("Home & Furniture", ["mattress", "pillow", "bedsheet", "curtain", "sofa", "chair", "table", "lamp",
                          "bulb", "led light", "fan", "mcb", "extension board", "power strip", "storage box",
                          "organiser", "organizer", "mop", "vacuum cleaner"]),
    ("Toys & Baby", ["toy", "lego", "diaper", "baby", "kids", "pampers", "stroller"]),
    ("Sports & Fitness", ["cricket", "football", "badminton", "yoga mat", "dumbbell", "cycle", "treadmill",
                          "protein", "whey", "gym"]),
    ("Automotive", ["helmet", "car charger", "tyre inflator", "dash cam", "bike", "car "]),
    ("Books & Media", ["book", "kindle", "novel", "paperback"]),
    ("Subscriptions", ["subscription", "membership", "prime", "/month", "per month", "ott"]),
]

CATEGORIES = [name for name, _ in CATEGORY_KEYWORDS] + ["Other"]

BRANDS = [
    "Apple", "Samsung", "OnePlus", "Xiaomi", "Redmi", "Mi", "Poco", "realme", "Vivo", "Oppo", "iQOO",
    "Motorola", "Nokia", "Nothing", "Google", "Lava", "Tecno", "Infinix", "HP", "Dell", "Lenovo", "Asus",
    "Acer", "MSI", "Microsoft", "boAt", "Noise", "boult", "Boult", "JBL", "Sony", "Bose", "Sennheiser",
    "Skullcandy", "Marshall", "Zebronics", "pTron", "Mivi", "Fire-Boltt", "Fastrack", "Titan", "Fossil",
    "Casio", "Amazfit", "LG", "Haier", "Whirlpool", "Godrej", "Voltas", "Daikin", "Blue Star", "Lloyd",
    "Panasonic", "Hitachi", "Carrier", "Cruise", "IFB", "Bosch", "Philips", "Havells", "Bajaj", "Prestige",
    "Pigeon", "Butterfly", "Morphy Richards", "Kent", "Eureka Forbes", "AO Smith", "Crompton", "Usha",
    "Orient", "Syska", "Wipro", "Inalsa", "Agaro", "Milton", "Cello", "Borosil", "Hawkins", "Wonderchef",
    "Nike", "Adidas", "Puma", "Reebok", "Skechers", "Asics", "Campus", "Bata", "Woodland", "Red Tape",
    "Levi's", "Allen Solly", "Van Heusen", "Peter England", "US Polo", "Jack & Jones", "H&M", "Wildcraft",
    "American Tourister", "Safari", "VIP", "Skybags", "Mamaearth", "Nivea", "Dove", "Lakme", "Maybelline",
    "L'Oreal", "Himalaya", "Bella Vita", "Wild Stone", "Park Avenue", "Beardo", "Parachute", "Tata",
    "Fortune", "Aashirvaad", "Saffola", "Nestle", "Cadbury", "Haldiram", "Amul", "Surf Excel", "Ariel",
    "Tide", "Duracell", "SanDisk", "Seagate", "WD", "Western Digital", "Kingston", "Crucial", "TP-Link",
    "Logitech", "Portronics", "Ambrane", "Anker", "Canon", "Nikon", "GoPro", "DJI", "Amazon", "Echo",
    "Kindle", "Lego", "Hot Wheels", "Decathlon", "Yonex", "Cosco", "Nivia", "Kapiva", "MuscleBlaze",
    "Optimum Nutrition", "Dyson", "Eufy", "Mijia", "Ecovacs", "Atomberg", "Shein", "Oscar",
]

_BRAND_PATTERNS = [
    (brand, re.compile(r"(?<![\w])" + re.escape(brand) + r"(?![\w])", re.IGNORECASE))
    for brand in sorted(BRANDS, key=len, reverse=True)
]


def guess_category(text: str) -> str:
    lowered = f" {text.lower()} "
    for name, keywords in CATEGORY_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return name
    return "Other"


def guess_brand(text: str) -> Optional[str]:
    for brand, pattern in _BRAND_PATTERNS:
        if pattern.search(text or ""):
            return brand
    return None
