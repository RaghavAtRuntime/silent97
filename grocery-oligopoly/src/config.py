"""Shared paths, banner metadata and classification rules.

Everything that encodes a judgement call (who owns which banner, what counts as
private label, how products map to categories, which statistical windows we
use) lives here so it can be reviewed in one place.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
DB_PATH = PROCESSED / "grocery.duckdb"

for _p in (RAW, PROCESSED, TABLES, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
SOURCES = {
    "hammer": {
        "url": "https://jacobfilipp.com/hammerdata/hammer-5-csv.zip",
        "files": {"hammer-4-product.csv": "hammer-product.csv", "hammer-4-raw.csv": "hammer-raw.csv"},
        "citation": "Filipp, J. (2024-2026). Project Hammer: Canadian grocery price data. "
        "https://jacobfilipp.com/hammer/ (see also https://projecthammer.org/)",
    },
    "statcan_prices": {
        "url": "https://www150.statcan.gc.ca/n1/tbl/csv/18100245-eng.zip",
        "files": {"18100245.csv": "18100245.csv"},
        "citation": "Statistics Canada. Table 18-10-0245-01 Monthly average retail prices for "
        "selected products. DOI: https://doi.org/10.25318/1810024501-eng",
    },
    "statcan_cpi": {
        "url": "https://www150.statcan.gc.ca/n1/tbl/csv/18100004-eng.zip",
        "files": {"18100004.csv": "18100004.csv"},
        "citation": "Statistics Canada. Table 18-10-0004-01 Consumer Price Index, monthly, not "
        "seasonally adjusted. DOI: https://doi.org/10.25318/1810000401-eng",
    },
}

# --------------------------------------------------------------------------
# Banner metadata. Ownership verified against corporate sources (see README).
# "role" is the banner's consumer value proposition in the portfolio.
# --------------------------------------------------------------------------
VENDORS: dict[str, dict[str, str]] = {
    "Loblaws": {"label": "Loblaws", "parent": "Loblaw", "role": "Conventional / premium-leaning"},
    "NoFrills": {"label": "No Frills", "parent": "Loblaw", "role": "Hard discount"},
    "TandT": {"label": "T&T", "parent": "Loblaw", "role": "Specialty (Asian)"},
    "Voila": {"label": "Voilà", "parent": "Empire (Sobeys)", "role": "Online conventional"},
    "Metro": {"label": "Metro", "parent": "Metro Inc.", "role": "Conventional"},
    "Walmart": {"label": "Walmart", "parent": "Walmart Canada", "role": "Mass discount (EDLP)"},
    "Galleria": {"label": "Galleria", "parent": "Independent", "role": "Specialty (Korean)"},
    "SaveOnFoods": {
        "label": "Save-On-Foods",
        "parent": "Pattison Food Group",
        "role": "Conventional (Western, out-of-market control)",
    },
}
# Sobeys appears for only 4 days (Oct 2026) and is excluded.
EXCLUDED_VENDORS = ("Sobeys",)

BIG3_PARENTS = ("Loblaw", "Empire (Sobeys)", "Metro Inc.")
# The banners that compete head-to-head for Toronto mainstream grocery trips.
CORE_TORONTO = ("Loblaws", "NoFrills", "Metro", "Walmart")

# Private-label (store brand) rules, applied to lower(brand || ' ' || product_name).
PRIVATE_LABEL_PATTERNS: dict[str, str] = {
    "Loblaws": r"president's choice|^pc |\bpc organics|\bpc blue menu|no name|farmer's market|\bt&t\b",
    "NoFrills": r"president's choice|^pc |\bpc organics|\bpc blue menu|no name|farmer's market|\bt&t\b",
    "TandT": r"^t&t|president's choice|^pc |no name",
    "Metro": r"selection|irr[ée]sistible|life smart|front street bakery|metrogo",
    "Walmart": r"great value|equate|mainstays|your fresh market|our finest",
    "Voila": r"compliments|farm boy|longo's",
    "SaveOnFoods": r"western family|save-on-foods|bake shop|only goodness|urban fare",
    "Galleria": r"^$a",  # no store brand identified
}

# Ordered keyword rules for a coarse category map (first match wins).
# Deliberately simple and transparent; it is used for descriptive cuts only.
CATEGORY_RULES: list[tuple[str, str]] = [
    ("Household & personal care", r"detergent|dish soap|shampoo|conditioner|toothpaste|toilet paper|bathroom tissue|"
     r"paper towel|diaper|deodorant|body wash|bleach|cleaner|garbage bag|trash bag|laundry|facial tissue|wipes|"
     r"razor|lotion|sunscreen|mouthwash|floss|feminine|pads|tampon"),
    ("Frozen", r"frozen|ice cream|popsicle|freezer"),
    ("Pantry", r"peanut butter|almond butter|nut butter|cookie butter"),
    ("Beverages", r"juice|water|soda|\bpop\b|coffee|\btea\b|drink|beverage|cola|sparkling|kombucha|lemonade|espresso"),
    ("Dairy & eggs", r"milk|cheese|yogurt|yoghurt|butter|cream|\beggs?\b|kefir|margarine|cottage|sour cream"),
    ("Bakery", r"bread|bagel|\bbuns?\b|muffin|croissant|tortilla|\bwraps?\b|cake|pita|loaf|\brolls?\b|naan|baguette"),
    ("Meat & seafood", r"chicken|beef|pork|turkey|\bham\b|bacon|sausage|salmon|shrimp|\bfish\b|tuna|lamb|steak|"
     r"ground|wiener|hot dog|salami|prosciutto|cod|tilapia|scallop|crab|lobster"),
    ("Snacks & confectionery", r"chips?\b|cookie|cracker|chocolate|candy|\bbars?\b|popcorn|\bnuts?\b|pretzel|gum|"
     r"gummies|licorice|snack|almonds|cashew|pistachio"),
    ("Produce", r"apple|banana|orange|lemon|\blime|grape|berr|melon|\bpear|peach|plum|mango|pineapple|avocado|"
     r"tomato|potato|onion|carrot|lettuce|spinach|kale|cucumber|broccoli|cauliflower|celery|mushroom|zucchini|"
     r"garlic|cabbage|squash|kiwi|cherries|bell pepper|herbs|cilantro|parsley|salad"),
    ("Pantry", r"pasta|spaghetti|rice|sauce|soup|cereal|flour|sugar|\boil\b|vinegar|spice|beans|canned|noodle|"
     r"ketchup|mustard|mayo|jam|oats|oatmeal|syrup|honey|salsa|dressing|seasoning|broth|lentil|chickpea|"
     r"granola|crackers|baking|tomato paste|stock|condiment|pickle|olive"),
]

# Event-detection parameters for regular-price changes.
MIN_CHANGE_PCT = 0.01      # ignore sub-1% moves (rounding / unit noise)
MAX_GAP_DAYS = 3           # the change date must be known to within 3 days
PERSIST_DAYS = 14          # a change reversed within 14 days is a temporary price cut
SYNC_WINDOWS = (0, 1, 3, 7)  # +/- days for "synchronised" changes
PLACEBO_SHIFTS = (-182, -126, -91, -63, 63, 91, 126, 182)  # days; time-shift placebo baseline

# Industry "price freeze" window that Metro publicly described as an industry
# practice (Nov 1 - Feb 5) — tested, not assumed.
FREEZE_START = (11, 1)
FREEZE_END = (2, 5)
