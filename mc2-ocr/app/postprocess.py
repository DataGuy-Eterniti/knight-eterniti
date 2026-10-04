"""Deterministic clean-up of the VLM's answer into the graded transcription."""
import re
import unicodedata

STATES = [
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", "CONNECTICUT",
    "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO", "ILLINOIS", "INDIANA", "IOWA",
    "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN",
    "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA",
    "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK", "NORTH CAROLINA",
    "NORTH DAKOTA", "OHIO", "OKLAHOMA", "OREGON", "PENNSYLVANIA", "RHODE ISLAND",
    "SOUTH CAROLINA", "SOUTH DAKOTA", "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA",
    "WASHINGTON", "WEST VIRGINIA", "WISCONSIN", "WYOMING", "DISTRICT OF COLUMBIA",
]

SLOGANS = [
    "THE LONE STAR STATE", "LONE STAR STATE", "EMPIRE STATE", "THE EMPIRE STATE",
    "SUNSHINE STATE", "LAND OF LINCOLN", "GARDEN STATE", "LIVE FREE OR DIE",
    "PURE MICHIGAN", "GREAT LAKES STATE", "FIRST IN FLIGHT", "FIRST IN FREEDOM",
    "THE FIRST STATE", "GRAND CANYON STATE", "THE GOLDEN STATE", "GOLDEN STATE",
    "EXCELSIOR", "VOLUNTEER STATE", "THE VOLUNTEER STATE", "LAND OF ENCHANTMENT",
    "FAMOUS POTATOES", "BIG SKY COUNTRY", "TREASURE STATE", "THE PEACH STATE",
    "PEACH STATE", "SHOW ME STATE", "THE SHOW ME STATE", "KEYSTONE STATE",
    "VISITPA COM", "BIRTHPLACE OF AVIATION", "THE HAWKEYE STATE", "GREEN MOUNTAIN STATE",
    "WILD WONDERFUL", "ALMOST HEAVEN", "AMERICA'S DAIRYLAND", "AMERICAS DAIRYLAND",
    "10,000 LAKES", "10000 LAKES", "SPORTSMAN'S PARADISE", "SPORTSMANS PARADISE",
    "THE OCEAN STATE", "OCEAN STATE", "CONSTITUTION STATE", "THE SILVER STATE",
    "SILVER STATE", "THE BEEHIVE STATE", "LIFE ELEVATED", "GREATEST SNOW ON EARTH",
    "VACATIONLAND", "THE SPIRIT OF AMERICA", "SPIRIT OF AMERICA", "THE NATURAL STATE",
    "NATURAL STATE", "ALOHA STATE", "PRAIRIE STATE", "THE COWBOY STATE", "SOONER STATE",
    "NATIVE AMERICA", "TAXATION WITHOUT REPRESENTATION", "END TAXATION WITHOUT REPRESENTATION",
    "DMV CA GOV", "USA", "U S A", "COUNTY", "DEALER", "STATE OF",
]

_PHRASES = sorted(set(STATES + SLOGANS), key=len, reverse=True)
_PHRASE_RE = re.compile(
    r"(?<![A-Z0-9])(?:" + "|".join(re.escape(p) for p in _PHRASES) + r")(?![A-Z0-9])")
_URL_RE = re.compile(r"\S*(?:WWW|\.COM|\.GOV|\.ORG|\.NET|\.US)\S*")
_DOTS = str.maketrans({"・": "·", "‧": "·", "•": "·", "∙": "·", "•": "·", "･": "·"})


def _parse(raw):
    kind, text = "SIGN", None
    for line in raw.splitlines():
        m = re.match(r"\s*\**TYPE\**\s*[:：]\s*(.+)", line, re.I)
        if m:
            kind = m.group(1).strip().upper()
            continue
        m = re.match(r"\s*\**TEXT\**\s*[:：]\s*(.*)", line, re.I)
        if m:
            text = m.group(1)
    if text is None:  # model ignored the format: use everything except TYPE lines
        text = " ".join(l for l in raw.splitlines() if not re.match(r"\s*TYPE\s*[:：]", l, re.I))
    if "CN" in kind or "CHIN" in kind:
        kind = "CN_PLATE"
    elif "US" in kind:
        kind = "US_PLATE"
    elif "PLATE" in kind:
        kind = "OTHER_PLATE"
    else:
        kind = "SIGN"
    return kind, text


def strip_banners(text):
    t = _URL_RE.sub(" ", text)
    t = t.replace(",", "")  # so "10,000 LAKES" style slogans match either way
    t = _PHRASE_RE.sub(" ", t)
    t = re.sub(r"\b(?:THE|OF|STATE)\b", " ", t)  # leftovers from partial slogans
    return re.sub(r"\s+", " ", t).strip()


def clean(raw):
    """Return (text, kind)."""
    kind, text = _parse(raw or "")
    text = unicodedata.normalize("NFKC", text).translate(_DOTS)
    text = text.strip().strip("`\"'“”‘’").strip()
    text = re.sub(r"^(?:TEXT|ANSWER|PLATE|SIGN)\s*[:：]\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip().upper()  # upper() leaves Chinese untouched
    if kind in ("US_PLATE", "OTHER_PLATE"):
        stripped = strip_banners(text)
        if re.search(r"[A-Z0-9]", stripped):  # never strip the answer away entirely
            text = stripped
    return text, kind
