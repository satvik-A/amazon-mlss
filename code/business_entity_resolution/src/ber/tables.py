"""SMALL hand-written normalisation dictionaries (explicitly allowed by the organisers, answers 2/8).
Everything larger is LEARNED from the provided records (see artifacts.py)."""

# street-type abbreviations -> canonical (US, India, France)
STREET = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "bd": "boulevard",
    "ln": "lane", "dr": "drive", "ct": "court", "cir": "circle", "pl": "place", "pkwy": "parkway", "hwy": "highway",
    "sq": "square", "ter": "terrace", "trl": "trail", "cres": "crescent", "mkt": "market", "ngr": "nagar", "clny": "colony",
    # France
    "r": "rue", "all": "allee", "imp": "impasse", "ch": "chemin", "rte": "route", "fbg": "faubourg", "crs": "cours",
}
# per-country overrides of STREET (French 'St' is Saint, not Street)
STREET_COUNTRY = {"France": {"st": "saint", "ste": "sainte", "sts": "saints", "stes": "saintes"}}
# address function words / number markers dropped from address words (N° 5, No 5, 'de la', ...)
ADDR_STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "n", "no", "nos", "num", "of", "the", "and", "et", "au", "aux", "en"}
# never a unit code: number markers and street types ('No 15', 'Rte 66' are not apartment units)
UNIT_EXCLUDE = {"n", "no", "nos", "num", "nr"} | set(STREET)
# ordinal / number words -> digits
NUMBER_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8",
    "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12", "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
    "sixteenth": "16", "seventeenth": "17", "eighteenth": "18", "nineteenth": "19", "twentieth": "20",
    "premier": "1", "premiere": "1", "deuxieme": "2", "troisieme": "3",
}
# legal forms (all countries); removed from the name "core", kept as a separate field
LEGAL = {
    "inc", "incorporated", "llc", "ltd", "limited", "pvt", "private", "public", "corp", "corporation", "co", "company",
    "lp", "llp", "pllc", "pc", "plc", "gmbh", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "cie", "compagnie",
}
# words that introduce an alias ("X f/k/a Y"); the text AFTER the marker is usually the real name (verified in artifacts)
ALIAS_MARKERS = ["f/k/a", "a/k/a", "d/b/a", "t/a", "fka", "aka", "dba", "formerly", "trading as", "doing business as"]
# landmark phrases (address)
LANDMARK = {"near", "nr", "opp", "opposite", "behind", "beside", "next", "adjacent", "facing", "pres", "face"}
# digits used as look-alike letters inside NAME words only
LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "9": "g"}

# honorific / prefix noise the generator inserts into names (removed on BOTH sides for blocking keys; kept for features)
HONORIFIC = {"the", "mr", "mrs", "ms", "dr", "smt", "shri", "sri", "sree", "shree", "m", "s", "messrs"}
# function words in names ('Clinique de Jean' = 'Clinique du Jean'); removed from the core like honorifics
NAME_STOP = {"and", "of", "de", "du", "des", "la", "le", "les", "l", "d", "et", "en", "au", "aux"}

# ---- region-specific place names (hand tables; active with BER_PLACES=1) --------------------------------------------
# Old / new / variant spellings of Indian places, measured on train true pairs where only the alias links S1 and copy
# (Kolkata/Calcutta 17.8k, Odisha/Orissa 13.5k, Bengaluru/Bangalore 10.1k, Thiruvananthapuram/Trivandrum 6.7k, ...).
# The canonical token is ADDED (the original word stays), so both spellings meet on it.
PLACE_ALIAS = {"India": {
    "calcutta": "kolkata", "kolkatta": "kolkata", "bombay": "mumbai", "bangalore": "bengaluru", "banglore": "bengaluru",
    "bengalooru": "bengaluru", "madras": "chennai", "gurgaon": "gurugram", "trivandrum": "thiruvananthapuram",
    "calicut": "kozhikode", "trichy": "tiruchirappalli", "tiruchi": "tiruchirappalli", "baroda": "vadodara",
    "vizag": "visakhapatnam", "vishakhapatnam": "visakhapatnam", "poona": "pune", "cochin": "kochi", "ernakulam": "kochi",
    "mysore": "mysuru", "allahabad": "prayagraj", "mangalore": "mangaluru", "belgaum": "belagavi", "gulbarga": "kalaburagi",
    "hubli": "hubballi", "ahmadabad": "ahmedabad", "nasik": "nashik", "kancheepuram": "kanchipuram",
    "tiruvallur": "thiruvallur", "rangareddi": "rangareddy", "paraganas": "parganas", "keralam": "kerala",
    "orissa": "odisha", "secunderabad": "hyderabad", "benaras": "varanasi", "banaras": "varanasi",
    "pondicherry": "puducherry", "simla": "shimla", "tuticorin": "thoothukudi", "cawnpore": "kanpur",
    "trichur": "thrissur", "quilon": "kollam", "alleppey": "alappuzha", "cuddapah": "kadapa", "gauhati": "guwahati",
    "jubbulpore": "jabalpur", "bhubaneshwar": "bhubaneswar"}}
# multi-word places -> one canonical token (a word shared by several states, "pradesh", must not chain them together)
PLACE_PHRASE = {"India": [
    (("andhra", "pradesh"), "st_ap_tg"), (("telangana",), "st_ap_tg"), (("telangaan",), "st_ap_tg"), (("aandhrapradesh",), "st_ap_tg"),
    (("madhya", "pradesh"), "st_mp"), (("madhy", "pradesh"), "st_mp"), (("uttar", "pradesh"), "st_up"), (("himachal", "pradesh"), "st_hp"),
    (("arunachal", "pradesh"), "st_ar"), (("west", "bengal"), "st_wb"), (("pashchimabangg",), "st_wb"), (("tamil", "nadu"), "st_tn"),
    (("gautam", "buddha", "nagar"), "noida"), (("ranga", "reddy"), "rangareddy"), (("navi", "mumbai"), "navimumbai"),
    (("new", "bombay"), "navimumbai"), (("chhatrapati", "sambhaji", "nagar"), "aurangabad"), (("sawai", "madhopur"), "sawaimadhopur"),
    (("new", "delhi"), "delhi")]}
# learned address synonyms that touch these words chain unrelated states (pradesh -> mp, up, telangana; west -> wv)
PLACE_SYN_BLOCK = {"pradesh", "west", "east", "north", "south", "bengal", "nadu", "tamil", "uttar", "madhya", "andhra",
                   "himachal", "arunachal", "w", "e", "n", "s"}
# generator filler in copy addresses with no counterpart in the S1 ("Door No 12", literal "null" / "<NULL>")
PLACE_ADDR_STOP = {"door", "dor", "doro", "null"}
