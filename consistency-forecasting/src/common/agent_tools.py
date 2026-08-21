import os
import json
import logging
import subprocess
import sys

try:
    from ddgs import DDGS
except ImportError:
    logging.warning("ddgs not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ddgs"])
    from ddgs import DDGS

# For RAG
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Universal Trusted Source Registry  (110+ sources, 15 domain categories)
# Each entry: (domain_string, priority_score, category_tags)
# Score 1 = tier-1 authoritative (official/primary).
# Score 2 = tier-2 authoritative (major established outlet).
# Score 3 = tier-3 reliable (specialist/regional).
# Score 4 = tier-4 supplementary (community/aggregator).
# ─────────────────────────────────────────────────────────────────────────────
_UNIVERSAL_TRUSTED_SOURCES = [

    # ══ WIRE SERVICES & GLOBAL BREAKING NEWS ═════════════════════════════════
    ("reuters.com",             1, ["news", "politics", "economics", "science", "sports", "health", "geopolitics", "energy", "military"]),
    ("apnews.com",              1, ["news", "politics", "economics", "science", "sports", "health", "geopolitics"]),
    ("afp.com",                 1, ["news", "politics", "sports", "geopolitics"]),
    ("bbc.com",                 1, ["news", "politics", "economics", "science", "sports", "health", "culture"]),
    ("bbc.co.uk",               1, ["news", "politics", "economics", "science", "sports", "health", "culture"]),
    ("dw.com",                  2, ["news", "politics", "geopolitics", "europe"]),
    ("aljazeera.com",           2, ["news", "politics", "geopolitics", "middle_east"]),
    ("france24.com",            2, ["news", "politics", "geopolitics", "europe"]),
    ("rfi.fr",                  3, ["news", "politics", "africa", "geopolitics"]),
    ("nhk.or.jp",               2, ["news", "politics", "asia"]),
    ("channelnewsasia.com",     2, ["news", "politics", "asia", "southeast_asia"]),
    ("straitstimes.com",        2, ["news", "politics", "asia", "southeast_asia"]),
    ("scmp.com",                2, ["news", "politics", "asia", "china", "economics"]),
    ("themoscowtimes.com",      3, ["news", "politics", "russia", "geopolitics"]),

    # ══ MAJOR WESTERN NEWSPAPERS & MAGAZINES ══════════════════════════════════
    ("theguardian.com",         2, ["news", "politics", "economics", "science", "sports", "health", "culture", "environment"]),
    ("nytimes.com",             2, ["news", "politics", "economics", "science", "culture", "health"]),
    ("washingtonpost.com",      2, ["news", "politics", "economics", "science", "geopolitics"]),
    ("theatlantic.com",         2, ["news", "politics", "science", "culture"]),
    ("time.com",                2, ["news", "politics", "culture", "science", "health"]),
    ("newsweek.com",            3, ["news", "politics", "science", "culture"]),
    ("politico.com",            2, ["politics", "policy", "geopolitics"]),
    ("foreignpolicy.com",       2, ["geopolitics", "politics", "military", "international"]),
    ("foreignaffairs.com",      2, ["geopolitics", "politics", "international"]),
    ("spiegel.de",              2, ["news", "politics", "europe", "geopolitics"]),
    ("lemonde.fr",              2, ["news", "politics", "europe", "culture"]),
    ("elpais.com",              3, ["news", "politics", "europe", "culture"]),

    # ══ ECONOMICS, FINANCE & BUSINESS ═════════════════════════════════════════
    ("bloomberg.com",           2, ["economics", "finance", "business", "politics", "energy", "markets"]),
    ("ft.com",                  2, ["economics", "finance", "business", "markets"]),
    ("economist.com",           2, ["economics", "politics", "science", "development"]),
    ("wsj.com",                 2, ["economics", "finance", "business", "politics", "markets"]),
    ("cnbc.com",                2, ["finance", "economics", "business", "markets", "technology"]),
    ("marketwatch.com",         3, ["finance", "markets", "economics"]),
    ("investopedia.com",        3, ["finance", "markets", "economics", "crypto"]),
    ("barrons.com",             3, ["finance", "markets", "economics"]),
    ("morningstar.com",         3, ["finance", "markets"]),
    ("seeking alpha.com",       4, ["finance", "markets"]),
    ("tradingeconomics.com",    3, ["economics", "finance", "markets", "development"]),
    ("statista.com",            3, ["economics", "statistics", "markets", "general"]),
    ("imf.org",                 1, ["economics", "finance", "currency", "development"]),
    ("worldbank.org",           1, ["economics", "development", "finance"]),
    ("oecd.org",                1, ["economics", "education", "policy", "development"]),
    ("bis.org",                 1, ["economics", "finance", "currency", "banking"]),
    ("federalreserve.gov",      1, ["economics", "finance", "currency", "policy"]),
    ("ecb.europa.eu",           1, ["economics", "finance", "currency", "europe"]),
    ("stats.bis.org",           1, ["economics", "finance", "banking"]),

    # ══ POLITICS & GOVERNANCE ═════════════════════════════════════════════════
    ("whitehouse.gov",          1, ["politics", "policy", "usa"]),
    ("congress.gov",            1, ["politics", "law", "policy", "usa"]),
    ("europa.eu",               1, ["politics", "law", "economics", "europe"]),
    ("un.org",                  1, ["politics", "geopolitics", "law", "environment", "development"]),
    ("nato.int",                1, ["military", "geopolitics", "politics", "security"]),
    ("state.gov",               1, ["geopolitics", "politics", "international", "usa"]),
    ("gov.uk",                  1, ["politics", "policy", "law", "economics"]),
    ("electionguide.org",       2, ["politics", "election"]),
    ("freedomhouse.org",        2, ["politics", "geopolitics", "law"]),
    ("cfr.org",                 2, ["geopolitics", "politics", "international", "military"]),
    ("brookings.edu",           2, ["politics", "economics", "policy", "development"]),
    ("iiss.org",                2, ["military", "geopolitics", "security", "defense"]),

    # ══ GEOPOLITICS & INTERNATIONAL AFFAIRS ═══════════════════════════════════
    ("icj-cij.org",             1, ["law", "geopolitics", "international"]),
    ("hrw.org",                 2, ["politics", "law", "geopolitics", "human_rights"]),
    ("amnesty.org",             2, ["politics", "law", "geopolitics", "human_rights"]),
    ("iccnow.org",              2, ["law", "geopolitics", "international"]),
    ("sipri.org",               2, ["military", "geopolitics", "security", "defense"]),
    ("crisisgroup.org",         2, ["geopolitics", "military", "conflict", "politics"]),
    ("globaltimes.cn",          3, ["news", "politics", "china", "asia"]),

    # ══ HEALTH & MEDICINE ════════════════════════════════════════════════════
    ("who.int",                 1, ["health", "medicine", "pandemic"]),
    ("cdc.gov",                 1, ["health", "medicine", "pandemic", "usa"]),
    ("ecdc.europa.eu",          1, ["health", "medicine", "pandemic", "europe"]),
    ("nature.com",              1, ["science", "medicine", "climate", "biology", "health"]),
    ("thelancet.com",           1, ["health", "medicine", "pandemic"]),
    ("nejm.org",                1, ["health", "medicine"]),
    ("pubmed.ncbi.nlm.nih.gov", 1, ["health", "medicine", "biology", "science"]),
    ("bmj.com",                 1, ["health", "medicine"]),
    ("webmd.com",               3, ["health", "medicine"]),
    ("mayoclinic.org",          2, ["health", "medicine"]),
    ("nih.gov",                 1, ["health", "medicine", "science", "biology"]),
    ("fda.gov",                 1, ["health", "medicine", "policy", "usa"]),

    # ══ SCIENCE, SPACE & ENVIRONMENT ════════════════════════════════════════
    ("science.org",             1, ["science", "medicine", "climate", "biology"]),
    ("arxiv.org",               2, ["science", "technology", "ai", "math", "physics"]),
    ("nasa.gov",                1, ["science", "space", "climate"]),
    ("esa.int",                 1, ["science", "space"]),
    ("noaa.gov",                1, ["climate", "weather", "environment", "science"]),
    ("climate.nasa.gov",        1, ["climate", "environment", "science"]),
    ("ipcc.ch",                 1, ["climate", "environment", "science"]),
    ("iea.org",                 1, ["energy", "climate", "economics", "environment"]),
    ("newscientist.com",        2, ["science", "health", "technology", "environment"]),
    ("scientificamerican.com",  2, ["science", "health", "technology", "environment", "climate"]),
    ("nationalgeographic.com",  3, ["science", "environment", "culture", "climate"]),

    # ══ TECHNOLOGY, AI & CYBERSECURITY ═══════════════════════════════════════
    ("techcrunch.com",          2, ["technology", "ai", "startups", "finance"]),
    ("theverge.com",            2, ["technology", "ai", "consumer", "culture"]),
    ("wired.com",               2, ["technology", "science", "culture", "ai"]),
    ("arstechnica.com",         2, ["technology", "science", "cybersecurity"]),
    ("mit.edu",                 2, ["technology", "ai", "science", "research"]),
    ("technologyreview.com",    2, ["technology", "ai", "science"]),
    ("venturebeat.com",         3, ["technology", "ai", "startups"]),
    ("zdnet.com",               3, ["technology", "cybersecurity", "ai"]),
    ("securityweek.com",        3, ["cybersecurity", "technology"]),
    ("krebs on security.com",   3, ["cybersecurity", "technology"]),
    ("openai.com",              2, ["ai", "technology"]),
    ("deepmind.com",            2, ["ai", "science", "technology"]),
    ("anthropic.com",           2, ["ai", "technology"]),
    ("huggingface.co",          3, ["ai", "technology", "research"]),

    # ══ CRYPTO & BLOCKCHAIN ══════════════════════════════════════════════════
    ("coindesk.com",            2, ["crypto", "finance", "technology", "blockchain"]),
    ("cointelegraph.com",       2, ["crypto", "finance", "blockchain"]),
    ("decrypt.co",              3, ["crypto", "blockchain", "technology"]),
    ("theblock.co",             2, ["crypto", "finance", "blockchain"]),
    ("messari.io",              3, ["crypto", "finance", "markets"]),
    ("glassnode.com",           3, ["crypto", "finance", "markets"]),

    # ══ ENERGY & COMMODITIES ═════════════════════════════════════════════════
    ("opec.org",                1, ["energy", "economics", "commodities"]),
    ("eia.gov",                 1, ["energy", "economics", "environment", "usa"]),
    ("oilprice.com",            3, ["energy", "commodities", "economics"]),
    ("spglobal.com",            2, ["energy", "finance", "commodities", "economics"]),
    ("woodmac.com",             3, ["energy", "economics", "commodities"]),

    # ══ SPORTS ═══════════════════════════════════════════════════════════════
    ("fifa.com",                1, ["sports", "football", "soccer"]),
    ("uefa.com",                1, ["sports", "football"]),
    ("olympics.com",            1, ["sports", "olympics"]),
    ("espn.com",                2, ["sports"]),
    ("skysports.com",           2, ["sports"]),
    ("bbc.com/sport",           1, ["sports"]),
    ("goal.com",                3, ["sports", "football"]),
    ("transfermarkt.com",       3, ["sports", "football"]),
    ("athletic.com",            2, ["sports"]),
    ("nba.com",                 1, ["sports", "basketball"]),
    ("nfl.com",                 1, ["sports", "american_football"]),
    ("atptour.com",             1, ["sports", "tennis"]),
    ("wtatennis.com",           1, ["sports", "tennis"]),
    ("formula1.com",            1, ["sports", "motorsport"]),
    ("cricinfo.com",            2, ["sports", "cricket"]),
    ("worldathletics.org",      1, ["sports", "athletics", "olympics"]),
    ("nhl.com",                 1, ["sports", "ice_hockey"]),
    ("mlb.com",                 1, ["sports", "baseball"]),

    # ══ MILITARY & DEFENSE ═══════════════════════════════════════════════════
    ("nato.int",                1, ["military", "geopolitics", "politics", "security"]),
    ("janes.com",               2, ["military", "defense", "security", "geopolitics"]),
    ("defensenews.com",         2, ["military", "defense", "geopolitics"]),
    ("breakingdefense.com",     3, ["military", "defense", "technology"]),
    ("rand.org",                2, ["military", "policy", "geopolitics", "economics"]),
    ("warisboring.com",         3, ["military", "geopolitics", "conflict"]),

    # ══ CULTURE, ENTERTAINMENT & MEDIA ═══════════════════════════════════════
    ("variety.com",             2, ["culture", "entertainment", "media"]),
    ("hollywoodreporter.com",   2, ["culture", "entertainment", "media"]),
    ("rollingstone.com",        3, ["culture", "entertainment", "music"]),
    ("pitchfork.com",           3, ["culture", "music", "entertainment"]),
    ("imdb.com",                3, ["culture", "entertainment"]),
    ("boxofficemojo.com",       3, ["culture", "entertainment", "finance"]),
    ("rottentomatoes.com",      4, ["culture", "entertainment"]),

    # ══ EDUCATION & ACADEMIA ═════════════════════════════════════════════════
    ("harvard.edu",             2, ["education", "research", "science", "economics"]),
    ("stanford.edu",            2, ["education", "research", "technology", "science"]),
    ("ox.ac.uk",                2, ["education", "research", "science", "economics"]),
    ("cam.ac.uk",               2, ["education", "research", "science"]),
    ("ssrn.com",                3, ["research", "economics", "law", "finance"]),
    ("scholar.google.com",      3, ["research", "science", "general"]),

    # ══ HUMANITARIAN & DEVELOPMENT ═══════════════════════════════════════════
    ("unhcr.org",               1, ["human_rights", "development", "geopolitics"]),
    ("unicef.org",              1, ["health", "development", "human_rights"]),
    ("wfp.org",                 1, ["development", "human_rights", "economics"]),
    ("ifrc.org",                1, ["human_rights", "health", "development"]),
    ("oxfam.org",               2, ["development", "economics", "human_rights"]),

    # ══ SOUTHEAST ASIA / VIETNAM REGIONAL ════════════════════════════════════
    ("vnexpress.net",           2, ["news", "politics", "economics", "southeast_asia", "vietnam"]),
    ("tuoitre.vn",              2, ["news", "politics", "economics", "southeast_asia", "vietnam"]),
    ("thanhnien.vn",            2, ["news", "politics", "culture", "southeast_asia", "vietnam"]),
    ("vietnamnews.vn",          2, ["news", "politics", "economics", "vietnam"]),
    ("nhandan.vn",              2, ["news", "politics", "vietnam"]),
    ("bangkokpost.com",         2, ["news", "politics", "economics", "southeast_asia"]),
    ("thejakartapost.com",      2, ["news", "politics", "economics", "southeast_asia"]),
    ("philstar.com",            3, ["news", "politics", "southeast_asia"]),
    ("malaymail.com",           3, ["news", "politics", "southeast_asia"]),
    ("khmertimeskh.com",        3, ["news", "politics", "southeast_asia"]),
    ("nationthailand.com",      3, ["news", "politics", "southeast_asia"]),

    # ══ ENCYCLOPEDIC / REFERENCE ════════════════════════════════════════════
    ("en.wikipedia.org",        4, ["general"]),
    ("britannica.com",          3, ["general", "education"]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Keyword → category mapping for query classification  (15 categories)
# ─────────────────────────────────────────────────────────────────────────────
_CATEGORY_KEYWORDS = {
    # ── Sports ──────────────────────────────────────────────────────────────
    "sports": [
        "world cup", "fifa", "euros", "copa america", "olympics", "paralympics",
        "championship", "champions league", "premier league", "la liga", "serie a",
        "bundesliga", "tournament", "match", "game", "fixture", "derby",
        "nba", "nfl", "nhl", "mlb", "mls", "nba finals",
        "tennis", "wimbledon", "us open", "french open", "australian open",
        "cricket", "ipl", "test match",
        "football", "soccer", "basketball", "baseball", "rugby", "golf",
        "formula 1", "f1", "motogp", "grand prix",
        "marathon", "athletics", "swimming", "cycling", "tour de france",
        "ufc", "boxing", "mma", "wrestling",
        "eliminated", "knockout", "final", "semifinal", "quarter final",
        "gold medal", "silver medal", "bronze medal",
        "athlete", "player", "coach", "transfer", "squad",
    ],

    # ── Politics & Elections ──────────────────────────────────────────────
    "politics": [
        "election", "elections", "vote", "voting", "ballot", "polling",
        "president", "prime minister", "chancellor", "premier", "secretary of state",
        "government", "cabinet", "minister", "senator", "congressman",
        "senate", "congress", "parliament", "house of representatives",
        "legislation", "bill", "policy", "reform", "mandate",
        "coup", "referendum", "impeachment", "inauguration",
        "democrat", "republican", "conservative", "labour", "liberal",
        "party", "coalition", "opposition",
        "sanction", "tariff", "treaty", "accord", "summit",
        "nato", "g7", "g20", "asean", "apec", "eu", "brics",
    ],

    # ── Geopolitics & International Affairs ──────────────────────────────
    "geopolitics": [
        "war", "conflict", "invasion", "ceasefire", "peace talks", "diplomacy",
        "territorial", "sovereignty", "annexation", "occupation",
        "ukraine", "russia", "china", "taiwan", "north korea", "iran",
        "middle east", "israel", "palestine", "gaza",
        "south china sea", "strait of taiwan", "arctic",
        "un security council", "nato", "nuclear",
        "embargo", "blockade", "geopolitical", "alliance",
        "refugee", "asylum", "border", "migration",
    ],

    # ── Economics & Macro ──────────────────────────────────────────────────
    "economics": [
        "gdp", "gnp", "economic growth", "recession", "depression",
        "inflation", "deflation", "cpi", "ppi", "pce",
        "interest rate", "monetary policy", "fiscal policy", "austerity",
        "federal reserve", "fed", "ecb", "central bank", "rate hike", "rate cut",
        "unemployment", "jobs", "labor market", "payroll",
        "trade deficit", "trade surplus", "export", "import",
        "supply chain", "tariff", "trade war",
        "imf", "world bank", "oecd", "wto",
        "fdi", "foreign investment", "development",
        "currency", "exchange rate", "dollar", "euro", "yuan",
    ],

    # ── Finance & Markets ─────────────────────────────────────────────────
    "finance": [
        "stock", "share", "equity", "nasdaq", "s&p 500", "dow jones", "ftse",
        "ipo", "listing", "spac", "offering",
        "merger", "acquisition", "takeover", "buyout", "m&a",
        "earnings", "revenue", "profit", "loss", "dividend",
        "bond", "yield", "treasury", "credit rating", "default",
        "market cap", "valuation", "hedge fund", "private equity",
        "venture capital", "funding round", "series a", "series b",
        "etf", "index fund", "portfolio",
    ],

    # ── Crypto & Blockchain ────────────────────────────────────────────────
    "crypto": [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
        "blockchain", "defi", "nft", "token", "altcoin",
        "solana", "cardano", "xrp", "ripple", "dogecoin",
        "stablecoin", "usdt", "usdc", "cbdc",
        "mining", "halving", "wallet", "exchange", "binance", "coinbase",
        "smart contract", "web3", "dao", "layer 2",
        "crypto regulation", "sec crypto", "etf bitcoin",
    ],

    # ── Health & Medicine ──────────────────────────────────────────────────
    "health": [
        "covid", "covid-19", "coronavirus", "pandemic", "epidemic",
        "vaccine", "vaccination", "booster", "mrna",
        "virus", "bacteria", "pathogen", "outbreak", "transmission",
        "disease", "cancer", "diabetes", "alzheimer", "dementia",
        "clinical trial", "drug approval", "fda", "ema", "who",
        "hospital", "icu", "surgery", "treatment", "therapy",
        "mental health", "depression", "anxiety",
        "mortality", "life expectancy", "public health",
        "antibiotic", "resistance", "hiv", "aids", "tuberculosis",
        "obesity", "nutrition", "exercise",
    ],

    # ── Science & Research ────────────────────────────────────────────────
    "science": [
        "climate change", "global warming", "co2", "greenhouse gas", "emissions",
        "nasa", "spacex", "rocket", "satellite", "mars", "moon", "asteroid",
        "physics", "quantum", "particle", "cern",
        "biology", "genetics", "gene", "genome", "crispr", "dna", "rna",
        "evolution", "species", "ecology",
        "chemistry", "material science", "nanotechnology",
        "discovery", "research paper", "study", "journal", "arxiv",
        "peer review", "experiment", "hypothesis",
        "mathematics", "algorithm", "theorem",
    ],

    # ── Technology & AI ─────────────────────────────────────────────────
    "technology": [
        "artificial intelligence", "ai", "machine learning", "deep learning",
        "llm", "large language model", "gpt", "claude", "gemini", "llama",
        "chatgpt", "openai", "anthropic", "google deepmind",
        "chip", "semiconductor", "gpu", "cpu", "nvidia", "amd", "intel", "tsmc",
        "5g", "6g", "quantum computing",
        "cybersecurity", "hacking", "ransomware", "data breach", "malware",
        "tech", "software", "app", "platform", "startup",
        "apple", "google", "microsoft", "meta", "amazon", "tesla",
        "autonomous", "self-driving", "robot", "drone",
        "cloud", "saas", "open source",
    ],

    # ── Energy & Commodities ──────────────────────────────────────────────
    "energy": [
        "oil", "crude oil", "brent", "wti", "opec", "petroleum",
        "natural gas", "lng", "pipeline",
        "coal", "mining", "metals",
        "gold", "silver", "copper", "iron ore", "lithium",
        "renewable energy", "solar", "wind", "nuclear", "hydrogen",
        "energy transition", "clean energy", "carbon",
        "electricity", "power grid", "blackout",
        "eia", "iea", "opec+",
    ],

    # ── Environment & Climate ─────────────────────────────────────────────
    "environment": [
        "climate", "weather", "temperature", "warming",
        "deforestation", "biodiversity", "extinction", "habitat",
        "pollution", "plastic", "waste", "recycling",
        "flood", "drought", "wildfire", "hurricane", "typhoon", "earthquake",
        "el niño", "la niña",
        "cop", "paris agreement", "carbon neutral", "net zero",
        "renewable", "sustainability", "green",
        "ocean", "coral reef", "glacier", "arctic", "rainforest",
    ],

    # ── Military & Defense ────────────────────────────────────────────────
    "military": [
        "army", "navy", "air force", "marines", "military",
        "weapon", "missile", "drone", "warship", "aircraft carrier",
        "defense", "pentagon", "ministry of defense",
        "arms", "ammunition", "artillery", "tank",
        "soldier", "troops", "deployment", "combat",
        "war", "battle", "offensive", "counteroffensive", "airstrikes",
        "nuclear", "ballistic", "hypersonic",
        "nato", "military alliance", "defense spending",
    ],

    # ── Culture & Entertainment ───────────────────────────────────────────
    "culture": [
        "film", "movie", "cinema", "box office", "oscar", "golden globe",
        "tv", "television", "streaming", "netflix", "disney", "hbo",
        "music", "album", "concert", "grammy", "billboard",
        "book", "novel", "author", "literary prize", "booker",
        "art", "museum", "exhibition", "fashion", "design",
        "celebrity", "actor", "singer", "director",
        "award", "emmy", "bafta", "cannes", "venice",
        "game", "video game", "esports", "gaming",
        "social media", "tiktok", "youtube", "instagram",
    ],

    # ── Education & Academia ──────────────────────────────────────────────
    "education": [
        "university", "college", "school", "education",
        "research", "study", "student", "professor", "faculty",
        "tuition", "scholarship", "enrollment", "graduation",
        "ranking", "qs ranking", "times higher education",
        "phd", "postdoc", "thesis", "academic",
        "pisa", "literacy", "curriculum",
    ],

    # ── Human Rights & Development ────────────────────────────────────────
    "human_rights": [
        "human rights", "civil rights", "democracy", "freedom",
        "protest", "demonstration", "activist", "dissent",
        "refugee", "asylum seeker", "displacement", "migration",
        "poverty", "inequality", "discrimination",
        "child labor", "trafficking", "slavery",
        "humanitarian", "aid", "ngos", "unicef", "unhcr",
        "prison", "detention", "torture", "execution",
    ],

    # ── Southeast Asia & Vietnam ─────────────────────────────────────────
    "southeast_asia": [
        "vietnam", "viet nam", "hanoi", "ho chi minh", "saigon",
        "thailand", "bangkok", "indonesia", "jakarta",
        "philippines", "manila", "malaysia", "kuala lumpur",
        "singapore", "myanmar", "cambodia", "laos",
        "asean", "southeast asia", "mekong",
        "vietnamese", "thai", "indonesian", "filipino",
        "vingroup", "viettel", "petrovietnam", "vng",
    ],
}


def _detect_categories(query: str) -> list:
    """Detect which topic categories a query belongs to."""
    q_lower = query.lower()
    detected = []
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            detected.append(cat)
    if not detected:
        detected = ["news"]  # Default to general news
    return detected


def _get_trusted_domains_for_query(query: str) -> dict:
    """
    Return a dict of {domain: priority_score} relevant to the query.
    Strategy:
      - Always include tier-1/2 global news sources (universal coverage).
      - Add all sources whose tags overlap with the detected categories.
      - Include ALL sources as supplementary when no strong match found.
    """
    categories = _detect_categories(query)
    categories_set = set(categories) | {"news"}  # always include general news

    domain_scores = {}
    for domain, score, tags in _UNIVERSAL_TRUSTED_SOURCES:
        tag_set = set(tags)
        # Primary criterion: tag overlap with detected categories or general news
        if tag_set & categories_set or "news" in tags:
            domain_scores[domain] = score
        # Secondary: always include tier-1 sources regardless of category
        elif score == 1:
            domain_scores[domain] = score

    return domain_scores




def _fetch_article_text(url: str, max_chars: int = 2500) -> str:
    """
    Fetch and extract readable text from a URL.
    Returns empty string on failure (caller falls back to snippet).
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(
            url, timeout=5,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        if resp.status_code != 200:
            return ""

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Remove boilerplate tags
        for tag in soup(["script", "style", "nav", "footer", "aside",
                          "header", "form", "button", "iframe", "noscript"]):
            tag.decompose()

        # Try structured article content first (most reliable across domains)
        candidates = [
            soup.find('article'),
            soup.find(attrs={"role": "main"}),
            soup.find('main'),
            soup.find(attrs={"class": lambda c: c and any(
                kw in " ".join(c).lower() for kw in
                ["article", "story", "content", "post", "body", "text"]
            )}),
            soup.body,
        ]
        node = next((c for c in candidates if c is not None), None)
        if node is None:
            return ""

        text = node.get_text(separator=' ', strip=True)
        # Collapse excessive whitespace
        import re
        text = re.sub(r'\s{3,}', '  ', text)
        return text[:max_chars]

    except Exception:
        return ""


def _is_recent_query(query: str) -> bool:
    """
    Heuristic: returns True if the query looks like it's asking about a recent or
    current event (contains a year, 'result', 'latest', 'today', 'winner', etc.).
    Used to decide whether to apply DuckDuckGo's timelimit filter.
    """
    import re
    q = query.lower()
    recent_keywords = [
        "result", "results", "outcome", "winner", "score", "latest", "today",
        "current", "now", "announced", "confirmed", "decided", "happened",
        "live", "ongoing", "vs", "match", "play", "game",
        "2026", "2025", "2024",
    ]
    return any(kw in q for kw in recent_keywords)



def search_web(query: str, max_results: int = 5) -> str:
    """
    Universal web search using DuckDuckGo.

    Key improvements:
    1. Auto-detects query category and ranks results by trusted-source priority.
    2. Applies timelimit='w' (past 7 days) for queries about recent/current events
       so that just-happened results (match scores, policy announcements, etc.)
       appear at the top.
    3. Falls back to an unrestricted time search if the time-limited query returns
       nothing, then retries with a shorter query if still empty.
    4. Fetches full article text; supplements with snippet when article is short.
    """
    logging.info(f"Agentic Tool [search_web]: query='{query}'")
    try:
        categories = _detect_categories(query)
        domain_scores = _get_trusted_domains_for_query(query)
        logging.info(f"  → Detected categories: {categories}. Trusted domains: {len(domain_scores)}")

        fetch_count = max_results * 3  # fetch more to allow sorting/filtering
        recent = _is_recent_query(query)

        # Step 1: Time-limited search for recent-event queries (past 7 days)
        raw_results = []
        with DDGS() as ddgs:
            if recent:
                logging.info("  → Recent-event query detected — applying timelimit='w' (past 7 days).")
                raw_results = list(ddgs.text(query, max_results=fetch_count, timelimit="w"))
            if not raw_results:
                # Either not recent, or time-limited search was empty — try without limit
                raw_results = list(ddgs.text(query, max_results=fetch_count))

        # Step 2: Fallback with a shorter query if still empty
        if not raw_results:
            short_query = " ".join(query.split()[:4])
            logging.info(f"  → No results. Retrying with shorter query: '{short_query}'")
            with DDGS() as ddgs:
                if recent:
                    raw_results = list(ddgs.text(short_query, max_results=fetch_count, timelimit="w"))
                if not raw_results:
                    raw_results = list(ddgs.text(short_query, max_results=fetch_count))

        if not raw_results:
            return "No web results found."

        # Step 3: Score and sort by trusted-source priority
        def result_score(r):
            url = r.get("href", "") or ""
            for domain, score in domain_scores.items():
                if domain in url:
                    return score
            return 99  # unverified/lower priority

        raw_results.sort(key=result_score)

        # Step 4: Build summary
        summary = []
        seen_urls = set()
        fetched = 0
        for r in raw_results:
            if fetched >= max_results:
                break
            url     = r.get("href", "") or ""
            title   = r.get("title", "") or ""
            snippet = r.get("body", "") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            matched_domain = next((d for d in domain_scores if d in url), None)
            if matched_domain:
                score = domain_scores[matched_domain]
                trust_label = (
                    "[PRIMARY SOURCE]" if score <= 2
                    else "[SECONDARY SOURCE]"
                )
            else:
                trust_label = "[UNVERIFIED SOURCE]"

            # Full article text with snippet supplement
            full_text = _fetch_article_text(url, max_chars=2500)
            if full_text and len(full_text) < 300 and snippet:
                content = f"{full_text}\n[Snippet]: {snippet}"
            elif full_text:
                content = full_text
            else:
                content = f"[Snippet]: {snippet}" if snippet else "(no content retrieved)"

            summary.append(
                f"[{fetched + 1}] {trust_label}\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}"
            )
            fetched += 1

        if not summary:
            return "No usable results found after filtering."

        time_note = " | Time filter: past 7 days" if recent else ""
        header = (
            f"[Search Query: '{query}' | Topics: {', '.join(categories)}{time_note} | "
            f"Sources: {len(summary)}]\n\n"
        )
        return header + "\n\n---\n\n".join(summary)

    except Exception as e:
        return f"Error performing web search: {e}"


# Simple in-memory RAG
_RAG_CORPUS = []
_RAG_DOCS = []
_RAG_VECTORIZER = None
_RAG_TFIDF_MATRIX = None


def _init_rag(data_dir: str = "src/data/tuples"):
    global _RAG_CORPUS, _RAG_DOCS, _RAG_VECTORIZER, _RAG_TFIDF_MATRIX
    if _RAG_VECTORIZER is not None:
        return  # Already initialized

    _RAG_CORPUS = []
    _RAG_DOCS = []

    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Depending on structure, just dump it as string
                        content = json.dumps(data, ensure_ascii=False)
                        _RAG_CORPUS.append(content)
                        _RAG_DOCS.append(filename)
                except Exception:
                    pass

    if not _RAG_CORPUS:
        # Fallback dummy doc
        _RAG_CORPUS = [
            "Historical data indicates that LLM capabilities double every 12 months.",
            "Previous consistency tests show hybrid methods outperform basic models by 24%."
        ]
        _RAG_DOCS = ["dummy1.md", "dummy2.md"]

    _RAG_VECTORIZER = TfidfVectorizer(stop_words='english')
    _RAG_TFIDF_MATRIX = _RAG_VECTORIZER.fit_transform(_RAG_CORPUS)


def load_custom_rag_sources(rag_urls=None, rag_files=None):
    """Load custom URLs and local files into the RAG corpus."""
    global _RAG_CORPUS, _RAG_DOCS, _RAG_VECTORIZER, _RAG_TFIDF_MATRIX
    
    # Initialize basic first if not
    _init_rag()
    
    updated = False
    
    if rag_urls:
        import requests
        from bs4 import BeautifulSoup
        for url in rag_urls:
            url = url.strip()
            if not url or url in _RAG_DOCS: continue
            try:
                # Add headers to avoid bot detection and 403 blocks on news sites
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                }
                resp = requests.get(url, headers=headers, timeout=8)
                soup = BeautifulSoup(resp.text, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                if text:
                    _RAG_CORPUS.append(text)
                    _RAG_DOCS.append(url)
                    updated = True
            except Exception as e:
                logging.error(f"Failed to load RAG URL {url}: {e}")
                
    if rag_files:
        for file_item in rag_files:
            if isinstance(file_item, str):
                file_path = file_item.strip()
                if not file_path or not os.path.exists(file_path) or file_path in _RAG_DOCS: continue
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        _RAG_CORPUS.append(f.read())
                        _RAG_DOCS.append(file_path)
                        updated = True
                except Exception as e:
                    logging.error(f"Failed to load RAG file {file_path}: {e}")
            elif isinstance(file_item, dict):
                # Handle uploaded file contents: {"name": "...", "content": "..."}
                name = file_item.get("name", "uploaded_file")
                content = file_item.get("content", "")
                if content and name not in _RAG_DOCS:
                    _RAG_CORPUS.append(content)
                    _RAG_DOCS.append(name)
                    updated = True
                
    if updated and _RAG_CORPUS:
        _RAG_VECTORIZER = TfidfVectorizer(stop_words='english')
        _RAG_TFIDF_MATRIX = _RAG_VECTORIZER.fit_transform(_RAG_CORPUS)


def search_internal_docs(query: str, top_k: int = 2) -> str:
    """
    Search internal historical documents using a simple TF-IDF RAG approach.
    """
    logging.info(f"Agentic Tool [search_internal_docs]: {query}")
    try:
        _init_rag()
        if _RAG_TFIDF_MATRIX is None or _RAG_TFIDF_MATRIX.shape[0] == 0:
            return "Internal documentation corpus is empty."
            
        query_vec = _RAG_VECTORIZER.transform([query])
        sims = cosine_similarity(query_vec, _RAG_TFIDF_MATRIX).flatten()
        
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if sims[idx] > 0.05:  # Minimum similarity threshold
                results.append(f"--- Document: {_RAG_DOCS[idx]} ---\n{_RAG_CORPUS[idx][:1500]}...\n")
                
        if not results:
            return "No relevant internal documents found."
            
        return "\n".join(results)
    except Exception as e:
        return f"Error performing internal document search: {e}"


TOOLS_REGISTRY = {
    "search_web": search_web,
    "search_internal_docs": search_internal_docs
}


async def execute_research_phase_async(question_text: str, model: str, api_key: str = None, base_url: str = None, rag_urls=None, rag_files=None) -> str:
    """
    Run an async multi-turn research phase using OpenAI's tool calling natively.
    Domain-agnostic: works for sports, politics, economics, technology, science, etc.
    """
    if rag_urls or rag_files:
        load_custom_rag_sources(rag_urls, rag_files)
        
    import openai
    import json
    from datetime import date
    
    today = date.today().strftime("%Y-%m-%d")
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    # Domain-agnostic tool definitions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": (
                    "Search the web for real-time, up-to-date information on any topic. "
                    f"Today is {today}. The event in the question may have ALREADY OCCURRED — "
                    "always check for the latest news or confirmed outcomes before assuming the event is in the future. "
                    "Use precise, specific queries that include the key entities, time period, and outcome keywords "
                    "(e.g., 'result', 'outcome', 'confirmed', 'announced', 'published', 'winner', 'decided'). "
                    "If the first search is inconclusive, try a second query from a different angle. "
                    "Trusted sources (Reuters, AP, BBC, official institutions) are marked as PRIMARY SOURCE."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                f"A specific web search query. Include the key entities, year ({today[:4]}), "
                                "and outcome keywords. Examples: "
                                "'Portugal Croatia 2026 World Cup result', "
                                "'Fed interest rate decision June 2026 outcome', "
                                "'OpenAI GPT-5 release date confirmed', "
                                "'Vietnam GDP growth 2026 official report'."
                            )
                        },
                    },
                    "required": ["query"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_internal_docs",
                "description": (
                    "Search custom uploaded documents, files, news URLs, and historical forecasting context "
                    "provided for this scenario. If the user has supplied custom RAG sources or files, "
                    "this tool retrieves their parsed contents. ALWAYS use this first to look up specific "
                    "news URLs or documents provided by the user."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query to match against custom documents/URLs"},
                    },
                    "required": ["query"],
                },
            }
        }
    ]
    
    has_custom = bool(rag_urls or rag_files)
    custom_instruction = ""
    if has_custom:
        custom_instruction = (
            "\n3. CRITICAL: The user has provided custom RAG files or news URLs for this query. "
            "You MUST call `search_internal_docs` using a specific search query to retrieve the contents of these "
            "custom uploaded documents and URLs. Prioritize the custom documents/URLs over generic web search."
        )

    messages = [
        {"role": "system", "content": (
            f"You are a Research Agent. Today's date is {today}. "
            "Your job is to find the most recent, accurate information relevant to the forecasting question below.\n\n"
            "INSTRUCTIONS:\n"
            f"1. The question may refer to an event that has ALREADY HAPPENED as of {today}. "
            "Do NOT assume the event is in the future — search for the latest outcome first.\n"
            f"2. Use `search_web` with specific, targeted queries.{custom_instruction}\n"
            "4. Write a concise 'Research Summary' with:\n"
            "   - Whether the event has already occurred and what the confirmed outcome was\n"
            "   - Key facts and data points relevant to estimating the probability\n"
            "   - Any relevant historical context or base rates\n"
            "   - Uncertainty level and data quality notes\n"
            "If the event has already resolved, state the actual outcome CLEARLY and PROMINENTLY."
        )},
        {"role": "user", "content": f"Research this forecasting question and summarize all relevant facts:\n\n{question_text}"}
    ]
    
    max_turns = 4  # enough for 2 web searches + internal docs + final summary
    for _ in range(max_turns):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=0.2
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        if not message.tool_calls:
            return message.content
            
        for tool_call in message.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}
                
            func_name = tool_call.function.name
            if func_name in TOOLS_REGISTRY:
                func = TOOLS_REGISTRY[func_name]
                import asyncio
                if asyncio.iscoroutinefunction(func):
                    result = await func(**args)
                else:
                    result = func(**args)
            else:
                result = f"Error: Tool {func_name} not found."
                
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
            
    return "Research limits reached. Gathered partial information."

def execute_research_phase_sync(question_text: str, model: str, api_key: str = None, base_url: str = None, rag_urls=None, rag_files=None) -> str:
    """
    Run a sync multi-turn research phase using OpenAI's tool calling natively.
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(execute_research_phase_async(question_text, model, api_key, base_url, rag_urls, rag_files))
