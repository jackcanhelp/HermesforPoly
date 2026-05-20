"""Deterministic keyword classifier for Polymarket questions.

Polymarket's Gamma API frequently returns an empty / "Unknown" category. That
silently breaks the two feedback loops that are actually wired into decisions:
per-category lesson retrieval (agent.get_lessons) and per-category calibration
(tracker.get_calibration_stats -> the Judge's bias-correction block). Both are
keyed on market_category, so when everything is "Unknown" they collapse into a
single undifferentiated bucket (or, for real categories, return nothing).

This module assigns a stable, meaningful category from the question text so
those loops fire per-topic. It is intentionally transparent and rule-based
(no LLM call) so it is fast, free, and reproducible inside the scan loop.
"""

import re

# Priority order matters: the first category whose pattern matches wins.
# Keywords are matched on word boundaries (\b) to avoid false hits like
# "oil" in "spoiler" or "war" in "toward".
_RULES = [
    ("Crypto", [
        "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto",
        "altcoin", "token", "fdv", "market cap", "stablecoin", "memecoin",
        "dogecoin", "doge", "xrp", "binance", "coinbase", "satoshi", "megaeth",
        "airdrop", "halving", "defi", "nft", "blockchain",
    ]),
    ("Tech/AI", [
        "openai", "anthropic", "gpt", "claude", "gemini", "llm", "ai model",
        "best ai", "nvidia", "apple", "tesla", "google", "microsoft", "meta",
        "chatgpt", "grok", "deepseek", "agi", "spacex", "starship",
    ]),
    ("Economy", [
        "fed", "federal reserve", "interest rate", "rate cut", "rate hike",
        "cpi", "inflation", "recession", "gdp", "wti", "crude oil", "oil",
        "s&p", "nasdaq", "dow", "jobs report", "unemployment", "treasury",
        "gold", "commodity", "tariff",
    ]),
    ("Geopolitics", [
        "iran", "iranian", "israel", "hezbollah", "hamas", "gaza", "ukraine", "russia",
        "putin", "china", "taiwan", "north korea", "hormuz", "uae", "saudi",
        "ceasefire", "peace deal", "war", "military", "missile", "strike",
        "nato", "nuclear", "sanction", "invade", "troops", "conflict",
        "blockade", "airstrike",
    ]),
    ("Politics", [
        "trump", "biden", "election", "president", "senate", "congress",
        "parliament", "vote", "poll", "polls", "governor", "prime minister",
        "win the most", "republican", "democrat", "impeach", "cabinet",
        "nominee", "referendum", "coup", "resign", "approval rating",
    ]),
    ("Sports", [
        "fc", "win on", "match", "league", "nba", "nfl", "mlb", "premier",
        "champions", "world cup", "playoff", "super bowl", "uefa", "psg",
        "paris saint-germain", "real madrid", "barcelona", "vs", "tournament",
        "grand slam", "formula 1", "f1",
    ]),
    ("Entertainment", [
        "movie", "box office", "oscar", "grammy", "album", "spotify",
        "rotten tomatoes", "imdb", "netflix", "celebrity", "billboard",
        "award", "song", "eurovision",
    ]),
]

# Pre-compile one alternation regex per category.
_COMPILED = [
    (cat, re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\b", re.I))
    for cat, kws in _RULES
]

# Values we treat as "no real category" coming from the API.
_EMPTY = {"", "unknown", "none", "null"}


def classify_market(question, api_category=None):
    """Return a stable, meaningful category for a market question.

    Precedence:
      1. A non-empty category supplied by the API (trust the source if present).
      2. Keyword inference from the question text.
      3. "Other" as a last resort -- never "Unknown", so downstream
         category-keyed loops always group on a real bucket.
    """
    if api_category is not None and str(api_category).strip().lower() not in _EMPTY:
        return str(api_category).strip()

    q = question or ""
    for cat, pattern in _COMPILED:
        if pattern.search(q):
            return cat
    return "Other"


if __name__ == "__main__":
    # Quick smoke test
    samples = [
        "Will Bitcoin reach $80,000 in April?",
        "Will OpenAI have the best AI model at the end of April?",
        "Will WTI Crude Oil (WTI) hit (HIGH) $120 in April?",
        "US x Iran permanent peace deal by April 22, 2026?",
        "Will the All India Trinamool Congress (AITC) win the most seats?",
        "Will Paris Saint-Germain FC win on 2026-04-28?",
        "Some totally unrelated question about the weather",
    ]
    for s in samples:
        print(f"{classify_market(s):14s} <- {s}")
