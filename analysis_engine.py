from collections import Counter
from urllib.parse import urlparse
import re


TRUSTED_DOMAINS = [
    "reuters.com",
    "bloomberg.com",
    "moneycontrol.com",
    "economictimes.indiatimes.com",
    "livemint.com",
    "business-standard.com",
    "cnbc.com",
    "wsj.com",
    "ft.com",
]


POSITIVE_WORDS = [
    "growth", "profit", "increase", "gain", "strong", "surge", "bullish",
    "expansion", "record high", "beat expectations", "uptrend"
]

NEGATIVE_WORDS = [
    "loss", "decline", "fall", "weak", "drop", "crash", "bearish",
    "slowdown", "record low", "missed expectations", "downtrend"
]

RISK_WORDS = [
    "may", "might", "could", "expected", "uncertain", "risk", "possibly"
]


def extract_domain(source: str) -> str:
    try:
        parsed = urlparse(source)
        return parsed.netloc.lower()
    except:
        return source.lower()


def calculate_trust_score(text: str, source: str) -> int:
    score = 50
    source_lower = source.lower()
    domain = extract_domain(source)

    # 1. Source reliability
    if any(td in domain for td in TRUSTED_DOMAINS):
        score += 20

    # 2. Numbers / data present
    if re.search(r"\d", text):
        score += 10

    # 3. Sufficient article length
    if len(text) > 700:
        score += 10

    # 4. Has quotes (usually indicates reporting)
    if '"' in text or "'" in text:
        score += 5

    # 5. Too much speculation reduces trust
    speculative_hits = sum(word in text.lower() for word in RISK_WORDS)
    score -= min(speculative_hits * 2, 15)

    return max(0, min(score, 100))


def analyze_impact(text: str) -> str:
    text_lower = text.lower()

    pos = sum(word in text_lower for word in POSITIVE_WORDS)
    neg = sum(word in text_lower for word in NEGATIVE_WORDS)

    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    return "Neutral"


def consensus_analysis(docs) -> dict:
    result = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for doc in docs:
        impact = analyze_impact(doc.page_content)
        result[impact] += 1

    return result


def get_consensus_label(consensus: dict) -> str:
    if consensus["Positive"] > max(consensus["Negative"], consensus["Neutral"]):
        return "Overall Positive"
    elif consensus["Negative"] > max(consensus["Positive"], consensus["Neutral"]):
        return "Overall Negative"
    return "Mixed / Neutral"


def summarize_sources(docs) -> list:
    summary = []

    for doc in docs:
        text = doc.page_content[:1200]
        source = doc.metadata.get("source", "Unknown Source")
        trust = calculate_trust_score(text, source)
        impact = analyze_impact(text)

        summary.append({
            "source": source,
            "trust_score": trust,
            "impact": impact,
            "preview": text[:250] + "..."
        })

    return summary