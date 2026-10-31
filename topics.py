"""
topics.py - Keyword-rule topic tagger for customer reviews.

Tags each review with one or more operational topics:
- delivery
- quality
- price
- support
- app/bugs

Why keyword rules first?
Keyword rules are 100% deterministic, require zero model downloads or latency,
and allow immediate auditing in interviews: any business stakeholder can inspect
and adjust the exact dictionary without retraining.
"""

import re
import pandas as pd

# Core keyword dictionaries for operational buckets
TOPIC_KEYWORDS = {
    "app/bugs": [
        r"\bcrash\b", r"\bcrashing\b", r"\bbug\b", r"\bbugs\b", r"\bfreeze\b",
        r"\bfreezing\b", r"\blag\b", r"\blagging\b", r"\bslow\b", r"\bupdate\b",
        r"\bloading\b", r"\bglitch\b", r"\binstall\b", r"\buninstall\b",
        r"\berror\b", r"\bbattery\b", r"\bblack screen\b", r"\bopen\b", r"\bclose\b"
    ],
    "support": [
        r"\bsupport\b", r"\bcustomer service\b", r"\bhelp desk\b", r"\bagent\b",
        r"\bresponse\b", r"\breply\b", r"\bignored\b", r"\bcontact\b",
        r"\bemail\b", r"\bcall\b", r"\bassistance\b"
    ],
    "delivery": [
        r"\bdelivery\b", r"\bdeliver\b", r"\bshipping\b", r"\bshipped\b",
        r"\bcourier\b", r"\bpackage\b", r"\barrived\b", r"\bdispatch\b",
        r"\blate\b", r"\btracking\b"
    ],
    "quality": [
        r"\bquality\b", r"\bbuild\b", r"\bbroken\b", r"\bbroke\b",
        r"\bdurable\b", r"\bmaterial\b", r"\bflimsy\b", r"\bcheap\b",
        r"\bfinish\b", r"\bdefect\b", r"\bfaulty\b"
    ],
    "price": [
        r"\bprice\b", r"\bexpensive\b", r"\bcost\b", r"\bworth\b",
        r"\bmoney\b", r"\boverpriced\b", r"\bvalue\b", r"\bcharge\b",
        r"\brefund\b", r"\baffordable\b", r"\bfee\b"
    ],
}

# Pre-compile regular expressions for high throughput
COMPILED_PATTERNS = {
    topic: [re.compile(p, re.IGNORECASE) for p in patterns]
    for topic, patterns in TOPIC_KEYWORDS.items()
}


def tag_topics(text):
    """
    Returns a list of matched topics for a given review text string.
    If no operational topic matches, returns ['other'].
    """
    if not isinstance(text, str) or not text.strip():
        return ["other"]

    matched = []
    for topic, patterns in COMPILED_PATTERNS.items():
        if any(p.search(text) for p in patterns):
            matched.append(topic)

    return matched if matched else ["other"]


def add_topics_to_df(df, text_col="text"):
    """
    Applies topic tagging across a DataFrame, adding a 'topics' list column
    and a primary 'primary_topic' column for single-topic aggregations.
    """
    df = df.copy()
    df["topics"] = df[text_col].apply(tag_topics)
    # primary_topic is the first operational topic found, or 'other'
    df["primary_topic"] = df["topics"].apply(lambda ts: ts[0] if ts else "other")
    return df


if __name__ == "__main__":
    from data_loader import load_dataset

    # Test on real dataset
    data = load_dataset()
    tagged = add_topics_to_df(data)

    print("=== Topic Distribution on Dataset ===")
    all_topics = [t for sublist in tagged["topics"] for t in sublist]
    topic_counts = pd.Series(all_topics).value_counts()
    print(topic_counts.to_string(), "\n")

    print("=== Sample Tagged Reviews ===")
    for _, row in tagged.sample(5, random_state=42).iterrows():
        sample_txt = row["text"].replace("\n", " ").strip()
        if len(sample_txt) > 80:
            sample_txt = sample_txt[:77] + "..."
        print(f"[{row['label']:8}] Topics: {row['topics']} | \"{sample_txt}\"")
