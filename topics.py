"""
topics.py - App-relevant keyword topic tagger for customer reviews.

Tags each review with one or more operational software topics:
- crashes/bugs   : app failures, errors, blank screens
- updates        : install, version, download issues
- speed/performance : lag, slow, memory, loading times
- login/sync     : accounts, bookmarks, tabs, history
- ads            : pop-ups, banners, spam
- battery/data   : drain, wifi, network, pages
- privacy        : tracking, incognito, permissions

Why keyword rules?
They are deterministic, instantly auditable, and map directly to engineering
backlogs without needing model training. Generic praise words (good, nice, ok)
are deliberately excluded — they carry no actionable topic signal.
"""

import re
import pandas as pd

# --- Topic keyword dictionaries (app-specific) ---
# Only words that point to a *specific operational area*.
# Generic sentiment words (good, nice, best, love) are NOT topics.

TOPIC_KEYWORDS = {
    "crashes/bugs": [
        r"\bcrash\b", r"\bcrashing\b", r"\bcrashes\b", r"\bcrashed\b", r"\bbug\b", r"\bbugs\b", r"\bbuggy\b",
        r"\bglitch\b", r"\bglitches\b", r"\bfreeze\b", r"\bfreezes\b", r"\bfreezing\b", r"\bfroze\b", r"\bfrozen\b",
        r"\berror\b", r"\berrors\b", r"\bforce close\b", r"\bstopped\b", r"\bhang\b", r"\bhanging\b", r"\bhangs\b",
        r"\bnot working\b", r"\bdoesn'?t work\b", r"\bwon'?t work\b", r"\bdid not work\b", r"\bnot work\b",
        r"\bbroken\b", r"\bfailed\b", r"\bfailing\b", r"\bfailure\b", r"\bproblem\b", r"\bproblems\b",
        r"\bissue\b", r"\bissues\b", r"\bfix\b", r"\bfixes\b", r"\bfixing\b", r"\bnot open\b", r"\bwon'?t open\b",
        r"\bcan'?t open\b", r"\bdoesn'?t open\b", r"\bunable to open\b", r"\bblank\b", r"\bblack screen\b",
        r"\bwhite screen\b", r"\bauto close\b", r"\bcloses\b", r"\bclosing\b", r"\bshutting\b", r"\bshut down\b",
        r"\btrouble\b"
    ],
    "updates": [
        r"\bupdate\b", r"\bupdates\b", r"\bupdated\b", r"\bupdating\b", r"\bupdation\b",
        r"\bversion\b", r"\bversions\b",
        r"\bdownload\b", r"\bdownloads\b", r"\bdownloading\b", r"\bdownloaded\b",
        r"\binstall\b", r"\binstalls\b", r"\binstalling\b", r"\binstalled\b", r"\binstallation\b",
        r"\buninstall\b", r"\buninstalling\b", r"\buninstalled\b",
        r"\bupgrade\b", r"\bupgrading\b", r"\bdowngrade\b",
        r"\bold version\b", r"\bnew version\b", r"\blatest version\b",
        r"\bpending\b", r"\bstuck\b", r"\bplay store\b"
    ],
    "speed/performance": [
        # Actual performance vocabulary only — no generic praise
        r"\bslow\b", r"\bslower\b", r"\bslowest\b", r"\bspeed\b", r"\bspeedy\b",
        r"\bfast\b", r"\bfaster\b", r"\bfastest\b",
        r"\blag\b", r"\blags\b", r"\blagging\b", r"\blaggy\b", r"\bsluggish\b",
        r"\bdelay\b", r"\bdelays\b", r"\bdelayed\b",
        r"\bperformance\b", r"\bloading\b", r"\bloads\b",
        r"\bresponsive\b", r"\bheavy\b", r"\blightweight\b",
        r"\bram\b", r"\bmemory\b", r"\bstorage\b", r"\bsmooth\b", r"\bsmoothly\b",
        r"\bcache\b", r"\bfluid\b"
    ],
    "login/sync": [
        r"\bsync\b", r"\bsyncing\b", r"\bsynced\b", r"\bsynchronize\b", r"\bsynchronizing\b",
        r"\blogin\b", r"\blog in\b", r"\blogged in\b", r"\blogging in\b",
        r"\bsign in\b", r"\bsigned in\b", r"\bsigning in\b",
        r"\baccount\b", r"\baccounts\b", r"\bpassword\b", r"\bpasswords\b",
        r"\bbookmark\b", r"\bbookmarks\b", r"\bprofile\b",
        r"\bgoogle account\b", r"\bgmail\b",
        r"\bhistory\b", r"\btabs\b", r"\btab\b", r"\btab group\b"
    ],
    "ads": [
        r"\bads\b", r"\badvert\b", r"\badverts\b",
        r"\badvertisement\b", r"\badvertisements\b", r"\badvertising\b",
        r"\bpop-?ups?\b", r"\bpopups?\b",
        r"\bspam\b", r"\bbanner\b", r"\bbanners\b"
    ],
    "battery/data": [
        r"\bbattery\b", r"\bdrain\b", r"\bdraining\b", r"\bdrained\b", r"\bdrains\b",
        r"\bpower\b", r"\bcharge\b", r"\bcharging\b",
        r"\bheat\b", r"\bheating\b", r"\boverheat\b",
        r"\bmobile data\b", r"\bwifi\b", r"\bwi-fi\b",
        r"\binternet\b", r"\bnetwork\b", r"\bconnection\b", r"\boffline\b"
    ],
    "privacy": [
        r"\bprivacy\b", r"\bprivate\b",
        r"\btrack\b", r"\btracking\b", r"\btracker\b", r"\btrackers\b",
        r"\bpermission\b", r"\bpermissions\b",
        r"\bsecure\b", r"\bsecurity\b", r"\bincognito\b",
        r"\bspy\b", r"\bspying\b", r"\bspyware\b",
        r"\bsafe browsing\b", r"\bcookies\b"
    ],
}

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
    df["primary_topic"] = df["topics"].apply(lambda ts: ts[0] if ts else "other")
    return df


if __name__ == "__main__":
    from data_loader import load_dataset

    data = load_dataset()
    tagged = add_topics_to_df(data)

    # Show all-reviews distribution
    all_topics = [t for sublist in tagged["topics"] for t in sublist]
    topic_counts = pd.Series(all_topics).value_counts()

    total = len(tagged)
    other_count = sum(1 for ts in tagged["topics"] if ts == ["other"])

    print("=== Topic Distribution (All Reviews) ===")
    print(topic_counts.to_string())
    print(f"\nTotal: {total} | 'other': {other_count} ({other_count / total * 100:.1f}%)\n")

    # Show negative-only distribution (what the dashboard chart uses)
    neg = tagged[tagged["label"] == "negative"]
    neg_topics = [t for sublist in neg["topics"] for t in sublist]
    neg_counts = pd.Series(neg_topics).value_counts()
    neg_other = sum(1 for ts in neg["topics"] if ts == ["other"])

    print("=== Topic Distribution (Negative Reviews Only) ===")
    print(neg_counts.to_string())
    print(f"\nNeg total: {len(neg)} | 'other': {neg_other} ({neg_other / len(neg) * 100:.1f}%)")
