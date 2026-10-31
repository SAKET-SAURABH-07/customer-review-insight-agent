"""
data_loader.py - Flexible customer review dataset loader.

Handles real-world datasets from Kaggle, Amazon, Flipkart, or Google Play Store
by automatically detecting review text and star rating columns.

Why auto-detection?
Different platforms use different column names (e.g. 'reviewText' vs 'Text',
'overall' vs 'Star'). Auto-detection lets the same pipeline train on any source
without manual data wrangling.
"""

import os
import re
import pandas as pd

# Common column name candidates across platforms
TEXT_CANDIDATES = [
    "text", "review", "reviewtext", "review_text", "content",
    "comment", "comments", "translated_review", "body"
]
RATING_CANDIDATES = [
    "rating", "star", "stars", "overall", "score", "star_rating"
]
LABEL_CANDIDATES = ["label", "sentiment", "sentiment_category"]
DATE_CANDIDATES = ["date", "review_date", "reviewtime", "created_at", "timestamp"]


def _match_column(columns, candidates):
    """Finds first matching column ignoring case, spaces, and underscores."""
    col_map = {re.sub(r"[\s_.]", "", c.lower()): c for c in columns}
    for cand in candidates:
        cand_key = re.sub(r"[\s_.]", "", cand.lower())
        if cand_key in col_map:
            return col_map[cand_key]
    return None


def star_to_sentiment(value):
    """
    Maps 1-5 star ratings to sentiment labels:
    - 1 to 2 stars -> negative
    - 3 stars      -> neutral
    - 4 to 5 stars -> positive
    Returns None for missing or unparseable ratings.
    """
    if pd.isna(value):
        return None
    try:
        # Extract the first float/int found in case of formats like '4 out of 5'
        match = re.search(r"(\d+(\.\d+)?)", str(value))
        if not match:
            return None
        rating = float(match.group(1))
        if rating <= 2.0:
            return "negative"
        elif rating <= 3.0:
            return "neutral"
        elif rating <= 5.0:
            return "positive"
    except (ValueError, TypeError):
        return None
    return None


def load_dataset(filepath=None):
    """
    Loads and standardizes a review dataset into a DataFrame with
    ['text', 'label'] columns (and optional 'date' if available).
    """
    if filepath is None:
        # Default priority: real data first, fallback to sample/demo data
        if os.path.exists("data/real_reviews.csv"):
            filepath = "data/real_reviews.csv"
        elif os.path.exists("data/sample_reviews.csv"):
            filepath = "data/sample_reviews.csv"
        elif os.path.exists("data/reviews.csv"):
            filepath = "data/reviews.csv"
        else:
            raise FileNotFoundError("No review dataset found in data/")

    df = pd.read_csv(filepath)
    text_col = _match_column(df.columns, TEXT_CANDIDATES)
    rating_col = _match_column(df.columns, RATING_CANDIDATES)
    label_col = _match_column(df.columns, LABEL_CANDIDATES)
    date_col = _match_column(df.columns, DATE_CANDIDATES)

    if not text_col:
        raise ValueError(f"Could not find a review text column in: {list(df.columns)}")

    result = pd.DataFrame()
    result["text"] = df[text_col].astype(str).str.strip()

    # Determine labels from rating or existing label column
    if rating_col:
        result["label"] = df[rating_col].apply(star_to_sentiment)
    elif label_col:
        # Standardize existing text labels
        result["label"] = df[label_col].astype(str).str.lower().str.strip()
    else:
        raise ValueError(f"Could not find rating or label column in: {list(df.columns)}")

    # Preserve review date if present for weekly temporal analysis
    if date_col:
        result["date"] = df[date_col]

    # Drop missing, invalid labels, and empty review strings
    valid_labels = {"positive", "neutral", "negative"}
    result = result[result["label"].isin(valid_labels)]
    result = result[result["text"].str.len() > 0]
    result = result.dropna(subset=["text", "label"]).reset_index(drop=True)

    return result


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_dataset(path)
    print(f"Loaded {len(data)} valid reviews.")
    print("Label distribution:\n", data["label"].value_counts().to_string())
    print("\nFirst 3 rows:\n", data.head(3))
