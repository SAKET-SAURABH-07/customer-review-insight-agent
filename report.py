"""
report.py - Aggregates reviews, detects topic complaint spikes, and exports insights.json.

Pipeline:
1. Ingest reviews from CSV (or SQLite database via --from-db).
2. Tag app-specific operational software topics.
3. Group by calendar week (or review batch if dates are absent).
4. Detect complaint surges using an explainable z-score anomaly heuristic.
5. Export clean results to dashboard/data/insights.json and dashboard/data/insights.js.
   (Injected test surges write to data/test_spike_insights.json, never touching dashboard/data).
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np

from data_loader import load_dataset
from topics import add_topics_to_df, TOPIC_KEYWORDS


def is_clean_review(text):
    """Filters out gibberish, spam, and non-alphabetic symbol dumps."""
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 15 or len(text) > 300:
        return False
    alpha_chars = sum(c.isalpha() or c.isspace() for c in text)
    return (alpha_chars / len(text)) >= 0.65


def detect_spikes(weekly_topic_negatives, threshold_z=1.3):
    """
    Detects sudden negative complaint surges across topics.
    Rule: A spike is flagged if the latest period's negative count is >= 3
    and its z-score compared to past periods exceeds threshold (or surge_ratio >= 1.5).
    """
    alerts = []
    for topic in TOPIC_KEYWORDS:
        if topic not in weekly_topic_negatives.columns:
            continue
        series = weekly_topic_negatives[topic].fillna(0)
        if len(series) < 2:
            continue
        history = series.iloc[:-1]
        latest = series.iloc[-1]
        latest_period = series.index[-1]

        mean = float(history.mean())
        std = float(history.std()) if len(history) > 1 else 0.0

        z_score = (latest - mean) / std if std > 0 else (latest - mean)
        surge_ratio = (latest / mean) if mean > 0 else float(latest)

        if latest >= 3 and (z_score >= threshold_z or surge_ratio >= 1.5):
            alerts.append({
                "topic": topic,
                "period": str(latest_period),
                "count": int(latest),
                "baseline_mean": round(mean, 1),
                "z_score": round(float(z_score), 2),
                "message": f"Critical spike in '{topic}': {int(latest)} complaints in {latest_period} (baseline avg: {mean:.1f})"
            })
    return alerts


def load_from_sqlite(db_path=os.path.join("data", "reviews.db")):
    """Reads reviews from SQLite table populated by collector.py."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite database not found at {db_path}. Run collector.py first.")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT text, label, date FROM reviews", conn)
    conn.close()
    print(f"[SQLite] Ingested {len(df)} records from {db_path}.")
    return df


def build_insights(from_db=False, db_path=None, inject_spike_topic=None, allow_sample=True):
    # 1. Ingest from SQLite or CSV
    if from_db:
        path = db_path if db_path else os.path.join("data", "reviews.db")
        df = load_from_sqlite(path)
    else:
        df = load_dataset()

    # Safety check: Prevent overwriting full metrics with small sample files
    if not allow_sample and len(df) < 500:
        raise ValueError(f"Safety check failed: dataset has only {len(df)} rows (< 500). Aborting overwrite.")

    df = add_topics_to_df(df)

    # 2. Date Detection: verify if calendar dates exist
    has_real_dates = False
    data_period = "Historical Archive"
    if "date" in df.columns:
        parsed_dates = pd.to_datetime(df["date"], errors="coerce")
        if parsed_dates.notna().sum() > 10 and parsed_dates.nunique() > 5:
            df["period_dt"] = parsed_dates
            df["period"] = df["period_dt"].dt.to_period("W").astype(str)
            has_real_dates = True
            min_str = parsed_dates.min().strftime("%b %Y")
            max_str = parsed_dates.max().strftime("%b %Y")
            data_period = f"{min_str} – {max_str} (Historical Archive)"

    if not has_real_dates:
        batch_size = max(10, len(df) // 8)
        df["period"] = [f"Batch {(i // batch_size) + 1}" for i in range(len(df))]
        data_period = "Synthetic Review Batch"

    trend_title = "Weekly Sentiment Trend" if has_real_dates else "Sentiment Trend (By Review Batch)"
    trend_sub = "Positive vs. Negative sentiment across calendar weeks" if has_real_dates else "Positive vs. Negative sentiment across review batches"

    # 3. Optional Spike Injection for Testing (Writes only to temp path!)
    if inject_spike_topic and inject_spike_topic in TOPIC_KEYWORDS:
        print(f"[TEST] Injecting synthetic complaint surge for '{inject_spike_topic}' into latest period...")
        latest_p = df["period"].iloc[-1]
        synthetic_rows = []
        for i in range(45):
            synthetic_rows.append({
                "text": f"INJECTED TEST ALERT: {inject_spike_topic} failure test case #{i}",
                "label": "negative",
                "topics": [inject_spike_topic],
                "primary_topic": inject_spike_topic,
                "period": latest_p,
                "date": str(df.get("date", pd.Series(["2021-03-05"] * len(df))).iloc[-1])
            })
        df = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True)

    # 4. KPIs
    total_reviews = len(df)
    counts = df["label"].value_counts()
    pos_count = int(counts.get("positive", 0))
    neu_count = int(counts.get("neutral", 0))
    neg_count = int(counts.get("negative", 0))
    pos_pct = round((pos_count / total_reviews) * 100, 1) if total_reviews else 0
    neg_pct = round((neg_count / total_reviews) * 100, 1) if total_reviews else 0
    neu_pct = round((neu_count / total_reviews) * 100, 1) if total_reviews else 0

    # 5. Topic summary
    topic_summary = []
    for topic in list(TOPIC_KEYWORDS.keys()) + ["other"]:
        sub = df[df["topics"].apply(lambda ts: topic in ts)]
        t_total = len(sub)
        if t_total > 0:
            t_counts = sub["label"].value_counts()
            topic_summary.append({
                "topic": topic,
                "total": t_total,
                "positive": int(t_counts.get("positive", 0)),
                "neutral": int(t_counts.get("neutral", 0)),
                "negative": int(t_counts.get("negative", 0)),
                "neg_rate": round((t_counts.get("negative", 0) / t_total) * 100, 1)
            })
    # Sort by negative complaint count descending
    topic_summary.sort(key=lambda x: x["negative"], reverse=True)
    top_complaint = topic_summary[0]["topic"] if topic_summary else "N/A"

    # 6. Periodic trend
    period_group = df.groupby(["period", "label"]).size().unstack(fill_value=0)
    for col in ["positive", "neutral", "negative"]:
        if col not in period_group.columns:
            period_group[col] = 0

    period_group = period_group.tail(8)
    period_trends = []
    for p_label, row in period_group.iterrows():
        p_total = int(row.sum())
        period_trends.append({
            "period": str(p_label),
            "total": p_total,
            "positive": int(row["positive"]),
            "neutral": int(row["neutral"]),
            "negative": int(row["negative"]),
            "pos_pct": round((row["positive"] / p_total) * 100, 1) if p_total else 0,
            "neg_pct": round((row["negative"] / p_total) * 100, 1) if p_total else 0
        })

    # 7. Spike Detection
    neg_subset = df[df["label"] == "negative"]
    period_topic_neg = neg_subset.groupby(["period", "primary_topic"]).size().unstack(fill_value=0)
    alerts = detect_spikes(period_topic_neg)

    # 8. Clean example reviews (Strictly truncated to max 100 chars)
    clean_reviews = []
    clean_pool = df[df["text"].apply(is_clean_review)]
    sample_size = min(len(clean_pool), 200)
    if sample_size > 0:
        for _, row in clean_pool.sample(sample_size, random_state=42).iterrows():
            raw_text = str(row["text"]).replace("\n", " ").strip()
            truncated = raw_text[:99] + "…" if len(raw_text) > 100 else raw_text
            clean_reviews.append({
                "text": truncated,
                "label": row["label"],
                "topic": row["primary_topic"],
                "date": str(row.get("date", "N/A"))
            })
            if len(clean_reviews) >= 30:
                break

    summary_text = (
        f"Analyzed {total_reviews:,} customer reviews ({data_period}). Sentiment stands at "
        f"{pos_pct:.1f}% positive and {neg_pct:.1f}% negative. Primary complaint volume driven by "
        f"'{top_complaint}'. Status: {len(alerts)} active topic surge alert(s)."
    )

    payload = {
        "generated_at": "2026-10-31T19:00:00Z",
        "data_period": data_period,
        "has_real_dates": has_real_dates,
        "trend_title": trend_title,
        "trend_sub": trend_sub,
        "kpis": {
            "total_reviews": total_reviews,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "neutral_pct": neu_pct,
            "active_alerts": len(alerts)
        },
        "executive_summary": summary_text,
        "alerts": alerts,
        "weekly_trends": period_trends,
        "topic_summary": topic_summary,
        "sample_reviews": clean_reviews
    }

    # 9. Output Path Destination Rule:
    # If this is a spike injection test, write to a temporary file, NEVER to dashboard/data.
    if inject_spike_topic:
        temp_path = os.path.join("data", "test_spike_insights.json")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[TEST] Injected spike output written to temporary path: {temp_path} (dashboard/data was NOT modified).")
        print(f"Total reviews: {total_reviews:,} | Active Alerts: {len(alerts)}")
        for a in alerts:
            print(f"  --> ALERT: {a['message']} (z-score: {a['z_score']})")
    else:
        out_dir = os.path.join("dashboard", "data")
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, "insights.json")
        js_path = os.path.join(out_dir, "insights.js")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(js_path, "w", encoding="utf-8") as f:
            f.write("window.INSIGHTS_DATA = " + json.dumps(payload, indent=2) + ";\n")

        print(f"Generated clean dashboard data at {json_path}")
        print(f"Coverage: {data_period} | Total reviews: {total_reviews:,} | Active Alerts: {len(alerts)}")

    return payload


if __name__ == "__main__":
    from_db_flag = "--from-db" in sys.argv
    db_file = None
    if "--db" in sys.argv:
        idx = sys.argv.index("--db")
        if idx + 1 < len(sys.argv):
            from_db_flag = True
            db_file = sys.argv[idx + 1]

    inject = None
    if "--inject-spike" in sys.argv:
        idx = sys.argv.index("--inject-spike")
        inject = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "login/sync"

    build_insights(from_db=from_db_flag, db_path=db_file, inject_spike_topic=inject)
