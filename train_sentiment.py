"""
train_sentiment.py - Train an honest sentiment classifier on customer reviews.

Pipeline: text -> TF-IDF (unigrams + bigrams) -> Logistic Regression
Why TF-IDF + Logistic Regression?
It is lightweight, fast to train (<2 seconds), fully explainable (interpretable
word weights), and serves as an ideal production baseline before trying heavy LLMs.

Input:  real customer review dataset (or fallback to demo data)
Output: models/sentiment.joblib + printed metrics, confusion matrix & error analysis
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_loader import load_dataset

# Safe Unicode output for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else None
MODEL_PATH = "models/sentiment.joblib"

# 1. Load data cleanly through flexible loader
df = load_dataset(DATA_PATH)
print(f"Loaded {len(df)} reviews")
print("Class breakdown:\n", df["label"].value_counts().to_string(), "\n")

# 2. Stratified train/test split: ensures minority classes are equally represented
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 3. Model Pipeline:
# - ngram_range=(1,2) captures phrases like "keeps crashing" or "not working"
# - min_df=2 filters out one-off typos
# - class_weight='balanced' prevents majority class bias
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=2)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

# 4. Honest Evaluation
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"=== Honest Test Accuracy: {acc * 100:.2f}% ===\n")
print("=== Classification Report ===")
print(classification_report(y_test, pred))

# 5. Confusion Matrix
labels = ["negative", "neutral", "positive"]
cm = confusion_matrix(y_test, pred, labels=labels)
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{lbl}" for lbl in labels],
    columns=[f"Pred_{lbl}" for lbl in labels],
)
print("=== Confusion Matrix ===")
print(cm_df.to_string(), "\n")

# 6. Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print("Saved model to", MODEL_PATH, "\n")

# 7. Error Analysis: 5 real mistakes to inspect in interview
print("=== 5 Example Test Set Mistakes ===")
test_records = pd.DataFrame({"text": X_test, "actual": y_test, "predicted": pred})
mistakes = test_records[test_records["actual"] != test_records["predicted"]]
sample_mistakes = mistakes.head(5)

for i, (_, row) in enumerate(sample_mistakes.iterrows(), 1):
    cleaned_text = row["text"].replace("\n", " ").strip()
    if len(cleaned_text) > 100:
        cleaned_text = cleaned_text[:97] + "..."
    print(f"{i}. Text: \"{cleaned_text}\"")
    print(f"   Actual: {row['actual']} | Predicted: {row['predicted']}\n")

# 8. Check the known problem case
print("=== Test Known Cases ===")
cases = [
    "The app keeps crashing and support ignores me",
    "Fast delivery and great quality, love it",
    "It is fine, nothing special",
]
for text, p_lbl in zip(cases, model.predict(cases)):
    print(f"  [{p_lbl:8}] {text}")
