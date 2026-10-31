"""
Creates a small DEMO dataset (data/reviews.csv) so you can run the pipeline
end to end before you have real data.

NOTE: these reviews are generated from templates, so the model will score
unrealistically high on them. Replace data/reviews.csv with a real review
dataset (columns: text,label) for honest accuracy numbers.
"""
import random
import pandas as pd

random.seed(42)

TOPICS = {
    "delivery": {
        "pos": ["Delivery was super fast", "Arrived a day early", "Packaging was neat and delivery on time"],
        "neg": ["Delivery was very late", "Package arrived damaged", "Courier never showed up and delivery took ages"],
    },
    "quality": {
        "pos": ["Great build quality", "The product feels premium", "Works perfectly and looks durable"],
        "neg": ["Poor quality, broke in a week", "Feels cheap and flimsy", "Stopped working after two days"],
    },
    "price": {
        "pos": ["Totally worth the price", "Great value for money", "Very affordable and good"],
        "neg": ["Way too expensive for what you get", "Overpriced, not worth it", "Not good value for money"],
    },
    "support": {
        "pos": ["Support team was helpful", "Customer service solved my issue quickly", "Got a quick and polite reply"],
        "neg": ["Support never replied", "Customer service was rude and useless", "Waited a week for a response"],
    },
}
NEUTRAL = [
    "It is okay, nothing special", "Average product, does the job",
    "Neither good nor bad", "Decent but I expected a bit more",
]
ENDINGS = ["", "", " overall.", " honestly.", ", just my experience."]

rows = []
for _ in range(400):
    topic = random.choice(list(TOPICS))
    kind = random.choices(["pos", "neg", "neu"], weights=[40, 40, 20])[0]
    if kind == "neu":
        text, label = random.choice(NEUTRAL), "neutral"
    else:
        text = random.choice(TOPICS[topic][kind])
        label = "positive" if kind == "pos" else "negative"
    rows.append({"text": text + random.choice(ENDINGS), "label": label})

import os
os.makedirs("data", exist_ok=True)
pd.DataFrame(rows).to_csv("data/reviews.csv", index=False)
print("Wrote data/reviews.csv with", len(rows), "reviews")
