import pandas as pd

# Reading label.txt
labels = []
tweet_ids = []

with open("label.txt", "r", encoding="utf-8") as f:
    for line in f:
        if ':' in line:
            label, tweet_id = line.strip().split(":")
            label = label.strip().lower()
            if label in ["true", "false"]:  # Only true/false
                labels.append(label)
                tweet_ids.append(tweet_id.strip())

# Create DataFrame for labels
df_labels = pd.DataFrame({
    "tweet_id": tweet_ids,
    "label": labels
})

# Convert labels to 0/1
df_labels["label"] = df_labels["label"].map({"false": 0, "true": 1})

# Reading source_tweets.txt
tweet_ids_text = []
texts = []

with open("source_tweets.txt", "r", encoding="utf-8") as f:
    for line in f:
        if "\t" in line:
            tweet_id, text = line.strip().split("\t", 1)
            tweet_ids_text.append(tweet_id.strip())
            texts.append(text.strip())

# Create DataFrame for texts with matching column name
df_texts = pd.DataFrame({
    "tweet_id": tweet_ids_text,
    "text": texts
})

# Merge on tweet_id
merged_df = pd.merge(df_labels, df_texts, on="tweet_id", how="inner")

# Save to new CSV file
merged_df.to_csv("dataset_politics_TWITTER.csv", index=False)
print("✅ File dataset_politics_TWITTER.csv created successfully!")
print(merged_df.head())
