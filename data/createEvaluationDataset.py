import pandas as pd

# קריאה של label.txt
labels = []
tweet_ids = []

with open("label.txt", "r", encoding="utf-8") as f:
    for line in f:
        if ':' in line:
            label, tweet_id = line.strip().split(":")
            label = label.strip().lower()
            if label in ["true", "false"]:  # רק true/false
                labels.append(label)
                tweet_ids.append(tweet_id.strip())

# DataFrame לתוויות
df_labels = pd.DataFrame({
    "tweet_id": tweet_ids,
    "label": labels
})

# המרת תוויות ל־0/1
df_labels["label"] = df_labels["label"].map({"false": 0, "true": 1})

# קריאה של source_tweets.txt
tweet_ids_text = []
texts = []

with open("source_tweets.txt", "r", encoding="utf-8") as f:
    for line in f:
        if "\t" in line:
            tweet_id, text = line.strip().split("\t", 1)
            tweet_ids_text.append(tweet_id.strip())
            texts.append(text.strip())

# DataFrame לטקסטים עם שם עמודה תואם
df_texts = pd.DataFrame({
    "tweet_id": tweet_ids_text,
    "text": texts
})

# מיזוג לפי tweet_id
merged_df = pd.merge(df_labels, df_texts, on="tweet_id", how="inner")

# שמירה לקובץ חדש
merged_df.to_csv("dataset_politics_TWITTER.csv", index=False)
print("✅ הקובץ dataset_politics_TWITTER.csv נוצר בהצלחה!")
print(merged_df.head())
