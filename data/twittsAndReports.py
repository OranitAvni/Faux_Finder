import pandas as pd
from sklearn.utils import shuffle

# === Load datasets ===
tweets_eval_df = pd.read_csv("evaluation_dataset.csv")  # columns: 'text', 'label'
tweets_other_df = pd.read_csv("ObTr1.csv", encoding="cp1255")
articles_df = pd.read_csv("balanced_10k_dataset_politics.csv")  # columns: 'text', 'label'

# === Standardize tweet dataset column names ===
tweets_other_df = tweets_other_df.rename(columns={"data_tweets": "text", "data_labels": "label"})

# === Combine tweet datasets and add source column ===
tweets_eval_df["source"] = "tweet"
tweets_other_df["source"] = "tweet"
all_tweets_df = pd.concat([tweets_eval_df, tweets_other_df], ignore_index=True)

# === Balance tweet data ===
tweets_0 = all_tweets_df[all_tweets_df["label"] == 0]
tweets_1 = all_tweets_df[all_tweets_df["label"] == 1]

train_tweets = pd.concat([tweets_0.iloc[:350], tweets_1.iloc[:350]], ignore_index=True)
eval_tweets = pd.concat([tweets_0.iloc[350:400], tweets_1.iloc[350:400]], ignore_index=True)

# === Balance article data and add source column ===
articles_df["source"] = "article"
articles_0 = articles_df[articles_df["label"] == 0]
articles_1 = articles_df[articles_df["label"] == 1]

train_articles = pd.concat([articles_0.iloc[:750], articles_1.iloc[:750]], ignore_index=True)
eval_articles = pd.concat([articles_0.iloc[750:800], articles_1.iloc[750:800]], ignore_index=True)

# === Combine train and eval datasets ===
train_df = pd.concat([train_tweets, train_articles], ignore_index=True)
eval_df = pd.concat([eval_tweets, eval_articles], ignore_index=True)

# === Shuffle datasets ===
train_df = shuffle(train_df, random_state=42).reset_index(drop=True)
eval_df = shuffle(eval_df, random_state=42).reset_index(drop=True)

# === Save final CSVs ===
train_df.to_csv("final_train_dataset.csv", index=False)
eval_df.to_csv("final_eval_dataset.csv", index=False)

print("✅ Datasets created successfully with source column included.")
print(f"Train size: {len(train_df)}, Eval size: {len(eval_df)}")
