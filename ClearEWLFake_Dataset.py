import pandas as pd

# Step 1: Load the dataset (replace with actual path if needed)
df = pd.read_csv("WELFake_Dataset.csv", encoding="utf-8", on_bad_lines="skip")

# Step 2: Keep only relevant columns
df = df[["title", "text", "label"]]

# Step 3: Drop rows with missing or empty title/text/label
df = df.dropna(subset=["title", "text", "label"])
df["title"] = df["title"].astype(str).str.strip()
df["text"] = df["text"].astype(str).str.strip()

# Remove rows where title or text is empty or too short
df = df[(df["title"] != "") & (df["text"] != "")]
df = df[df["text"].str.len() > 20]  # Remove "junk" short rows

# Step 4: Combine title and text into one column
df["full_text"] = df["title"] + " " + df["text"]

# Step 5: Clean label column to ensure only 0/1
df["label"] = df["label"].astype(str).str.strip()
df = df[df["label"].isin(["0", "1"])]
df["label"] = df["label"].astype(int)

# Step 6: Keep only what we need for training
df_ready = df[["full_text", "label"]]
df_ready.columns = ["text", "label"]  # Rename for consistency

# Optional: show some rows
print("✅ Cleaned dataset preview:")
print(df_ready.sample(5))
print("Total rows:", len(df_ready))

# Optional: Save cleaned version
df_ready.to_csv("cleaned_WELFake_dataset.csv", index=False)
