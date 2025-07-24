import pandas as pd

# Step 1: Load the original real and fake datasets
real_df = pd.read_csv("real.csv")
fake_df = pd.read_csv("fake.csv")

# Step 2: Add label column
real_df["label"] = 1
fake_df["label"] = 0

# Step 3: Merge title and text into one field
real_df["text"] = real_df["title"].astype(str).str.strip() + " " + real_df["text"].astype(str).str.strip()
fake_df["text"] = fake_df["title"].astype(str).str.strip() + " " + fake_df["text"].astype(str).str.strip()

# Step 4: Keep only necessary columns (text, label, subject, date)
real_df = real_df[["text", "label", "subject", "date"]]
fake_df = fake_df[["text", "label", "subject", "date"]]

# Step 5: Combine real and fake datasets
combined_df = pd.concat([real_df, fake_df], ignore_index=True)

# Step 6: Drop rows where text is missing or empty
combined_df = combined_df.dropna(subset=["text"])
combined_df["text"] = combined_df["text"].astype(str).str.strip()
combined_df = combined_df[combined_df["text"] != ""]

# Step 7: Drop rows with missing label
combined_df = combined_df.dropna(subset=["label"])
combined_df["label"] = combined_df["label"].astype(int)

# Step 8: Save the cleaned dataset
combined_df.to_csv("full_dataset_kaggle_40000.csv", index=False, encoding="utf-8")

print("✅ Successfully created 'combined_dataset_fake&real_cleaned.csv' with", len(combined_df), "rows.")
