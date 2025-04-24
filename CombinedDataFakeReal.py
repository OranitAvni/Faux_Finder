import pandas as pd
# 21418 rows
real_df = pd.read_csv("real.csv")
# 23504 rows
fake_df = pd.read_csv("fake.csv")

# adding label colum
real_df["label"] = 1
fake_df["label"] = 0

# make title+text -> text
real_df["text"] = real_df["title"].astype(str).str.strip() + " " + real_df["text"].astype(str).str.strip()
fake_df["text"] = fake_df["title"].astype(str).str.strip() + " " + fake_df["text"].astype(str).str.strip()

combined_df = pd.concat([real_df, fake_df], ignore_index=True)

combined_df = combined_df[["text", "label", "subject", "date"]]

combined_df.to_csv("combined_dataset_fake&real.csv", index=False)
# created combined dataset with 23460 fake+ 21417 real =  44877 rows
print("✅ combined_dataset_ready.csv created!")
