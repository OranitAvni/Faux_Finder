import pandas as pd

# --- Step 1: Load the files ---
df_self = pd.read_csv("covidSelfDataset.csv")           # Includes outcome: 'real' / 'fake'
df_zenodo = pd.read_csv("zenodo_original.csv")          # Includes outcome: 0 / 1 and column 'headlines'

# --- Step 2: Standardize column names and values ---
df_zenodo = df_zenodo.rename(columns={'headlines': 'text'})          # Match column name
df_zenodo['outcome'] = df_zenodo['outcome'].replace({0: 'fake', 1: 'real'})  # Convert to unified values

# --- Step 3: Combine and remove duplicates ---
combined_df = pd.concat([
    df_self[['text', 'outcome']],
    df_zenodo[['text', 'outcome']]
], ignore_index=True)

combined_df = combined_df.drop_duplicates(subset='text')

# --- Step 4: Split and balance ---
df_true = combined_df[combined_df['outcome'] == 'real']
df_fake = combined_df[combined_df['outcome'] == 'fake']

# Balance based on real samples count
df_fake_sampled = df_fake.sample(n=len(df_true), random_state=42)

# --- Step 5: Combine and shuffle ---
balanced_df = pd.concat([df_true, df_fake_sampled])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Step 6: Save files ---
df_true.to_csv("all_real_from_both_sources.csv", index=False)
df_fake_sampled.to_csv("sampled_fake_to_match_real.csv", index=False)
balanced_df.to_csv("balanced_combined_from_self_and_zenodo.csv", index=False)
# comment
print("✅ The following files were created:")
print("• all_real_from_both_sources.csv")
print("• sampled_fake_to_match_real.csv")
print("• balanced_combined_from_self_and_zenodo.csv")
