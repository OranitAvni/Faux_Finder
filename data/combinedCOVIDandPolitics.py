import pandas as pd

# שלב 1: קריאה ודגימה מה-covidSelfDataset
df1 = pd.read_csv("covidSelfDataset.csv")
real_df1 = df1[df1['outcome'] == 'real'].sample(n=7000, random_state=42)
fake_df1 = df1[df1['outcome'] == 'fake'].sample(n=7000, random_state=42)
real_df1['label'] = 1
fake_df1['label'] = 0
real_df1['source'] = 'COVID'
fake_df1['source'] = 'COVID'
df1_clean = pd.concat([real_df1, fake_df1])[['text', 'label', 'source']].sample(frac=1, random_state=42).reset_index(drop=True)

# שלב 2: קריאה ודגימה מה-combined_dataset
df2 = pd.read_csv("combined_dataset_fake&real_cleaned.csv")
real_df2 = df2[df2['label'] == 1].sample(n=7000, random_state=42)
fake_df2 = df2[df2['label'] == 0].sample(n=7000, random_state=42)
real_df2['source'] = 'politics'
fake_df2['source'] = 'politics'
df2_clean = pd.concat([real_df2, fake_df2])[['text', 'label', 'source']].sample(frac=1, random_state=42).reset_index(drop=True)

# שלב 3: איחוד שני הדאטהסטים
combined_df = pd.concat([df1_clean, df2_clean]).sample(frac=1, random_state=42).reset_index(drop=True)

# בדיקה
print(combined_df['source'].value_counts())
print(combined_df['label'].value_counts())
print(combined_df.head())

# שלב 4: שמירה
combined_df.to_csv("combined_balanced_dataset_with_source.csv", index=False)
