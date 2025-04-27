import pandas as pd

# קריאה לקובץ
df = pd.read_csv("WELFake_Dataset.csv", encoding="latin1")

# וידוא שיש רק עמודות טקסט ולייבל
df = df[['text', 'label']]

# בדיקה - מה הערכים הקיימים
print(df['label'].value_counts())

# הפרדה של אמת ושקר לפי מספרים
real_df = df[df['label'] == 1]
fake_df = df[df['label'] == 0]

# בדיקה שיש מספיק
if len(real_df) < 2500 or len(fake_df) < 2500:
    raise ValueError("אין מספיק דוגמאות של אמת או שקר בדאטהסט!")

# בחירת 2500 דוגמאות מכל סוג
sample_real = real_df.sample(2500, random_state=42)
sample_fake = fake_df.sample(2500, random_state=42)

# חיבור הדוגמאות וערבוב
sample_df = pd.concat([sample_real, sample_fake]).sample(frac=1, random_state=42)

# שמירת הדאטהסט החדש
sample_df.to_csv("small_dataset.csv", index=False, encoding="utf-8")

# מחיקת הדוגמאות מהדאטהסט המקורי
remaining_df = df.drop(sample_real.index).drop(sample_fake.index)

# שמירת הדאטהסט המקורי אחרי המחיקה
remaining_df.to_csv("WELFake_Dataset_remaining.csv", index=False, encoding="utf-8")

print("הכל בוצע בהצלחה! נוצר קובץ small_dataset.csv עם 5000 דוגמאות, וקובץ WELFake_Dataset_remaining.csv עם השאר.")
