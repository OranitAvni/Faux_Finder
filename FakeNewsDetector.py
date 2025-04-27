from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# קריאה לקובץ ה-CSV ושמירה רק בעמודות הטקסט והתווית
df = pd.read_csv("small_dataset.csv", encoding="latin1")
df = df[['text', 'label']]  # נשארים רק עם טקסט ותווית

# המרת טקסטים ותוויות למחרוזות
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(str)

# המרת תווית ל-0 ו-1
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# חלוקה לנתוני אימון ונתוני בדיקה
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

class FakeNewsDetector(tf.keras.Model):
    def __init__(self):
        super(FakeNewsDetector, self).__init__()
        # טוען את Tokenizer ומודל של RoBERTa
        self.roberta = TFRobertaModel.from_pretrained('roberta-base')
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # הפקת embeddings
        outputs = self.roberta(inputs)[0]  # לקחת את ה-last hidden states
        x = tf.reduce_mean(outputs, axis=1)  # ממוצע על פני כל הטוקנים
        x = self.dense(x)  # סיווג
        return x

# יצירת tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# המרת טקסטים ל-token IDs
train_encodings = tokenizer(list(train_texts.astype(str)), truncation=True, padding=True, max_length=512, return_tensors="tf")
test_encodings = tokenizer(list(test_texts.astype(str)), truncation=True, padding=True, max_length=512, return_tensors="tf")

# יצירת אובייקט של המודל
model = FakeNewsDetector()

# קומפילציה של המודל
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# אימון המודל
model.fit(
    train_encodings['input_ids'],
    train_labels,
    validation_data=(test_encodings['input_ids'], test_labels),
    epochs=3,
    batch_size=16
)

# פונקציה לחיזוי
def predict(text):
    encoding = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="tf")
    prediction = model.predict(encoding['input_ids'])
    return "Fake" if prediction[0][0] > 0.5 else "Real"

# דוגמה לשימוש
sample_text = "The government has announced a new policy to improve education."
result = predict(sample_text)
print(f"The news is: {result}")
