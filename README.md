# FauxFinder: Dual-Model Fake News Detection

This project combines two distinct machine learning approaches to detect fake news: a traditional **Convolutional Neural Network (CNN)** and a transformer-based model, **DistilBERT**. The system provides an explainable, ensemble-based credibility score that leverages the strengths of both architectures.

---

## 📊 Dataset Overview

Each entry is labeled as:

* `0` → **Fake**
* `1` → **Real**
  
 split 80% for train 20% for test
---

## 📊 Model Comparison

| Feature           | CNN                       | DistilBERT                  |
| ----------------- | ------------------------- | --------------------------- |
| Architecture Type | Convolutional             | Transformer (Pretrained)    |
| Input Format      | Word vectors / embeddings | Tokenized sentences         |
| Strengths         | Fast, lightweight         | High accuracy, deep context |
| Weaknesses        | Limited context window    | Requires more computation   |

---

## 🤷 How Ensemble Works

The final prediction score is the **average of both models' probability outputs**. This helps mitigate the weaknesses of each model and yields a more balanced and generalizable credibility score.

### ✅ Example:

```
CNN:        P(real) = 0.61
DistilBERT: P(real) = 0.82
Final score = (0.61 + 0.82) / 2 = 0.715
```

---

## 🎓 Use Cases

* Academic comparison of ML vs. transformer approaches
* Fake news detection across multiple domains
* Scalable, lightweight detection via CNN
* Deep, contextual analysis via DistilBERT

---

## 🚀 Try It Out

Launch the Streamlit app:

```bash
streamlit run app_streamlit.py
```

Type or paste any news headline or tweet, and get an instant credibility score powered by both models.

---


## 📓 Credits

Developed by Oranit Avni & Talia Bar Zohar
