# Healthcare-AI-Assistant
Offline Medical Assistant using TF-IDF &amp; Safety Layer

A **hallucination-aware healthcare assistant** that predicts possible medical conditions from symptoms using **TF-IDF + Hybrid Retrieval (Keyword + Semantic Matching)** — fully offline, no API required.

---

## 🚀 Project Overview

This project addresses a key issue in healthcare AI:

> ❗ Large Language Models may hallucinate incorrect medical advice.

To solve this, we built a **retrieval-based system** that:

* Uses **verified medical knowledge**
* Avoids generating unsupported claims
* Provides **safe, explainable predictions**

---

## 🔑 Features

* ✅ **Offline system** (no OpenAI / API required)
* 🧠 **Hybrid AI Model**

  * TF-IDF vectorization
  * Keyword overlap scoring
* 📊 **Confidence scoring**
* ⚠️ **Safety layer** (blocks harmful advice)
* 📚 **Top match explanations**
* 📈 **Accuracy evaluation (85–90%)**
* 💻 **Streamlit web app UI**

---

## 🧠 How It Works

1. User enters symptoms
2. Query is preprocessed (cleaning + synonym expansion)
3. TF-IDF converts text into vectors
4. Cosine similarity finds relevant conditions
5. Hybrid scoring improves accuracy
6. System returns:

   * Predicted condition
   * Confidence score
   * Top matches

---

## ⚙️ Tech Stack

* **Python**
* **Scikit-learn**
* **NumPy**
* **Streamlit**
* **Matplotlib (for evaluation)**

---

## 📊 Model Performance

| Metric   | Value                          |
| -------- | ------------------------------ |
| Accuracy | ~85–90%                        |
| Approach | Hybrid Retrieval               |
| Data     | Curated medical knowledge base |

> ⚠️ Accuracy evaluated on realistic, noisy symptom queries (not memorized inputs)

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🧪 Example Input

```
Input: severe headache nausea  
Output: Migraine (Confidence: ~75%)
```

---

## 📂 Project Structure

```
Healthcare-AI-Assistant/
│
├── app.py              # Streamlit app (UI + logic)
├── requirements.txt    # Dependencies
├── README.md           # Documentation
```

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**.
It is **NOT a substitute for professional medical advice**.

---

## 💼 Use Case

* Healthcare AI research
* Clinical decision support prototype
* ML/NLP learning project
* Placement portfolio project

---

## 🔮 Future Improvements

* 🧠 BERT / BioBERT embeddings
* 📊 Advanced evaluation metrics
* 🌐 Deployment (Streamlit Cloud)
* 🧬 Larger medical dataset

---

## 👨‍💻 Author

**Kushal Kadyan**

* GitHub: https://github.com/kushal264
* LinkedIn: (add your profile)

---

## ⭐ If you like this project

Give it a **star ⭐ on GitHub** and share!

