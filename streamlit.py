# =========================================
# STREAMLIT HEALTHCARE AI ASSISTANT
# (MAIN LOGIC + UI IN ONE FILE)
# =========================================

import streamlit as st
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================
# 1. DATA
# =========================================
medical_docs = [
    {"condition": "Diabetes", "text": "frequent urination fatigue thirst blurred vision"},
    {"condition": "Hypertension", "text": "high blood pressure headache dizziness nosebleeds"},
    {"condition": "Asthma", "text": "breathing difficulty wheezing chest tightness cough"},
    {"condition": "Migraine", "text": "severe headache nausea vomiting light sensitivity"},
    {"condition": "COVID-19", "text": "fever cough fatigue loss of smell sore throat"},
    {"condition": "Anemia", "text": "fatigue pallor dizziness cold hands"},
    {"condition": "Pneumonia", "text": "fever cough chest pain breathing difficulty"},
    {"condition": "UTI", "text": "burning urination pelvic pain cloudy urine"},
    {"condition": "Depression", "text": "sadness low energy sleep problems hopelessness"},
    {"condition": "Appendicitis", "text": "lower abdomen pain nausea fever"}
]

corpus = [d["text"] for d in medical_docs]

# =========================================
# 2. TF-IDF
# =========================================
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
doc_vectors = vectorizer.fit_transform(corpus)

# =========================================
# 3. SYNONYMS (BOOST ACCURACY)
# =========================================
SYNONYMS = {
    "tired": "fatigue",
    "weak": "fatigue",
    "vomiting": "nausea",
    "pee": "urination",
    "toilet": "urination",
    "sad": "depression"
}

def preprocess(query):
    query = query.lower()
    query = re.sub(r"[^a-z0-9\s]", " ", query)

    words = query.split()
    expanded = []

    for w in words:
        if w in SYNONYMS:
            expanded.append(SYNONYMS[w])
        expanded.append(w)

    return " ".join(expanded)

# =========================================
# 4. RETRIEVE (HYBRID)
# =========================================
def retrieve(query, top_k=3):
    clean = preprocess(query)
    query_vec = vectorizer.transform([clean])
    tfidf_scores = cosine_similarity(query_vec, doc_vectors)[0]

    query_tokens = set(clean.split())
    scores = []

    for i, doc in enumerate(medical_docs):
        doc_tokens = set(doc["text"].split())

        overlap = len(query_tokens & doc_tokens)
        keyword_score = overlap / (len(query_tokens) + 1e-5)

        final_score = 0.6 * tfidf_scores[i] + 0.4 * keyword_score

        # symptom boost
        if "pain" in query_tokens and "pain" in doc_tokens:
            final_score += 0.05

        scores.append(final_score)

    scores = np.array(scores)
    top_idx = scores.argsort()[-top_k:][::-1]

    return [(medical_docs[i], scores[i]) for i in top_idx]

# =========================================
# 5. CONFIDENCE
# =========================================
def classify_conf(score):
    if score >= 0.60:
        return "HIGH"
    elif score >= 0.35:
        return "MEDIUM"
    elif score >= 0.20:
        return "LOW"
    else:
        return "NONE"

# =========================================
# 6. SAFETY
# =========================================
def safety_check(text):
    unsafe = ["overdose", "high dosage", "guaranteed cure"]
    for w in unsafe:
        if w in text.lower():
            return "⚠️ Unsafe advice detected"
    return None

# =========================================
# 7. STREAMLIT UI
# =========================================
st.set_page_config(page_title="Healthcare AI Assistant")

st.title("🏥 Healthcare AI Assistant")
st.write("Offline AI Diagnosis System (TF-IDF + Hybrid)")

query = st.text_input("Enter symptoms")

if st.button("Analyze"):

    if query.strip() == "":
        st.warning("Please enter symptoms")
    else:
        results = retrieve(query)
        best_doc, score = results[0]

        confidence = classify_conf(score)

        st.subheader("💡 Prediction")
        st.success(f"{best_doc['condition']}")

        st.write(f"📊 Confidence: {round(score*100,2)}% ({confidence})")

        warning = safety_check(best_doc["text"])
        if warning:
            st.warning(warning)

        st.subheader("📚 Top Matches")
        for doc, s in results:
            st.write(f"{doc['condition']} ({round(s*100,2)}%)")

        st.info("⚠️ Not medical advice")