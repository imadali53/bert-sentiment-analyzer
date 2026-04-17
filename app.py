
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="BERT Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 BERT Sentiment Analyzer")
st.markdown("Powered by DistilBERT — built by Imad Khan")

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

st.subheader("Analyze a Single Review")
user_input = st.text_area("Paste your review here:", height=150)

if st.button("Analyze"):
    if user_input.strip():
        result = classifier(user_input)[0]
        label = "POSITIVE ✅" if result['label'] == "POSITIVE" else "NEGATIVE ❌"
        confidence = round(result['score'] * 100, 1)
        st.markdown(f"### Result: {label}")
        st.markdown(f"**Confidence:** {confidence}%")
        st.progress(result['score'])
    else:
        st.warning("Please enter a review first.")

st.subheader("Analyze Multiple Reviews")
bulk_input = st.text_area(
    "Paste multiple reviews (one per line):",
    height=200
)

if st.button("Analyze All"):
    if bulk_input.strip():
        reviews = [r.strip() for r in bulk_input.split("\n") if r.strip()]
        results = classifier(reviews)
        positive = 0
        negative = 0
        for review, result in zip(reviews, results):
            label = "✅ POSITIVE" if result['label'] == "POSITIVE" else "❌ NEGATIVE"
            confidence = round(result['score'] * 100, 1)
            st.markdown(f"**{label}** ({confidence}%) — {review}")
            if result['label'] == "POSITIVE":
                positive += 1
            else:
                negative += 1
        st.markdown("---")
        st.markdown(f"**Summary — Positive: {positive} | Negative: {negative}**")
    else:
        st.warning("Please enter at least one review.")
