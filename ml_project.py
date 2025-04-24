import os
import re
import fitz  # PyMuPDF
import streamlit as st
from transformers import pipeline

# Load models
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="pszemraj/led-large-book-summary", tokenizer="pszemraj/led-large-book-summary")

@st.cache_resource
def load_ner_pipeline():
    return pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

def clean_text(text):
    text = re.sub(r"(Creative Commons.*?)(?:\n|$)", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def summarize_text(summarizer, text):
    if len(text) < 10:
        return "Not enough text to summarize."
    summary_output = summarizer(text[:4000], max_length=500, min_length=50, do_sample=False)
    return summary_output[0]['summary_text']

def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# UI
st.title("Disease Research Paper Summarizer")
disease_name = st.text_input("Enter the disease name:")

uploaded_file = st.file_uploader("Upload a PDF related to the disease", type="pdf")

if uploaded_file and disease_name:
    st.info(f"Processing file for disease: {disease_name}")
    raw_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = clean_text(raw_text)

    if cleaned_text:
        st.subheader("Summarizing...")
        summarizer = load_summarizer()
        summary = summarize_text(summarizer, cleaned_text)
        st.success("Summary Ready!")
        st.write(summary)

        st.subheader("Biomedical Named Entities")
        ner_pipeline = load_ner_pipeline()
        ner_results = ner_pipeline(cleaned_text)

        for entity in ner_results:
            st.markdown(f"- **{entity['entity_group']}**: {entity['word']} (score: {entity['score']:.2f})")
    else:
        st.warning("No valid text found in the PDF.")
