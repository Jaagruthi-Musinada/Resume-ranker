# AI-Powered Resume Ranker - Simplified Version with ATS Feedback Only

import streamlit as st
import PyPDF2
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import Counter

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Helper: Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Helper: Clean and preprocess text
def clean_text(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text.lower()

# Extract keywords from job description
def extract_keywords(text, top_n=15):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    common_words = set(["with", "from", "your", "have", "will", "this", "that", "they", "which", "about"])
    words = [w for w in words if w not in common_words]
    most_common = Counter(words).most_common(top_n)
    return [word for word, _ in most_common]

# Compute semantic similarity
def compute_similarity(resume_text, jd_text):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(resume_embedding, jd_embedding)
    return float(similarity_score[0][0])

# Keyword match ratio
def compute_keyword_match(resume_text, jd_keywords):
    resume_words = set(resume_text.split())
    matched = [kw for kw in jd_keywords if kw in resume_words]
    return matched, len(matched) / len(jd_keywords) if jd_keywords else 0

# Streamlit UI
st.title("\U0001F4CB AI-Powered Resume Ranker (ATS Feedback Only)")
st.write("Upload your resume and paste a job description to receive a score and targeted feedback.")

uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_input = st.text_area("Paste Job Description", height=250)

if uploaded_resume and jd_input:
    with st.spinner("Processing..."):
        resume_text = extract_text_from_pdf(uploaded_resume)
        resume_text_clean = clean_text(resume_text)
        jd_text_clean = clean_text(jd_input)

        # Score and feedback
        similarity_score = compute_similarity(resume_text_clean, jd_text_clean)
        jd_keywords = extract_keywords(jd_input, top_n=15)
        matched_keywords, match_ratio = compute_keyword_match(resume_text_clean, jd_keywords)
        missing = list(set(jd_keywords) - set(matched_keywords))

        # Output results
        st.success(f"\U0001F4C8 Overall Match Score: {(0.7 * similarity_score + 0.3 * match_ratio) * 100:.2f}%")

        st.markdown("---")
        if similarity_score > 0.75 and match_ratio > 0.6:
            st.markdown("### \U0001F389 Strong Match! Your resume aligns well with the job description.")
        elif similarity_score > 0.5 or match_ratio > 0.4:
            st.markdown("### \U0001F914 Moderate Match — Consider revising your resume to include more relevant experience and keywords.")
        else:
            st.markdown("### ❌ Weak Match — Your resume may be lacking in relevant skills or keywords mentioned in the job description.")

        if missing:
            st.markdown("### ⚠️ Suggestions to Improve:")
            st.markdown("- Include key terms from the job description in your resume (e.g., project descriptions, skills list).")
            st.markdown("- Strengthen alignment by adding more technical contributions and responsibilities relevant to the job role.")

        st.markdown("---")
        st.markdown("**Tip:** Use the job description as a guide to tailor your resume content and terminology.")
