import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2


st.title("📄 Resume Ranker using NLP")
st.markdown("Upload resumes and compare them to a job description")


job_desc_file = st.file_uploader("Upload Job Description (.txt)", type="txt")
resumes = st.file_uploader("Upload Resumes (.txt or .pdf)", type=["txt", "pdf"], accept_multiple_files=True)

if job_desc_file and resumes:
    job_desc = job_desc_file.read().decode("utf-8")
    resume_texts = []
    resume_names = []

    for file in resumes:
        resume_names.append(file.name)
        if file.name.endswith(".txt"):
            resume_texts.append(file.read().decode("utf-8"))
        elif file.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            resume_texts.append(text)

  
    all_docs = [job_desc] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    ranked = sorted(zip(resume_names, similarities), key=lambda x: x[1], reverse=True)

    st.subheader("📊 Resume Rankings")
    for i, (name, score) in enumerate(ranked, 1):
        st.write(f"{i}. {name} — Score: {score:.2f}")
