import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


with open("job_description.txt", "r", encoding="utf-8") as file:
    job_description = file.read()


resume_folder = "resumes"
resumes = []
resume_names = []


for filename in os.listdir(resume_folder):
    filepath = os.path.join(resume_folder, filename)

    if filename.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            resumes.append(f.read())
            resume_names.append(filename)

    elif filename.endswith(".pdf"):
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            resumes.append(text)
            resume_names.append(filename)

documents = [job_description] + resumes
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)


scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Rank resumes
ranked_resumes = sorted(zip(resume_names, scores), key=lambda x: x[1], reverse=True)

print("\n📊 Resume Ranking Based on Job Match:\n")
for i, (name, score) in enumerate(ranked_resumes, 1):
    print(f"{i}. {name} — Score: {score:.2f}")
