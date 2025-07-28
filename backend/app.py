from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import spacy
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load NLP models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load job roles data
with open("job_roles.json", "r") as f:
    job_data = json.load(f)

job_roles = list(job_data.keys())
job_embeddings = {
    role: model.encode(job_data[role]) for role in job_roles
}


def extract_text(file_stream):
    reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def extract_keywords(text):
    doc = nlp(text)
    return list(set([token.text.lower() for token in doc if token.is_alpha and not token.is_stop]))


@app.route("/upload", methods=["POST"])
def upload_resume():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        text = extract_text(file)
        skills = extract_keywords(text)
        resume_vector = model.encode(text)

        match_scores = []
        for role, vector in job_embeddings.items():
            score = cosine_similarity([resume_vector], [vector])[0][0]
            match_scores.append((role, round(float(score), 4)))

        match_scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = match_scores[:5]

        response = {
            "message": f"Resume '{file.filename}' processed.",
            "skills": skills,
            "match_scores": top_matches,
            "predicted_domain": top_matches[0][0]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8900)
