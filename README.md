# shl-assessment-recommender
A GenAI-powered recommendation engine that suggests SHL assessments based on job descriptions using semantic similarity.
# 🧠 SHL Assessment Recommendation Engine

This is a web-based AI-powered tool that recommends the most relevant SHL assessments based on a job description, title, or skills using semantic search (RAG-style). Built for SHL's GenAI Internship Assessment Submission.

## 🚀 Features

- 🔍 Accepts any natural language input (job description or role)
- 🤖 Uses Sentence Transformers to semantically match SHL assessments
- 📊 Ranks top matches by relevance score
- 🌐 Fully deployed as a Streamlit web app
- 🧠 Lightweight and blazing fast

## 📁 Files

- `app.py` – Main Streamlit app
- `shl_products.csv` – SHL assessments catalog
- `requirements.txt` – Python dependencies

## 🔧 Tech Stack

- Streamlit
- SentenceTransformers (`all-MiniLM-L6-v2`)
- Pandas
- Python

## ▶️ How to Run Locally

```bash
git clone https://github.com/yourusername/shl-assessment-recommender.git
cd shl-assessment-recommender
pip install -r requirements.txt
streamlit run app.py
