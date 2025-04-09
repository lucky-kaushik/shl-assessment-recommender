# SHL Assessment Recommendation Engine

🔍 A smart, AI-powered engine that recommends the most relevant SHL assessments based on a job title, description, or key skills.

---

## 🚀 Live Links

- **🔗 Web App:** [Streamlit App](https://shl-assessment-recommendergit-rpqvyjraka2xnozk66sb3s.streamlit.app/)
- **🔗 API Endpoint:** [`/recommend`](https://shl-assessment-recommender-api.onrender.com/recommend)
- **🔗 API Docs:** [`/docs`](https://shl-assessment-recommender-api.onrender.com/docs)
- **🔗 GitHub:** [GitHub Repo](https://github.com/lucky-kaushik/shl-assessment-recommender)

---

## 💡 Features

- 💬 Natural Language Input (job description, title, or skills)
- 🎯 Embedding-based semantic matching
- 📊 SHL product descriptions embedded using `sentence-transformers`
- 📄 Sample Questions shown for each assessment
- ⚡ FastAPI backend with `/recommend` endpoint
- 📱 Interactive frontend built using Streamlit

---

## 📦 Technologies Used

- `Streamlit` – Web UI
- `FastAPI` – RESTful API backend
- `sentence-transformers` – Semantic similarity
- `pandas` – Data loading
- `gdown` – Model download from Google Drive
- `Render` – Deployment platform

---

## 🔁 How to Run Locally

### 🔹 Frontend (Streamlit App)
```bash
# Create venv and activate it
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### 🔹 Backend (FastAPI API)
```bash
uvicorn api:app --reload
# Go to http://localhost:8000/docs to test
```

---

## 📩 How `/recommend` API Works

- **Endpoint:** `/recommend`
- **Method:** POST
- **Request Body (JSON):**
```json
{
  "query": "Sales leader with strong communication"
}
```
- **Response (JSON):**
```json
[
  {
    "Product Name": "Verbal Reasoning",
    "Description": "Assesses understanding and interpretation of written information",
    "Score": 0.91
  },
  ...
]
```

---

## 📁 Project Structure

- `app.py` – Streamlit UI
- `api.py` – FastAPI backend
- `shl_products.csv` – Assessment metadata
- `requirements.txt` – Dependencies list

---

## ✍️ Author

Developed with dedication and love for the SHL Research Intern Assessment by Lucky Kaushik

---

## 📃 License

MIT License
