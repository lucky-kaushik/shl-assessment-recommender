import streamlit as st
import pandas as pd
import os
import zipfile
import gdown
from sentence_transformers import SentenceTransformer, util

# Sample questions for assessments
sample_questions = {
    "Cognitive Ability Test": [
        "What is the next number in the sequence: 2, 4, 8, 16, ?",
        "If John is older than Mary, and Mary is older than Alice, who is the oldest?"
    ],
    "Numerical Reasoning": [
        "What is 25% of 240?",
        "A stock price increases by 15% then drops by 10%. What is the net change?"
    ],
    "Verbal Reasoning": [
        "Choose the word most similar to 'elated': (a) sad (b) thrilled (c) bored",
        "Which word completes the sentence: She has a _____ for classical music."
    ],
    "Situational Judgement Test": [
        "You see a teammate struggling. What do you do?",
        "Your manager asks you to stay late, but you have a prior commitment. What do you do?"
    ],
    "Leadership Potential Assessment": [
        "Describe a time you led a team through a difficult situation.",
        "How do you handle team conflict?"
    ],
}

# Load SHL product data
@st.cache_data
def load_data():
    return pd.read_csv("shl_products.csv")

# Load transformer model (download from Google Drive)
@st.cache_resource
def load_model():
    model_path = "all-MiniLM-L6-v2"
    if not os.path.exists(model_path):
        zip_url = "https://drive.google.com/uc?id=183eNzwaXDwT_pUsvS95GVBpJe_aaIl_k"
        zip_path = "model.zip"
        gdown.download(zip_url, zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
    return SentenceTransformer(model_path)

# Recommender logic
def recommend_assessments(query, df, model, top_n=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(df["Description"].tolist(), convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = scores.argsort(descending=True)[:top_n]
    recommendations = df.iloc[top_results.cpu().numpy()]
    recommendations["Score"] = scores[top_results].cpu().numpy()
    return recommendations

# Streamlit UI
st.set_page_config(page_title="SHL Assessment Recommendation Engine", layout="centered")
st.title("🔍 SHL Assessment Recommendation Engine")

df = load_data()
model = load_model()

user_input = st.text_area("Enter a job description, title, or skill:", height=200)

if st.button("Recommend Assessments") and user_input.strip():
    with st.spinner("Finding best SHL assessments..."):
        results = recommend_assessments(user_input, df, model)
        for i, row in results.iterrows():
            product_name = row["Product Name"]
            st.markdown(f"### ✅ {product_name}")
            st.markdown(f"*{row['Description']}*")
            st.markdown(f"**Match Score:** {row['Score']:.2f}")

            # Create a unique button key
            button_key = f"sample_{i}"

            # Initialize session state
            if button_key not in st.session_state:
                st.session_state[button_key] = False

            # Toggle session state on button click
            if st.button(f"View Sample Questions for {product_name}", key=button_key):
                st.session_state[button_key] = not st.session_state[button_key]

            # Show questions if state is toggled
            if st.session_state[button_key]:
                questions = sample_questions.get(product_name, ["Sample questions not available."])
                st.markdown("**Sample Questions:**")
                for q in questions:
                    st.markdown(f"- {q}")

            st.markdown("---")


