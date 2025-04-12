import streamlit as st
import pandas as pd
import os
import zipfile
import gdown
from sentence_transformers import SentenceTransformer, util

# Sample questions
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

# Smart match function
def get_matched_sample(product_name):
    name = product_name.lower()
    if "numerical" in name:
        return sample_questions.get("Numerical Reasoning")
    elif "verbal" in name:
        return sample_questions.get("Verbal Reasoning")
    elif "situational" in name:
        return sample_questions.get("Situational Judgement Test")
    elif "leadership" in name:
        return sample_questions.get("Leadership Potential Assessment")
    elif "cognitive" in name:
        return sample_questions.get("Cognitive Ability Test")
    else:
        return ["Sample questions not available."]

# Load SHL data
@st.cache_data
def load_data():
    return pd.read_csv("shl_products.csv")

# Load transformer model
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

# Recommendation logic
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

            # ✅ Toggle to show questions
            if st.toggle(f"View Sample Questions for {product_name}", key=f"toggle_{i}"):
                questions = get_matched_sample(product_name)
                st.markdown("**Sample Questions:**")
                for q in questions:
                    st.markdown(f"- {q}")

            st.markdown("---")
