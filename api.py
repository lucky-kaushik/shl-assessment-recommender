from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import pandas as pd

app = FastAPI()

# Load model and data
model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_csv("shl_products.csv")

class QueryRequest(BaseModel):
    query: str
    top_n: int = 5

@app.post("/recommend")
async def recommend_assessments(data: QueryRequest):
    query_embedding = model.encode(data.query, convert_to_tensor=True)
    corpus_embeddings = model.encode(df["Description"].tolist(), convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = scores.argsort(descending=True)[:data.top_n]
    recommendations = df.iloc[top_results.cpu().numpy()]
    scores = scores[top_results].cpu().numpy()

    output = []
    for i, row in recommendations.iterrows():
        output.append({
            "Product Name": row["Product Name"],
            "Description": row["Description"],
            "Score": round(float(scores[i]), 2)
        })
    return {"results": output}
