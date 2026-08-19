import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import joblib
import numpy as np

# Step 1: Load saved model, index and data
print("Loading model and index...")
model = SentenceTransformer('MODELS/sentence_model')
index = faiss.read_index('MODELS/namaste_index.faiss')
df = joblib.load('MODELS/namaste_data.pkl')

def predict_icd11(query_term, top_k=3):
    # Convert input term to embedding
    query_embedding = model.encode([query_term]).astype('float32')
    
    # Search for closest matches in FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    print(f"\n🔍 Query: '{query_term}'")
    print(f"Top {top_k} matches:\n")
    
    for rank, idx in enumerate(indices[0]):
        row = df.iloc[idx]
        distance = distances[0][rank]
        print(f"{rank+1}. NAMASTE: {row['namaste_term']} ({row['namaste_code']})")
        print(f"   ICD-11: {row['icd11_term']} ({row['icd11_code']})")
        print(f"   Distance: {distance:.4f}\n")

# Test with a sample query
if __name__ == "__main__":
    test_query = "fever"
    predict_icd11(test_query)