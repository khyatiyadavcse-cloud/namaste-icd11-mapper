import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import joblib
import numpy as np
import os

# Step 1: Load dataset
print("Loading dataset...")
df = pd.read_csv("DATA/namaste_icd11_mapping.csv")
print(f"Loaded {len(df)} rows")
print(df.head())

# Step 2: Load pre-trained sentence embedding model
print("\nLoading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Create embeddings for NAMASTE terms
print("Creating embeddings...")
namaste_terms = df['english_synonym'].tolist()
embeddings = model.encode(namaste_terms, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# Step 4: Build FAISS index for similarity search
print("\nBuilding FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 5: Save everything to MODELS folder
os.makedirs("MODELS", exist_ok=True)

faiss.write_index(index, "MODELS/namaste_index.faiss")
joblib.dump(df, "MODELS/namaste_data.pkl")
model.save("MODELS/sentence_model")

print("\n✅ Training complete!")
print("Saved: MODELS/namaste_index.faiss")
print("Saved: MODELS/namaste_data.pkl")
print("Saved: MODELS/sentence_model")