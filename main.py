import os
import sys
import pandas as pd
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add current and SRC directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SRC"))

app = FastAPI(
    title="NAMASTE to ICD-11 AI Mapping Microservice",
    description="AI-powered semantic search & mapping service bridging AYUSH NAMASTE terms with WHO ICD-11 codes.",
    version="1.0.0"
)

# Enable CORS for frontend and backend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dataset fallback definitions
DEFAULT_MAPPING = [
    {"namaste_code": "AY001", "namaste_term": "Jwara", "english_synonym": "fever pyrexia body temperature high", "icd11_code": "1A00", "icd11_term": "Fever unspecified", "category": "General Medicine"},
    {"namaste_code": "AY002", "namaste_term": "Kasa", "english_synonym": "cough respiratory congestion phlegm", "icd11_code": "CA80", "icd11_term": "Cough", "category": "Respiratory System"},
    {"namaste_code": "AY003", "namaste_term": "Atisara", "english_synonym": "diarrhoea loose motion dysentery gastroenteritis", "icd11_code": "DA92", "icd11_term": "Diarrhoea", "category": "Digestive System"},
    {"namaste_code": "AY004", "namaste_term": "Shwasa", "english_synonym": "asthma breathing difficulty dyspnea shortness of breath", "icd11_code": "CA23", "icd11_term": "Asthma", "category": "Respiratory System"},
    {"namaste_code": "AY005", "namaste_term": "Shiroroga", "english_synonym": "headache migraine cephalalgia head pain", "icd11_code": "8A80", "icd11_term": "Headache disorder", "category": "Nervous System"},
    {"namaste_code": "AY006", "namaste_term": "Amlapitta", "english_synonym": "acid reflux heartburn hyperacidity gerd acidity", "icd11_code": "DA22", "icd11_term": "Gastro-oesophageal reflux disease", "category": "Digestive System"},
    {"namaste_code": "AY007", "namaste_term": "Pandu", "english_synonym": "anaemia paleness low hemoglobin iron deficiency", "icd11_code": "3A00", "icd11_term": "Anaemia", "category": "Hematology"},
    {"namaste_code": "AY008", "namaste_term": "Kamala", "english_synonym": "jaundice yellowing bilirubin hepatitis liver disease", "icd11_code": "DB99", "icd11_term": "Jaundice", "category": "Hepatology"},
    {"namaste_code": "AY009", "namaste_term": "Arsha", "english_synonym": "piles haemorrhoids rectal bleeding swollen veins", "icd11_code": "DB30", "icd11_term": "Haemorrhoids", "category": "Digestive System"},
    {"namaste_code": "AY010", "namaste_term": "Vatarakta", "english_synonym": "gout joint pain uric acid arthritis inflammatory joint", "icd11_code": "FA20", "icd11_term": "Gout", "category": "Musculoskeletal"},
    {"namaste_code": "AY011", "namaste_term": "Prameha", "english_synonym": "diabetes mellitus high blood sugar hyperglycemia frequent urination", "icd11_code": "5A11", "icd11_term": "Type 2 diabetes mellitus", "category": "Endocrinology"},
    {"namaste_code": "AY012", "namaste_term": "Hridroga", "english_synonym": "heart disease chest pain angina cardiovascular disorder", "icd11_code": "BA80", "icd11_term": "Ischaemic heart disease", "category": "Cardiology"}
]

class TerminologyMapper:
    def __init__(self):
        self.df = self._load_data()
        self.has_ml = False
        self.model = None
        self.index = None
        self._init_ml_models()

    def _load_data(self) -> pd.DataFrame:
        csv_path = os.path.join(os.path.dirname(__file__), "DATA", "namaste_icd11_mapping.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'category' not in df.columns:
                    df['category'] = 'AYUSH Clinical'
                return df
            except Exception as e:
                print(f"Warning loading CSV: {e}")
        return pd.DataFrame(DEFAULT_MAPPING)

    def _init_ml_models(self):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            print("Initializing SentenceTransformer model (all-MiniLM-L6-v2)...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            corpus = [f"{row['namaste_term']} {row.get('english_synonym', '')} {row['icd11_term']}" for _, row in self.df.iterrows()]
            embeddings = self.model.encode(corpus, convert_to_numpy=True).astype('float32')
            faiss.normalize_L2(embeddings)
            
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product for Cosine Similarity
            self.index.add(embeddings)
            self.has_ml = True
            print("✅ SentenceTransformer & FAISS Vector Index loaded successfully.")
        except Exception as e:
            print(f"⚠️ Vector search fallback active (ML library note: {e})")
            self.has_ml = False

    def predict(self, query: str, top_k: int = 3, alpha: float = 0.5):
        query_clean = query.strip().lower()
        results = []

        if self.has_ml and self.model and self.index:
            try:
                q_emb = self.model.encode([query_clean], convert_to_numpy=True).astype('float32')
                import faiss
                faiss.normalize_L2(q_emb)
                sims, indices = self.index.search(q_emb, min(top_k, len(self.df)))
                
                for rank, idx in enumerate(indices[0]):
                    if idx < 0 or idx >= len(self.df):
                        continue
                    row = self.df.iloc[idx]
                    score = float(sims[0][rank])
                    confidence = round(max(50.0, min(99.5, score * 100)), 1)
                    results.append({
                        "rank": rank + 1,
                        "namaste_code": str(row.get('namaste_code', 'AY000')),
                        "namaste_term": str(row.get('namaste_term', '')),
                        "english_synonym": str(row.get('english_synonym', '')),
                        "icd11_code": str(row.get('icd11_code', '')),
                        "icd11_term": str(row.get('icd11_term', '')),
                        "category": str(row.get('category', 'AYUSH Terminology')),
                        "confidence": confidence,
                        "engine": "FAISS Vector Semantic Search"
                    })
                return results
            except Exception as ex:
                print(f"Predict ML error fallback: {ex}")

        # String matching & Keyword Similarity Fallback
        scored_rows = []
        for idx, row in self.df.iterrows():
            namaste_term = str(row.get('namaste_term', '')).lower()
            synonym = str(row.get('english_synonym', '')).lower()
            icd_term = str(row.get('icd11_term', '')).lower()
            icd_code = str(row.get('icd11_code', '')).lower()
            namaste_code = str(row.get('namaste_code', '')).lower()

            score = 0.0
            if query_clean == namaste_term or query_clean == namaste_code:
                score = 0.98
            elif query_clean in namaste_term or namaste_term in query_clean:
                score = 0.90
            elif query_clean in synonym:
                score = 0.85
            elif any(w in synonym or w in namaste_term or w in icd_term for w in query_clean.split()):
                score = 0.72
            elif query_clean in icd_term or query_clean in icd_code:
                score = 0.65
            else:
                score = 0.40

            confidence = round(score * 100, 1)
            scored_rows.append((score, confidence, row))

        scored_rows.sort(key=lambda x: x[0], reverse=True)
        for rank, (score, confidence, row) in enumerate(scored_rows[:top_k]):
            results.append({
                "rank": rank + 1,
                "namaste_code": str(row.get('namaste_code', 'AY000')),
                "namaste_term": str(row.get('namaste_term', '')),
                "english_synonym": str(row.get('english_synonym', '')),
                "icd11_code": str(row.get('icd11_code', '')),
                "icd11_term": str(row.get('icd11_term', '')),
                "category": str(row.get('category', 'AYUSH Terminology')),
                "confidence": confidence,
                "engine": "Hybrid Keyword Engine"
            })

        return results

# Initialize Terminology Mapper Singleton
mapper = TerminologyMapper()

class PredictRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    alpha: Optional[float] = 0.5

@app.get("/")
def read_root():
    return {
        "service": "NAMASTE to ICD-11 AI Mapping Microservice",
        "status": "online",
        "model": "SentenceTransformers (all-MiniLM-L6-v2) + FAISS",
        "dataset_size": len(mapper.df)
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "ml_ready": mapper.has_ml}

@app.post("/predict")
def predict_post(req: PredictRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query string is required")
    results = mapper.predict(query=req.query, top_k=req.top_k, alpha=req.alpha)
    return {
        "success": True,
        "query": req.query,
        "results_count": len(results),
        "results": results
    }

@app.get("/predict")
def predict_get(q: str = Query(..., description="AYUSH terminology or condition"), top_k: int = 3, alpha: float = 0.5):
    results = mapper.predict(query=q, top_k=top_k, alpha=alpha)
    return {
        "success": True,
        "query": q,
        "results_count": len(results),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
