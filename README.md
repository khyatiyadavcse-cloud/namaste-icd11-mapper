# 🏥 NAMASTE ⇄ ICD-11 Mapping Engine

An AI-powered Semantic Search & Mapping tool designed to bridge traditional **AYUSH (NAMASTE)** terminology with global **WHO ICD-11** standard disease/condition codes for seamless Electronic Health Records (EHR) integration.

---

## 🚀 Key Features

- **Hybrid Search Engine:** Combines **BM25 (Exact Keyword Match)** and **FAISS Vector Search (Dense Semantic Match)** for accurate medical terminology mapping.
- **Normalized Confidence Scores:** Real-time similarity scores calculated using Cosine Similarity and displayed as a percentage (0% to 100%).
- **Interactive Web Interface:** Clean and responsive Streamlit UI for clinicians and researchers.
- **Fast Inference:** High-speed vector retrieval using FAISS index and MiniLM embeddings.

---

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Frontend / Dashboard:** Streamlit
- **ML / NLP Models:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Database:** Meta FAISS (`faiss-cpu`)
- **Keyword Search:** BM25 (`rank-bm25`)
- **Data Processing:** Pandas, NumPy

---

## 📂 Project Structure

```text
AYUSH-ICD11-ML/
├── DATA/
│   └── dataset.csv          # AYUSH & ICD-11 Mapping Data
├── MODELS/                  # Artifacts generated post training
│   ├── index.bin            # FAISS Index
│   ├── bm25.pkl             # BM25 Keyword Model
│   └── processed_data.pkl   # Processed Data Cache
├── SRC/
│   ├── train.py             # Model training & vector index script
│   └── predict.py           # Hybrid search inference engine
├── app.py                   # Streamlit web application
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
