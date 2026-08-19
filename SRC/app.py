import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import joblib
import numpy as np

# Load model, index, data (cached so it loads only once)
@st.cache_resource
def load_resources():
    model = SentenceTransformer('MODELS/sentence_model')
    index = faiss.read_index('MODELS/namaste_index.faiss')
    df = joblib.load('MODELS/namaste_data.pkl')
    return model, index, df

model, index, df = load_resources()

st.set_page_config(page_title="NAMASTE-ICD11 Mapper", page_icon="🏥")
st.title("🏥 NAMASTE ↔ ICD-11 Terminology Mapper")
st.write("AYUSH traditional-medicine terms ko standardized ICD-11 codes se map karo.")

query = st.text_input("Enter a symptom or NAMASTE term (English):", "")
top_k = st.slider("Number of matches to show:", 1, 5, 3)

if query:
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)

    st.subheader(f"Results for: '{query}'")
    for rank, idx in enumerate(indices[0]):
        row = df.iloc[idx]
        distance = distances[0][rank]
        confidence = max(0, 100 - distance * 30)  # simple confidence score
        
        st.markdown(f"**{rank+1}. {row['namaste_term']}** ({row['namaste_code']}) → **{row['icd11_term']}** ({row['icd11_code']})")
        st.progress(float(min(confidence / 100, 1.0)))
        st.caption(f"Confidence: {confidence:.1f}% | Distance: {distance:.4f}")
        st.divider()