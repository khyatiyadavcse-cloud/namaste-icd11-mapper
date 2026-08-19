import streamlit as st
from predict import TerminologyMapper

# Page Configuration
st.set_page_config(page_title="NAMASTE ⇄ ICD-11 Mapper", page_icon="🏥", layout="wide")

@st.cache_resource
def load_mapper():
    # Model aur indexes ko memory me load karein
    return TerminologyMapper()

st.title("🏥 NAMASTE to ICD-11 Mapping Engine")
st.markdown("Bridge traditional AYUSH terminology with WHO ICD-11 standard codes using Hybrid ML matching.")

with st.spinner("Loading Hybrid Search Engine..."):
    try:
        mapper = load_mapper()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Sidebar
st.sidebar.header("⚙️ Search Settings")
top_k = st.sidebar.slider("Number of Results", 1, 10, 3)
alpha = st.sidebar.slider("Keyword vs Semantic Weight", 0.0, 1.0, 0.5, help="0 = Pure Semantic, 1 = Pure Keyword")

# Search UI
query = st.text_input("🔍 Enter AYUSH Term (e.g., Jwara, Amavata, Kasa):", placeholder="Type terminology here...")

if query:
    with st.spinner("Searching..."):
        results = mapper.predict(query=query, top_k=top_k, alpha=alpha)
        
        st.subheader(f"Top Matches for: **'{query}'**")
        st.write("---")
        
        for rank, res in enumerate(results):
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"#### {rank+1}. {res['namaste_term']}")
                    st.caption(f"**NAMASTE Code:** {res['namaste_code']} | **Category:** {res['category']}")
                    
                with col2:
                    st.metric(label="ICD-11 Code", value=res['icd11_code'])
                    
                with col3:
                    st.metric(label="Confidence", value=f"{res['confidence']}%")
                
                st.progress(res['confidence'] / 100.0)
                st.write("")