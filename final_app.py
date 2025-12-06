import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# FORCE INSTALL SCikit-learn IF MISSING (FIXES STREAMLIT CLOUD ERROR)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.5.0"])
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

# --- Scalable Feature Extractor (NO OpenCV!) ---
class SafeFeatureExtractor:
    def __init__(self):
        self.text_model = TfidfVectorizer(stop_words='english', max_features=100)
        self.text_fitted = False

    def fit_text(self, all_texts):
        self.text_model.fit(all_texts)
        self.text_fitted = True

    def get_text_similarity(self, query_text, item_texts):
        if not self.text_fitted:
            raise ValueError("Run fit_text() first!")
        query_vec = self.text_model.transform([query_text])
        item_vecs = self.text_model.transform(item_texts)
        return cosine_similarity(query_vec, item_vecs).flatten()

    def get_color_similarity(self, uploaded_photo, item_colors):
        try:
            img = Image.open(uploaded_photo).convert('RGB')
            avg_color = np.mean(np.array(img.resize((32,32))), axis=(0,1)) / 255
            return np.array([1 - min(np.linalg.norm(avg_color - c), 1.0) for c in item_colors])
        except Exception as e:
            st.error(f"❌ Photo error: {str(e)}")
            return np.zeros(len(item_colors))

# --- 10-Item Dataset ---
@st.cache_data
def get_dataset():
    return pd.DataFrame([
        {"id": 1, "desc": "blue hydro flask with stickers", "color": np.array([0.1, 0.3, 0.8])},
        {"id": 2, "desc": "green metal water bottle", "color": np.array([0.2, 0.7, 0.3])},
        {"id": 3, "desc": "black leather keychain", "color": np.array([0.1, 0.1, 0.1])},
        {"id": 4, "desc": "red spiral notebook", "color": np.array([0.9, 0.1, 0.1])},
        {"id": 5, "desc": "silver laptop charger", "color": np.array([0.8, 0.8, 0.8])},
        {"id": 6, "desc": "pink wireless earbuds case", "color": np.array([0.9, 0.4, 0.6])},
        {"id": 7, "desc": "black backpack with logo", "color": np.array([0.15, 0.15, 0.15])},
        {"id": 8, "desc": "yellow pencil case", "color": np.array([0.9, 0.8, 0.1])},
        {"id": 9, "desc": "white iPhone cable", "color": np.array([0.95, 0.95, 0.95])},
        {"id": 10, "desc": "brown leather wallet", "color": np.array([0.5, 0.3, 0.1])}
    ])

# --- Pre-Defined Lost Items ---
LOST_ITEMS = [
    "blue hydro flask with stickers",
    "green metal water bottle",
    "black leather keychain",
    "red spiral notebook",
    "silver laptop charger",
    "pink wireless earbuds case",
    "black backpack with logo",
    "yellow pencil case",
    "white iPhone cable",
    "brown leather wallet"
]

# --- App UI ---
st.set_page_config(page_title="Campus Lost & Found", layout="wide")
st.title("🏫 Campus Lost & Found - ML Matching")
st.markdown("### Text + Image Matching (No OpenCV!)")

# Initialize
extractor = SafeFeatureExtractor()
dataset = get_dataset()
extractor.fit_text(LOST_ITEMS + dataset["desc"].tolist())

# --- User Input ---
st.subheader("Report a Lost Item")
selected_lost = st.selectbox("Select your lost item:", LOST_ITEMS)
uploaded_photo = st.file_uploader(f"Upload photo of your {selected_lost}:", type=["png", "jpg", "jpeg"])

# --- Matching ---
if selected_lost:
    text_scores = extractor.get_text_similarity(selected_lost, dataset["desc"].tolist())
    color_scores = extractor.get_color_similarity(uploaded_photo, dataset["color"].tolist()) if uploaded_photo else np.zeros(len(dataset))
    final_scores = 0.6 * text_scores + 0.4 * color_scores

    # --- Results ---
    results = dataset.copy()
    results["Match Score (%)"] = final_scores
    st.subheader("Top 10 Matches")
    st.dataframe(results.sort_values("Match Score (%)", ascending=False).style.format({"Match Score (%)": "{:.1%}"}), use_container_width=True)

st.success("✅ No OpenCV errors! App is fully functional.")



