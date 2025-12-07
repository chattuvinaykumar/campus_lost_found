# --- ALL IMPORTS FIRST (REQUIRED FOR STREAMLIT) ---
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import math
from collections import defaultdict

# --- FIRST STREAMLIT COMMAND: PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title="Campus Lost & Found", layout="wide")

# --- MANUAL TF-IDF IMPLEMENTATION (NO EXTERNAL DEPENDENCIES) ---
class ManualTFIDF:
    def __init__(self):
        self.vocab = []
        self.doc_count = 0
        self.word_doc_count = defaultdict(int)

    def fit(self, documents):
        """Train TF-IDF on a list of documents"""
        self.doc_count = len(documents)
        all_words = []
        
        # Build vocabulary and count document frequency
        for doc in documents:
            words = self._preprocess(doc)
            unique_words = set(words)
            all_words.extend(unique_words)
            for word in unique_words:
                self.word_doc_count[word] += 1
        
        # Create sorted vocabulary
        self.vocab = sorted(list(set(all_words)))

    def transform(self, documents):
        """Convert documents to TF-IDF vectors"""
        vectors = []
        for doc in documents:
            words = self._preprocess(doc)
            word_count = defaultdict(int)
            for word in words:
                word_count[word] += 1
            
            # Calculate TF-IDF for each word in vocab
            vector = []
            total_words = len(words)
            for word in self.vocab:
                # Term Frequency (TF)
                tf = word_count[word] / total_words if total_words > 0 else 0
                # Inverse Document Frequency (IDF)
                idf = math.log(self.doc_count / (self.word_doc_count[word] + 1)) if word in self.word_doc_count else 0
                # TF-IDF score
                tfidf = tf * idf
                vector.append(tfidf)
            vectors.append(vector)
        return np.array(vectors)

    def _preprocess(self, text):
        """Simple text preprocessing (lowercase, remove stopwords)"""
        stopwords = {"the", "and", "of", "a", "to", "in", "is", "it", "you", "that", "this"}
        text = text.lower()
        words = [word.strip(".,!?") for word in text.split() if word not in stopwords]
        return words

# --- COSINE SIMILARITY IMPLEMENTED MANUALLY ---
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

# --- 10-Item Found Item Dataset ---
@st.cache_data
def get_found_item_dataset():
    """Pre-defined dataset of found items with descriptions and color profiles"""
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

# --- Pre-Defined Lost Items for Demo ---
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

# --- PROFESSIONAL APP UI ---
st.title("🏫 Campus Lost & Found - Automated ML Matching System")
st.markdown("### Multi-Modal Text + Image Matching Using Traditional Machine Learning")

# Initialize feature extractor and dataset
extractor = ManualTFIDF()
found_dataset = get_found_item_dataset()
extractor.fit(LOST_ITEMS + found_dataset["desc"].tolist())  # Fit model once on all text

# --- User Input Section ---
st.subheader("Report a Lost Item")
selected_lost_item = st.selectbox("Select your lost item:", LOST_ITEMS)
uploaded_image = st.file_uploader(f"Upload photo of your {selected_lost_item}:", type=["png", "jpg", "jpeg"])

# --- Matching Logic ---
if selected_lost_item:
    # Calculate text similarity
    text_similarity_scores = extractor.get_text_similarity(selected_lost_item, found_dataset["desc"].tolist())
    
    # Calculate image similarity (if image is uploaded)
    image_similarity_scores = np.zeros(len(found_dataset))
    if uploaded_image is not None:
        st.image(Image.open(uploaded_image), caption="Uploaded Lost Item", width=200)
        try:
            img = Image.open(uploaded_image).convert('RGB')
            avg_color = np.mean(np.array(img.resize((32, 32))), axis=(0, 1)) / 255  # Normalize to 0-1 range
            image_similarity_scores = [1 - min(np.linalg.norm(avg_color - c), 1.0) for c in found_dataset["color"]]
            st.success("✅ Image features extracted successfully! Match score updated.")
        except Exception as e:
            st.error(f"❌ Failed to process image: {str(e)}")
    
    # Combine scores
    final_scores = (0.6 * np.array(text_similarity_scores)) + (0.4 * np.array(image_similarity_scores))

    # --- Display Results ---
    st.subheader("Top 10 Found Item Matches")
    results = found_dataset.copy()
    results["Match Score (%)"] = final_scores
    results = results.sort_values("Match Score (%)", ascending=False).reset_index(drop=True)
    
    # Format and display results table
    st.dataframe(
        results[["desc", "Match Score (%)"]].style.format({"Match Score (%)": "{:.1%}"}),
        use_container_width=True,
        height=500
    )

# --- Professional Footer ---
st.markdown("---")
st.info("💡 This system uses manually implemented TF-IDF for text matching and color profile analysis for image matching, ensuring scalability and deployment compatibility.")
