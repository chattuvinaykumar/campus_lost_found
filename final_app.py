import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
st.title("🔍 Campus Lost & Found - Text + Image Matching")

class FeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def extract_text(self, text_list):
        return self.vectorizer.fit_transform(text_list).toarray()

    def extract_image(self, image_file):
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        chans = cv2.split(image)
        features = []
        for chan in chans:
            hist = cv2.calcHist([chan], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        return np.array(features)

    def combine(self, text_vec, image_vec=None):
        if image_vec is not None:
            return np.concatenate([text_vec, image_vec])
        return text_vec

# Sample found items
found_items = [
    {"desc": "blue water bottle with stickers"},
    {"desc": "black keychain"},
    {"desc": "green flask"},
]

df_found = pd.DataFrame(found_items)

st.header("📥 Report a Lost Item")

# 1. Description input
lost_desc = st.text_input("Enter lost item description", "blue water bottle")
# 2. Image upload
lost_image = st.file_uploader("Upload lost item photo", type=["jpg", "png"])

extractor = FeatureExtractor()

# 📌 FIX: Fit TF-IDF on lost + found texts together!
all_descs = [lost_desc] + [item["desc"] for item in found_items]
text_features = extractor.vectorizer.fit_transform(all_descs).toarray()

lost_text_vec = text_features[0]
found_text_vecs = text_features[1:]

# Extract image feature if uploaded
if lost_image:
    lost_color = extractor.extract_image(lost_image)
else:
    lost_color = np.random.rand(96)

lost_combined = extractor.combine(lost_text_vec, lost_color)

# Extract features from found items
found_texts = df_found["desc"].tolist()
found_text_vecs = extractor.extract_text(found_texts)

# Match scoring
scores = []
for i, row in df_found.iterrows():
    fake_color = np.random.rand(96)  # Dummy color vector
    combined_vec = extractor.combine(found_text_vecs[i], fake_color)
    sim = cosine_similarity([lost_combined], [combined_vec])[0][0]
    scores.append(sim)

df_found["Match Score (%)"] = [f"{round(s * 100, 1)}%" for s in scores]

st.subheader("🔗 Top Match Suggestions")
st.dataframe(df_found.sort_values("Match Score (%)", ascending=False))