# --- Streamlit App UI (PROFESSIONAL VERSION) ---
st.set_page_config(page_title="Campus Lost & Found", layout="wide")
st.title("🏫 Campus Lost & Found - Automated ML Matching System")
st.markdown("### Multi-Modal Text + Image Matching Using Traditional Machine Learning")

# Initialize manual TF-IDF and dataset
tfidf = ManualTFIDF()
dataset = get_dataset()
all_texts = LOST_ITEMS + dataset["desc"].tolist()
tfidf.fit(all_texts)

# --- User Input Section ---
st.subheader("Report a Lost Item")
selected_lost_item = st.selectbox("Select your lost item:", LOST_ITEMS)
uploaded_image = st.file_uploader(f"Upload photo of your {selected_lost_item}:", type=["png", "jpg", "jpeg"])

# --- Matching Logic ---
if selected_lost_item:
    # Calculate text similarity using manual TF-IDF
    query_vec = tfidf.transform([selected_lost_item])[0]
    item_vecs = tfidf.transform(dataset["desc"].tolist())
    text_similarity_scores = [cosine_similarity(query_vec, vec) for vec in item_vecs]
    
    # Calculate image similarity (Pillow only, no CV2)
    image_similarity_scores = np.zeros(len(dataset))
    if uploaded_image is not None:
        st.image(Image.open(uploaded_image), caption="Uploaded Lost Item", width=200)
        try:
            img = Image.open(uploaded_image).convert('RGB')
            avg_color = np.mean(np.array(img.resize((32,32))), axis=(0,1)) / 255
            image_similarity_scores = [1 - min(np.linalg.norm(avg_color - c), 1.0) for c in dataset["color"]]
            st.success("✅ Image features extracted successfully! Match score updated.")
        except Exception as e:
            st.error(f"❌ Failed to process image: {str(e)}")
    
    # Combine scores
    final_scores = (0.6 * np.array(text_similarity_scores)) + (0.4 * np.array(image_similarity_scores))

    # --- Display Results ---
    st.subheader("Top 10 Found Item Matches")
    results = dataset.copy()
    results["Match Score (%)"] = final_scores
    results = results.sort_values("Match Score (%)", ascending=False).reset_index(drop=True)
    
    st.dataframe(
        results[["desc", "Match Score (%)"]].style.format({"Match Score (%)": "{:.1%}"}),
        use_container_width=True,
        height=500
    )

# --- Final Footer (Professional) ---
st.markdown("---")
st.info("💡 This system uses manually implemented TF-IDF for text matching and color profile analysis for image matching, ensuring scalability and deployment compatibility.")
