import streamlit as st
import pandas as pd
import io
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset
from streamlit_option_menu import option_menu

# Set the page configuration first

def recommendation_page(df, dataset, model, index):
    st.title("🔍 Fashion Recommender System")

    # Sidebar filters
    with st.sidebar:
        st.header("🎯 Filters")
        gender = st.selectbox("Choose Gender:", sorted(df["gender"].dropna().unique()))
        base_colour = st.selectbox("Choose Base Colour:", sorted(df["baseColour"].dropna().unique()))
        season = st.selectbox("Choose Season:", sorted(df["season"].dropna().unique()))

    # Generate query embedding
    query_text = f"{season} {base_colour} {gender}"
    query_embedding = model.encode([query_text], convert_to_numpy=True)

    # Retrieve recommendations
    distance, index_result = index.search(query_embedding, k=20)
    matched_items = df.iloc[index_result[0]]

    # Filter results by exact match
    filtered_items = matched_items[
        (matched_items['gender'] == gender) &
        (matched_items['baseColour'] == base_colour) &
        (matched_items['season'] == season)
    ]

    # Resize image function
    def resize_image(image, size=(300, 300)):
        return ImageOps.fit(image, size, Image.LANCZOS)

    # Display recommendations
    st.subheader("🛍️ Recommended Outfits")
    if not filtered_items.empty:
        cols = st.columns(3)
        for idx, (_, item) in enumerate(filtered_items.iterrows()):
            with cols[idx % 3]:
                st.write(f"**{item['gender'].capitalize()} | {item['baseColour'].capitalize()} | {item['season'].capitalize()}**")
                st.write(f"👚 Category: {item['masterCategory']} > {item['subCategory']} > {item['articleType']}")

                # Get the image from the dataset object
                try:
                    image = dataset[item.name]['image']  # item.name is the index in the original dataset
                    st.image(image, use_column_width=True)

                    # If you want to provide download functionality:
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    byte_im = buf.getvalue()

                    st.download_button(
                        label="⬇ Download Image",
                        data=byte_im,
                        file_name=f"fashion_{idx+1}.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.warning(f"⚠ Unable to load image: {e}")
    else:
        st.warning("No exact matches found. Try adjusting the filters.")
