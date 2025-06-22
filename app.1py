import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import io
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from streamlit_option_menu import option_menu

from ar_tryon import virtual_tryon_page
from recommender import recommendation_page

# Streamlit app settings
st.set_page_config(page_title="Fashion App", layout="wide")

# Navigation Menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Recommendation System", "AR Try-On"],
    icons=["house", "search", "camera"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Load Dataset and Model Only Once
@st.cache_resource
def load_data_and_model():
    dataset = load_dataset("ashraq/fashion-product-images-small", split="train")
    df = dataset.to_pandas().fillna("")
    df["Text"] = df["season"] + " " + df["baseColour"] + " " + df["gender"]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["Text"].astype(str).tolist(), convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return dataset, df, model, embeddings, index

dataset, df, model, embeddings, index = load_data_and_model()

# Pages
if selected == "Home":
    st.title("👗 Welcome to the Fashion Recommender & AR Try-On App")
    st.markdown("""
    Discover recommended outfits based on season, color, and gender preferences,  
    or try out the virtual fitting room with your own image!
    
    👈 Use the tabs above to navigate.
    """)
elif selected == "Recommendation System":
    recommendation_page(df, dataset, model, index)
elif selected == "AR Try-On":
    virtual_tryon_page()
