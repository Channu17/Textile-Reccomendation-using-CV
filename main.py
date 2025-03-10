import os
import requests
import numpy as np
import streamlit as st
import tensorflow as tf
import sqlite3
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image  # Replacing OpenCV with PIL
from sklearn.metrics.pairwise import cosine_similarity
from database import get_features, insert_features, create_table  

DB_NAME = "textile.db"
BASE_URL = "https://anandbrothersmysuru.in"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model = load_model('feature_extractor.keras')
feature_extractor = Model(inputs=model.input, outputs=model.output)

def preprocess_image(image):
    """Preprocess image for model input."""
    img_array = np.array(image) / 255.0
    img_array = tf.image.resize(img_array, (224, 224))
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

def extract_features_from_url(image_url):
    """Fetch image from URL and extract features using the model."""
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            features = feature_extractor.predict(img_array).flatten()
            return features
        else:
            print(f"Error: Unable to fetch image from {image_url}")
            return None
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return None

def find_similar(query_feature, top_n=5):
    """Find the most similar images in the database using cosine similarity."""
    json_data = get_features()
    if not json_data:
        return []

    links = [item.get('link', '') for item in json_data]  
    features = np.array([item['features'] for item in json_data])

    query_feature = query_feature.reshape(1, -1)
    similarities = cosine_similarity(query_feature, features).flatten()

    top_indices = similarities.argsort()[-top_n:][::-1]
    results = [{"link": links[idx], "similarity": similarities[idx]} for idx in top_indices]
    
    return results

def reset_and_update_db():
    """Delete existing database, create a new one, and fetch fresh product data."""
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)  
        print("Existing database deleted.")

    create_table()  
    print("New database created.")

    page = 1
    while True:
        response = requests.get(f"{BASE_URL}/api/v1/products?page={page}")
        
        try:
            json_data = response.json()
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            break

        if not json_data.get('products'):
            break

        for product in json_data['products']:
            all_images = [product.get('heroImage', '')] + product.get('otherImages', [])
            
            for img_path in all_images:
                if img_path:
                    image_url = f"{BASE_URL}/{img_path}"
                    print(f"Processing image: {image_url}")
                    features = extract_features_from_url(image_url)
                    
                    if features is not None:
                        insert_features(image_url, features)  
                        print(f"Features stored for {image_url}")

        page += 1

st.title("Cloth Recommendation System")
st.write("Upload a cloth image to find the most similar clothes from the database.")

if st.button("Reset & Update Database"):
    st.write("Resetting and updating the database. Please wait...")
    reset_and_update_db()
    st.success("Database reset and updated successfully!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing the image...")

    # Using PIL to open the uploaded image
    img = Image.open(uploaded_file).convert("RGB")

    # Preprocess the image for the model
    preprocessed_image = preprocess_image(img)
    features = feature_extractor.predict(preprocessed_image).flatten()

    st.write("Finding similar items...")
    top_matches = find_similar(features, top_n=5)

    if top_matches:
        st.write("Most similar items:")
        for match in top_matches:
            st.image(match['link'], caption=f"Similarity: {match['similarity']:.4f}", use_column_width=True)
    else:
        st.write("No similar images found.")

