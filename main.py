import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.metrics.pairwise import cosine_similarity
from database import get_features  
import numpy as np
import cv2
import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
model = load_model('feature_extractor.keras')
feature_extractor = Model(inputs=model.input, outputs=model.output)


def preprocess_image(image):
    img_array = np.array(image) / 255.0
    img_array = tf.image.resize(img_array, (224, 224))  
    img_array = tf.expand_dims(img_array, axis=0)  
    return img_array

def find_similar(query_feature, top_n=5):
   
    json_data = get_features()  
    paths = [item['path'] for item in json_data]
    features = np.array([item['features'] for item in json_data])

    query_feature = query_feature.reshape(-1)  
    query_feature = np.expand_dims(query_feature, axis=0)  


    similarities = cosine_similarity(query_feature, features).flatten()

    top_indices = similarities.argsort()[-top_n:][::-1]
    results = [
        {"path": paths[idx], "similarity": similarities[idx]} for idx in top_indices
    ]
    return results

st.title("Image Similarity Finder")
st.write("Upload an image to find the most similar items from the database.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing the image...")

    file_bytes = uploaded_file.read()
    np_image = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    preprocessed_image = preprocess_image(img)
    
    features = feature_extractor.predict(preprocessed_image).flatten()


    st.write("Finding similar items...")
    top_matches = find_similar(features, top_n=5)


    st.write("Most similar items:")
    for match in top_matches:
        st.write(f"Path: {match['path']}, Similarity: {match['similarity']:.4f}")
        st.image(match['path'], caption=f"Similarity: {match['similarity']:.4f}", use_column_width=True)