import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Lazy Landmark Finder", page_icon="📸", layout="wide")

# --- Model Loading ---
@st.cache_resource
def load_my_model():
    """Loads and returns the trained Keras model."""
    model_path = os.path.join('saved_model', 'landmark_model.h5')
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Image Preprocessing ---
def preprocess_image(image_pil):
    """
    Takes a PIL image, resizes it, normalizes it, and prepares it for the model.
    """
    image_resized = image_pil.resize((224, 224))
    image_array = np.array(image_resized)
    
    if image_array.shape[2] == 4: # Handle RGBA images
        image_array = image_array[:, :, :3]
        
    image_normalized = image_array / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded

# --- Main App Logic ---
model = load_my_model()

try:
    train_dir = os.path.join('data', 'train')
    class_names = sorted(os.listdir(train_dir))
except FileNotFoundError:
    st.error("Error: 'data/train' directory not found. Please make sure the data is set up correctly.")
    class_names = []


# --- Streamlit User Interface ---
st.title("📸 Lazy Landmark Finder")
st.markdown("Upload a photo of a landmark—even a lazy, bad-angle one—and I'll try to guess what it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if model and class_names and uploaded_file is not None:
    # Create two columns for layout
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Your Uploaded Image', use_column_width=True)
    
    with col2:
        with st.spinner('Classifying...'):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)[0] # Get the prediction array
            
            # ## NEW FEATURE ##: Get top 3 predictions
            top_k = 3
            top_indices = np.argsort(prediction)[-top_k:][::-1] # Get indices of top 3 scores
            
            top_predictions = []
            for i in top_indices:
                class_name = class_names[i].replace("_", " ").title()
                confidence = prediction[i] * 100
                top_predictions.append({"Landmark": class_name, "Confidence": confidence})

        st.subheader("Top Predictions")

        # ## NEW FEATURE ##: Display top predictions in a more structured way
        for p in top_predictions:
            st.write(f"**{p['Landmark']}**: {p['Confidence']:.1f}%")

        # ## NEW FEATURE ##: Display a bar chart of the top predictions
        df_predictions = pd.DataFrame(top_predictions)
        df_predictions.set_index('Landmark', inplace=True)
        st.bar_chart(df_predictions['Confidence'])

        # ## NEW FEATURE ##: Add a message for low confidence
        top_confidence = top_predictions[0]['Confidence']
        if top_confidence < 50:
            st.warning("🤔 I'm not very confident about this one. You might have found a truly lazy photo or a new landmark!")

st.markdown("---")
st.write("Built with TensorFlow & Streamlit.")