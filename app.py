import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Title
st.image("C:/Users/Jayesh/OneDrive/Desktop/Skin Care Ai/logo.png", width=150)
st.title("🧴 FaceWise Ai")
st.markdown("### Your Smart Skin Problem Detector 💡")
st.write("Upload a close-up face image with a skin issue, and FaceWise AI will identify the condition.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("C:/Users/Jayesh/OneDrive/Desktop/Skin Care Ai/mobilenet_skin_model.h5")
    return model

model = load_model()
st.write("Model loaded ✅")
st.write("🔢 Model output shape:", model.output_shape)


# Define your skin issue labels (adjust based on your training)
import os

DATA_DIR = "C:/Users/Jayesh/OneDrive/Desktop/Skin Care Ai/files"
class_names = sorted(os.listdir(DATA_DIR))  # This will be ['acne', 'pigmentation', 'redbags']
st.write("📂 Classes loaded:", class_names)


# Image Preprocessing and Prediction
def process_and_predict(image_data, model):
    img = image_data.convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)

    st.write("Raw predictions:", predictions)
    st.write("Predictions shape:", predictions.shape)
    st.write("Number of classes (model output):", predictions.shape[1])

    if predictions.shape[1] != len(class_names):
        raise ValueError(f"Mismatch between model output size ({predictions.shape[1]}) and number of class names ({len(class_names)}).")

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence


# Upload Section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='📷 Uploaded Image', use_container_width=True)


    st.write("🔍 Predicting...")
    try:
        label, confidence = process_and_predict(img, model)
        st.success(f"🧠 Prediction: **{label}** ({confidence:.2f}% confidence)")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {str(e)}")
# Footer
st.markdown("© 2025 FaceWise AI | Built with ❤️ using Streamlit and TensorFlow")
