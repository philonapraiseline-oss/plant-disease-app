
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load model
model = tf.keras.models.load_model("plant_disease_model.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# App title
st.title("🌱 Plant Disease Detection")
st.write("Upload a tomato, potato, or pepper leaf image to detect its disease.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    # Prepare image
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # Display result
    st.subheader("Prediction")

    st.success(
        f"Disease: {predicted_class.replace('_', ' ')}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )
