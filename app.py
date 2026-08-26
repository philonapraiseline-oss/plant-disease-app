import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.2rem;
    }

    .card {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #dddddd;
        margin-bottom: 1rem;
        background: white;
    }

    .result-box {
        padding: 1.5rem;
        border-radius: 16px;
        border: 2px solid #4caf50;
        background: #f1f8e9;
        margin-top: 1rem;
    }

    .disease-title {
        font-size: 1.8rem;
        font-weight: bold;
    }

    .confidence {
        font-size: 1.3rem;
        font-weight: bold;
    }

    .small-text {
        color: #666666;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================
MODEL_URL = "https://huggingface.co/philona777/plant-disease-model/resolve/main/plant_disease_model.keras"

@st.cache_resource
def load_model():
    model_path = tf.keras.utils.get_file(
        "plant_disease_model.keras",
        MODEL_URL
    )
    return tf.keras.models.load_model(model_path)

model = load_model()

# =========================================================
# CLASS NAMES
# =========================================================
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# =========================================================
# DISEASE INFORMATION
# =========================================================
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "name": "Pepper Bacterial Spot",
        "description": "A bacterial disease that can cause spots and lesions on pepper leaves and fruit.",
        "tips": "Avoid overhead watering, remove severely affected plant material, and maintain good airflow."
    },

    "Pepper__bell___healthy": {
        "name": "Healthy Pepper Leaf",
        "description": "The model detected no major disease pattern in this pepper leaf.",
        "tips": "Continue good watering, sunlight, nutrition, and regular monitoring."
    },

    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "description": "A fungal disease that commonly produces dark spots on potato leaves.",
        "tips": "Remove affected leaves, avoid wetting foliage unnecessarily, and maintain good plant spacing."
    },

    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "description": "A serious disease of potato plants that can spread quickly under favorable conditions.",
        "tips": "Remove affected plant material and improve airflow. Monitor nearby plants carefully."
    },

    "Potato___healthy": {
        "name": "Healthy Potato Leaf",
        "description": "The model detected no major disease pattern in this potato leaf.",
        "tips": "Continue regular monitoring and maintain healthy growing conditions."
    },

    "Tomato_Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "description": "A bacterial disease that can produce small dark spots on tomato leaves and fruit.",
        "tips": "Avoid overhead watering and keep foliage dry when possible."
    },

    "Tomato_Early_blight": {
        "name": "Tomato Early Blight",
        "description": "A fungal disease that can cause characteristic dark lesions on tomato leaves.",
        "tips": "Remove affected leaves and improve airflow around the plant."
    },

    "Tomato_Late_blight": {
        "name": "Tomato Late Blight",
        "description": "A plant disease that can rapidly affect tomato foliage under suitable environmental conditions.",
        "tips": "Remove affected material and avoid prolonged leaf wetness."
    },

    "Tomato_Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "description": "A fungal disease often associated with humid conditions and poor airflow.",
        "tips": "Improve ventilation and avoid excessive humidity around the foliage."
    },

    "Tomato_Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "description": "A fungal disease that causes numerous small spots on tomato leaves.",
        "tips": "Remove affected leaves and avoid splashing water onto foliage."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Tomato Spider Mites",
        "description": "Tiny pests that can cause stippling, discoloration, and leaf damage.",
        "tips": "Inspect the underside of leaves regularly and maintain appropriate plant care."
    },

    "Tomato__Target_Spot": {
        "name": "Tomato Target Spot",
        "description": "A fungal disease that produces circular lesions on tomato leaves and other plant parts.",
        "tips": "Improve airflow and remove severely affected plant material."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "description": "A viral disease that can cause yellowing, curling, and reduced plant growth.",
        "tips": "Monitor for insect vectors such as whiteflies and remove severely affected plants where appropriate."
    },

    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "description": "A viral disease that can cause mottled or mosaic-like patterns on tomato leaves.",
        "tips": "Use clean gardening tools and avoid spreading plant sap between plants."
    },

    "Tomato_healthy": {
        "name": "Healthy Tomato Leaf",
        "description": "The model detected no major disease pattern in this tomato leaf.",
        "tips": "Continue regular monitoring, appropriate watering, sunlight, and plant nutrition."
    }
}

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("🌱 PlantCare AI")
st.sidebar.caption("AI-Powered Plant Disease Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Detect Disease",
        "📚 Disease Guide",
        "🤖 About Model",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Project")
st.sidebar.caption("Plant Disease Detection")

# =========================================================
# HOME
# =========================================================
if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>🌱 PlantCare AI</h1>
        <p>Intelligent Plant Disease Detection using Machine Learning</p>
        <p>Upload a leaf image and let the AI analyze it.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h2>🔍 Detect</h2>
            <p>Upload a plant leaf image and get an AI-powered prediction.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h2>📊 Analyze</h2>
            <p>View the predicted disease and the model's confidence score.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h2>🌿 Learn</h2>
            <p>Explore information and general care tips for detected conditions.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 🌾 Supported Plants")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("🌶️ Pepper")

    with c2:
        st.success("🥔 Potato")

    with c3:
        st.success("🍅 Tomato")

    st.markdown("---")

    st.info(
        "💡 Tip: For the best prediction, upload a clear image where the leaf "
        "is visible and well lit."
    )

# =========================================================
# DETECT DISEASE
# =========================================================
elif page == "🔍 Detect Disease":

    st.title("🔍 Plant Disease Detection")
    st.write(
        "Upload a clear image of a tomato, potato, or pepper leaf."
    )

    uploaded_file = st.file_uploader(
        "📷 Choose a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                image,
                caption="Uploaded Leaf",
                use_container_width=True
            )

        with col2:
            st.markdown("### 📋 Image Information")
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
            st.write("**Format:** RGB")

        st.markdown("---")

        if st.button("🔬 Analyze Leaf", type="primary", use_container_width=True):

            with st.spinner("🤖 AI is analyzing the leaf..."):

                img = image.resize((224, 224))
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)

                predictions = model.predict(
                    img_array,
                    verbose=0
                )

                predicted_index = np.argmax(predictions[0])
                predicted_class = class_names[predicted_index]
                confidence = float(
                    predictions[0][predicted_index] * 100
                )

            info = disease_info.get(
                predicted_class,
                {
                    "name": predicted_class.replace("_", " "),
                    "description": "The model identified this class.",
                    "tips": "Continue monitoring the plant and consider consulting a plant specialist."
                }
            )

            st.markdown("## 🎯 Detection Result")

            st.markdown(f"""
            <div class="result-box">
                <div class="disease-title">🌿 {info["name"]}</div>
                <br>
                <div class="confidence">
                    🎯 Confidence: {confidence:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(
                min(confidence / 100, 1.0)
            )

            if confidence >= 80:
                st.success("✅ High-confidence prediction")

            elif confidence >= 60:
                st.warning("⚠️ Moderate-confidence prediction")

            else:
                st.warning(
                    "⚠️ Low-confidence prediction. Try uploading a clearer leaf image."
                )

            st.markdown("### 📖 About This Result")
            st.write(info["description"])

            st.markdown("### 🌱 General Care Tips")
            st.info(info["tips"])

            st.caption(
                "Note: This AI prediction is for educational/project purposes "
                "and should not replace expert agricultural diagnosis."
            )

# =========================================================
# DISEASE GUIDE
# =========================================================
elif page == "📚 Disease Guide":

    st.title("📚 Disease Guide")

    st.write(
        "Explore the plant conditions included in the trained model."
    )

    selected_disease = st.selectbox(
        "Select a disease or plant condition",
        class_names
    )

    info = disease_info.get(
        selected_disease,
        {
            "name": selected_disease.replace("_", " "),
            "description": "Information is not available for this class.",
            "tips": "Monitor the plant and seek expert advice if necessary."
        }
    )

    st.markdown(f"## 🌿 {info['name']}")

    st.markdown(
        f"""
        <div class="card">
            <h3>📖 Description</h3>
            <p>{info['description']}</p>

            <h3>🌱 General Care</h3>
            <p>{info['tips']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# ABOUT MODEL
# =========================================================
elif page == "🤖 About Model":

    st.title("🤖 About the AI Model")

    st.write(
        "PlantCare AI uses a deep learning image classification model "
        "trained to recognize different plant leaf conditions."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🌿 Classes", "15")

    with col2:
        st.metric("🎯 Validation Accuracy", "91.3%")

    with col3:
        st.metric("🖼️ Input Size", "224 × 224")

    st.markdown("---")

    st.subheader("🧠 How It Works")

    st.markdown("""
    **1. 📷 Upload Image**

    The user uploads a leaf image.

    **2. 🔄 Image Processing**

    The image is converted to RGB and resized to 224 × 224 pixels.

    **3. 🧠 Deep Learning Model**

    The trained neural network analyzes visual patterns in the leaf.

    **4. 🎯 Prediction**

    The model selects the class with the highest predicted probability.

    **5. 📊 Result**

    The application displays the predicted condition and confidence score.
    """)

    st.markdown("---")

    st.subheader("🌿 Supported Categories")

    for class_name in class_names:
        st.write("•", class_name.replace("_", " "))

# =========================================================
# ABOUT PROJECT
# =========================================================
elif page == "ℹ️ About Project":

    st.title("ℹ️ About PlantCare AI")

    st.markdown("""
    ### 🎯 Project Objective

    The goal of this project is to develop a machine learning application
    capable of identifying plant leaf diseases from images.

    ### 🛠️ Technologies Used

    - 🐍 Python
    - 🧠 TensorFlow / Keras
    - 📊 NumPy
    - 🖼️ Pillow
    - 🎨 Streamlit
    - 🤗 Hugging Face
    - 💻 GitHub

    ### 🌱 Application Features

    - Leaf image upload
    - AI-based disease classification
    - Confidence score
    - Disease information
    - General care tips
    - Model information
    - User-friendly interface

    ### 🚀 Deployment

    The application is deployed using Streamlit and the trained model
    is hosted through Hugging Face.
    """)

    st.markdown("---")

    st.success(
        "🌱 PlantCare AI — Turning machine learning into a practical "
        "plant-health tool."
    )
