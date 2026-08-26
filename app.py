import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# MODERN CSS
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f7f9f8;
}

/* Hide Streamlit default elements */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e8ece9;
}

section[data-testid="stSidebar"] h1 {
    font-weight: 800;
}

/* Main headings */
h1, h2, h3 {
    color: #111814;
    letter-spacing: -0.5px;
}

/* Hero */
.hero {
    padding: 60px 50px;
    border-radius: 28px;
    background: linear-gradient(135deg, #e9f7ef 0%, #f7fbf8 55%, #e4f4ea 100%);
    border: 1px solid #d9eadf;
    margin-bottom: 30px;
}

.hero-label {
    color: #23864b;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.hero-title {
    font-size: 48px;
    line-height: 1.08;
    font-weight: 800;
    margin: 12px 0;
    color: #102018;
}

.hero-text {
    font-size: 18px;
    line-height: 1.6;
    color: #53635a;
    max-width: 700px;
}

/* Cards */
.card {
    background: #ffffff;
    border: 1px solid #e5ebe7;
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.03);
}

.card-title {
    font-size: 19px;
    font-weight: 700;
    color: #16231b;
}

.card-text {
    color: #65736a;
    line-height: 1.6;
}

/* Upload area */
.upload-card {
    background: #ffffff;
    border: 2px dashed #b8d9c2;
    border-radius: 24px;
    padding: 35px;
    text-align: center;
    margin: 20px 0;
}

/* Result */
.result-card {
    background: #ffffff;
    border-radius: 24px;
    border: 1px solid #dce8df;
    padding: 30px;
    margin-top: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
}

.result-label {
    font-size: 13px;
    color: #6b7b71;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
}

.result-name {
    font-size: 30px;
    font-weight: 800;
    color: #17251c;
    margin-top: 6px;
}

.confidence-number {
    font-size: 38px;
    font-weight: 800;
    color: #23864b;
}

/* Stats */
.stat-card {
    background: #ffffff;
    border: 1px solid #e5ebe7;
    border-radius: 18px;
    padding: 22px;
    text-align: center;
}

.stat-number {
    font-size: 30px;
    font-weight: 800;
    color: #23864b;
}

.stat-label {
    font-size: 13px;
    color: #69766e;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    min-height: 48px;
}

/* Footer */
.app-footer {
    text-align: center;
    color: #7a857e;
    padding: 35px 0 10px;
    font-size: 13px;
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

    "Pepper__bell___Bacterial_spot": (
        "Pepper Bacterial Spot",
        "A bacterial disease that can cause spots and lesions on pepper leaves and fruit.",
        "Improve airflow, avoid unnecessary leaf wetness, and remove severely affected plant material."
    ),

    "Pepper__bell___healthy": (
        "Healthy Pepper Leaf",
        "The model detected a healthy pepper leaf.",
        "Continue regular watering, sunlight, nutrition, and monitoring."
    ),

    "Potato___Early_blight": (
        "Potato Early Blight",
        "A fungal disease that commonly produces dark spots on potato leaves.",
        "Remove affected leaves and maintain good spacing and airflow."
    ),

    "Potato___Late_blight": (
        "Potato Late Blight",
        "A serious disease that can spread rapidly under favorable conditions.",
        "Remove affected plant material and avoid prolonged leaf wetness."
    ),

    "Potato___healthy": (
        "Healthy Potato Leaf",
        "The model detected a healthy potato leaf.",
        "Continue normal plant care and monitor regularly."
    ),

    "Tomato_Bacterial_spot": (
        "Tomato Bacterial Spot",
        "A bacterial disease that can produce dark spots on tomato leaves and fruit.",
        "Avoid overhead watering and keep foliage dry when possible."
    ),

    "Tomato_Early_blight": (
        "Tomato Early Blight",
        "A fungal disease that can cause dark lesions on tomato foliage.",
        "Remove affected leaves and improve airflow around plants."
    ),

    "Tomato_Late_blight": (
        "Tomato Late Blight",
        "A disease that can rapidly affect tomato plants under suitable conditions.",
        "Remove affected material and minimize prolonged leaf wetness."
    ),

    "Tomato_Leaf_Mold": (
        "Tomato Leaf Mold",
        "A fungal disease often associated with humid conditions and poor airflow.",
        "Improve ventilation and avoid excessive humidity around foliage."
    ),

    "Tomato_Septoria_leaf_spot": (
        "Tomato Septoria Leaf Spot",
        "A fungal disease that produces numerous small spots on tomato leaves.",
        "Remove affected leaves and avoid splashing water onto foliage."
    ),

    "Tomato_Spider_mites_Two_spotted_spider_mite": (
        "Tomato Spider Mites",
        "Tiny pests that can cause stippling and discoloration of leaves.",
        "Inspect the underside of leaves regularly and monitor plants closely."
    ),

    "Tomato__Target_Spot": (
        "Tomato Target Spot",
        "A fungal disease that produces circular lesions on tomato leaves.",
        "Improve airflow and remove severely affected plant material."
    ),

    "Tomato__Tomato_YellowLeaf__Curl_Virus": (
        "Tomato Yellow Leaf Curl Virus",
        "A viral disease that can cause yellowing, curling, and reduced growth.",
        "Monitor insect vectors such as whiteflies and remove severely affected plants where appropriate."
    ),

    "Tomato__Tomato_mosaic_virus": (
        "Tomato Mosaic Virus",
        "A viral disease that can produce mottled or mosaic-like leaf patterns.",
        "Keep tools clean and avoid transferring plant sap between plants."
    ),

    "Tomato_healthy": (
        "Healthy Tomato Leaf",
        "The model detected a healthy tomato leaf.",
        "Continue regular monitoring, watering, sunlight, and plant nutrition."
    )
}

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("# 🌿 PlantCare AI")
st.sidebar.caption("AI-powered plant health assistant")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MENU",
    [
        "Home",
        "Detect Disease",
        "Disease Library",
        "About Model",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption("Powered by TensorFlow")
st.sidebar.caption("Plant Disease Classification")

# =========================================================
# HOME
# =========================================================

if page == "Home":

    st.markdown("""
    <div class="hero">

        <div class="hero-label">
        AI-POWERED PLANT HEALTH
        </div>

        <div class="hero-title">
        Know your plant.<br>
        Protect your harvest.
        </div>

        <div class="hero-text">
        PlantCare AI analyzes leaf images using deep learning
        to identify common diseases in tomato, potato, and pepper plants.
        </div>

    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">15</div>
            <div class="stat-label">Disease Classes</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">91.3%</div>
            <div class="stat-label">Validation Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">3</div>
            <div class="stat-label">Plant Types</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##")

    col1, col2 = st.columns([1.3, 1])

    with col1:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        🌱 How PlantCare AI works
        </div>

        <br>

        <div class="card-text">
        <b>01 — Upload</b><br>
        Upload a clear photo of a plant leaf.
        <br><br>

        <b>02 — Analyze</b><br>
        Our trained deep learning model analyzes the image.
        <br><br>

        <b>03 — Understand</b><br>
        Receive a predicted condition, confidence score,
        and general care information.
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        🔬 Supported plants
        </div>

        <br>

        <div class="card-text">
        🍅 Tomato<br><br>
        🥔 Potato<br><br>
        🌶️ Pepper
        </div>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.info(
        "For best results, use a clear, well-lit image where the leaf "
        "is clearly visible."
    )

# =========================================================
# DETECT
# =========================================================

elif page == "Detect Disease":

    st.title("Detect Plant Disease")

    st.write(
        "Upload a leaf image and let PlantCare AI analyze it."
    )

    uploaded_file = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is None:

        st.markdown("""
        <div class="upload-card">

        <div style="font-size:50px;">📷</div>

        <h3>Upload a leaf image</h3>

        <p style="color:#68766e;">
        JPG, JPEG or PNG • Clear images work best
        </p>

        </div>
        """, unsafe_allow_html=True)

    else:

        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("### Preview")

        col1, col2 = st.columns([1.1, 1])

        with col1:

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.markdown("""
            <div class="card">

            <div class="card-title">
            Image ready
            </div>

            <br>

            <div class="card-text">
            Your leaf image is ready to be analyzed.
            <br><br>
            <b>Recommended:</b><br>
            • Clear leaf<br>
            • Good lighting<br>
            • Minimal background distraction
            </div>

            </div>
            """, unsafe_allow_html=True)

            analyze = st.button(
                "🔬 Analyze Leaf",
                type="primary",
                use_container_width=True
            )

            if analyze:

                with st.spinner("Analyzing leaf with AI..."):

                    img = image.resize((224, 224))

                    img_array = np.array(img)

                    img_array = np.expand_dims(
                        img_array,
                        axis=0
                    )

                    predictions = model.predict(
                        img_array,
                        verbose=0
                    )

                    predicted_index = np.argmax(
                        predictions[0]
                    )

                    predicted_class = class_names[
                        predicted_index
                    ]

                    confidence = float(
                        predictions[0][predicted_index] * 100
                    )

                name, description, tips = disease_info.get(
                    predicted_class,
                    (
                        predicted_class.replace("_", " "),
                        "The model identified this condition.",
                        "Monitor the plant and consider expert advice."
                    )
                )

                st.markdown("""
                <div class="result-card">

                <div class="result-label">
                DETECTION COMPLETE
                </div>

                <div class="result-name">
                🌿 %s
                </div>

                <br>

                <div class="result-label">
                MODEL CONFIDENCE
                </div>

                <div class="confidence-number">
                %.2f%%
                </div>

                </div>
                """ % (name, confidence), unsafe_allow_html=True)

                st.progress(
                    min(confidence / 100, 1.0)
                )

                if confidence >= 80:
                    st.success(
                        "High-confidence prediction"
                    )
                elif confidence >= 60:
                    st.warning(
                        "Moderate-confidence prediction"
                    )
                else:
                    st.warning(
                        "Low-confidence prediction. "
                        "Try a clearer image."
                    )

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown("""
                    <div class="card">

                    <div class="card-title">
                    📖 What this means
                    </div>

                    <br>

                    <div class="card-text">
                    %s
                    </div>

                    </div>
                    """ % description, unsafe_allow_html=True)

                with col2:

                    st.markdown("""
                    <div class="card">

                    <div class="card-title">
                    🌱 General care
                    </div>

                    <br>

                    <div class="card-text">
                    %s
                    </div>

                    </div>
                    """ % tips, unsafe_allow_html=True)

                st.caption(
                    "This prediction is intended for educational and "
                    "project purposes and should not replace professional "
                    "agricultural diagnosis."
                )

# =========================================================
# DISEASE LIBRARY
# =========================================================

elif page == "Disease Library":

    st.title("Disease Library")

    st.write(
        "Explore the plant conditions included in the model."
    )

    search = st.text_input(
        "Search diseases",
        placeholder="Search tomato, potato, bacterial..."
    )

    filtered = []

    for disease in class_names:

        display_name = disease.replace("_", " ")

        if search.lower() in display_name.lower():
            filtered.append(disease)

    if not filtered:
        st.warning("No matching disease found.")

    for disease in filtered:

        name, description, tips = disease_info.get(
            disease,
            (
                disease.replace("_", " "),
                "Information unavailable.",
                "Monitor the plant."
            )
        )

        with st.expander(name):

            st.write(description)

            st.markdown("**General care:**")

            st.info(tips)

# =========================================================
# ABOUT MODEL
# =========================================================

elif page == "About Model":

    st.title("About the AI Model")

    st.write(
        "PlantCare AI uses a deep learning image classification "
        "model trained on plant leaf images."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Classes", "15")

    with col2:
        st.metric("Validation Accuracy", "91.3%")

    with col3:
        st.metric("Image Size", "224 × 224")

    st.markdown("---")

    st.subheader("Model Pipeline")

    steps = [
        ("01", "Image Upload", "User provides a plant leaf image."),
        ("02", "Preprocessing", "Image is converted to RGB and resized."),
        ("03", "Deep Learning", "The trained neural network analyzes visual patterns."),
        ("04", "Classification", "The model selects the most likely class."),
        ("05", "Result", "Prediction and confidence are displayed.")
    ]

    for number, title, description in steps:

        st.markdown(f"""
        <div class="card">

        <b>{number} — {title}</b>

        <p class="card-text">
        {description}
        </p>

        </div>
        """, unsafe_allow_html=True)

# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.title("About PlantCare AI")

    st.markdown("""
    ## 🎯 Project Goal

    PlantCare AI is a machine learning project designed to identify
    common plant diseases from leaf images.

    The system demonstrates how deep learning and computer vision
    can be combined with a web application to create an accessible
    plant-health analysis tool.

    ## 🛠️ Technology Stack

    **Machine Learning**
    - TensorFlow
    - Keras
    - NumPy

    **Application**
    - Streamlit
    - Pillow

    **Deployment**
    - GitHub
    - Hugging Face
    - Streamlit

    ## 🌿 Supported Plants

    🍅 Tomato  
    🥔 Potato  
    🌶️ Pepper

    ## 📊 Model Performance

    The trained model achieved approximately **91.3% validation accuracy**
    during development.

    """)

    st.markdown("---")

    st.markdown("""
    <div class="app-footer">
    🌿 PlantCare AI<br>
    AI-powered plant disease detection
    </div>
    """, unsafe_allow_html=True)
