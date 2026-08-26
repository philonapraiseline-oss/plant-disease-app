import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PROFESSIONAL DESIGN
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f0fff7 0%, #f8fbff 50%, #eefcf5 100%);
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #063b2b 0%, #087443 55%, #0ca678 100%);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.brand {
    text-align: center;
    padding: 20px 5px 15px 5px;
}

.brand-icon {
    font-size: 48px;
}

.brand-title {
    font-size: 25px;
    font-weight: 800;
    color: white;
}

.brand-subtitle {
    font-size: 12px;
    color: #d5f8e6;
}

.hero {
    background: linear-gradient(135deg, #063b2b, #087443, #10b981);
    padding: 55px;
    border-radius: 30px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 15px 45px rgba(6, 59, 43, 0.20);
}

.hero-tag {
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 2px;
    color: #b9f6d5;
}

.hero-title {
    font-size: 52px;
    line-height: 1.05;
    font-weight: 800;
    margin: 15px 0;
}

.hero-text {
    font-size: 18px;
    line-height: 1.6;
    max-width: 800px;
    color: #e7fff2;
}

.card {
    background: rgba(255,255,255,0.96);
    border: 1px solid #d9eee3;
    border-radius: 22px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 8px 30px rgba(20, 70, 45, 0.07);
}

.card-title {
    color: #063b2b;
    font-size: 21px;
    font-weight: 800;
}

.card-text {
    color: #52665c;
    line-height: 1.6;
}

.stat-card {
    background: white;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    border: 1px solid #dcece4;
    box-shadow: 0 8px 25px rgba(0,0,0,0.05);
}

.stat-number {
    color: #087443;
    font-size: 34px;
    font-weight: 800;
}

.stat-label {
    color: #718078;
    font-size: 13px;
    margin-top: 5px;
}

.result-card {
    background: linear-gradient(135deg, #ffffff, #effcf5);
    border: 2px solid #9be2bc;
    border-radius: 25px;
    padding: 30px;
    margin: 20px 0;
    box-shadow: 0 12px 35px rgba(8, 116, 67, 0.10);
}

.result-title {
    color: #073b2b;
    font-size: 32px;
    font-weight: 800;
}

.confidence {
    color: #087443;
    font-size: 42px;
    font-weight: 800;
}

.feature {
    background: white;
    border-radius: 20px;
    padding: 25px;
    min-height: 180px;
    border: 1px solid #dcece4;
    box-shadow: 0 8px 25px rgba(0,0,0,0.05);
}

.feature-icon {
    font-size: 35px;
}

.feature-title {
    font-size: 20px;
    font-weight: 800;
    color: #063b2b;
    margin-top: 10px;
}

.feature-text {
    color: #66756d;
    line-height: 1.5;
}

.info-box {
    background: #ecfdf5;
    border-left: 5px solid #10b981;
    border-radius: 12px;
    padding: 18px;
    margin: 12px 0;
}

.warning-box {
    background: #fff8e6;
    border-left: 5px solid #e5a400;
    border-radius: 12px;
    padding: 18px;
    margin: 12px 0;
}

.chat-user {
    background: #087443;
    color: white;
    padding: 16px 20px;
    border-radius: 20px 20px 5px 20px;
    margin: 15px 0 15px 15%;
}

.chat-ai {
    background: white;
    color: #26382f;
    border: 1px solid #dcebe3;
    padding: 18px 20px;
    border-radius: 20px 20px 20px 5px;
    margin: 15px 15% 15px 0;
    box-shadow: 0 5px 20px rgba(0,0,0,0.04);
}

.section-title {
    color: #063b2b;
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 8px;
}

.section-subtitle {
    color: #66756d;
    font-size: 16px;
    margin-bottom: 25px;
}

.stButton > button {
    border-radius: 14px;
    min-height: 48px;
    font-weight: 700;
}

div[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #8ed7ad;
    border-radius: 20px;
    padding: 15px;
}

.footer-custom {
    text-align: center;
    padding: 35px 0 15px 0;
    color: #718078;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================
MODEL_URL = (
    "https://huggingface.co/philona777/plant-disease-model/"
    "resolve/main/plant_disease_model.keras"
)


@st.cache_resource
def load_model():
    model_path = tf.keras.utils.get_file(
        "plant_disease_model.keras",
        MODEL_URL
    )
    return tf.keras.models.load_model(model_path)


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
        "plant": "Pepper",
        "type": "Bacterial disease",
        "description": "A bacterial disease that can cause dark spots and lesions on pepper leaves and fruit.",
        "symptoms": "Small dark spots, leaf lesions and damaged fruit.",
        "solution": "Remove badly affected material, improve airflow and avoid keeping leaves wet.",
        "prevention": "Avoid overhead watering, maintain plant spacing and clean gardening tools."
    },

    "Pepper__bell___healthy": {
        "name": "Healthy Pepper",
        "plant": "Pepper",
        "type": "Healthy",
        "description": "The model detected a healthy pepper leaf pattern.",
        "symptoms": "No major disease pattern detected.",
        "solution": "Continue normal plant care and monitor the plant.",
        "prevention": "Provide sunlight, appropriate watering and good nutrition."
    },

    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "plant": "Potato",
        "type": "Fungal disease",
        "description": "A fungal disease that commonly produces dark lesions on potato leaves.",
        "symptoms": "Dark circular spots, yellowing and leaf damage.",
        "solution": "Remove severely affected leaves and improve airflow around plants.",
        "prevention": "Avoid prolonged leaf wetness and maintain good plant spacing."
    },

    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "plant": "Potato",
        "type": "Disease",
        "description": "A serious disease that can spread rapidly under favorable environmental conditions.",
        "symptoms": "Dark irregular leaf lesions, discoloration and rapid plant decline.",
        "solution": "Remove affected material, reduce prolonged leaf wetness and seek local agricultural advice for serious outbreaks.",
        "prevention": "Use healthy planting material, good airflow and regular monitoring."
    },

    "Potato___healthy": {
        "name": "Healthy Potato",
        "plant": "Potato",
        "type": "Healthy",
        "description": "The model detected a healthy potato leaf pattern.",
        "symptoms": "No major disease pattern detected.",
        "solution": "Continue normal plant care.",
        "prevention": "Monitor regularly and maintain suitable watering and nutrition."
    },

    "Tomato_Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "plant": "Tomato",
        "type": "Bacterial disease",
        "description": "A bacterial disease that can affect tomato leaves and fruit.",
        "symptoms": "Small dark spots and lesions on leaves or fruit.",
        "solution": "Remove severely affected material and avoid unnecessarily wetting foliage.",
        "prevention": "Use clean tools, maintain airflow and avoid overhead watering."
    },

    "Tomato_Early_blight": {
        "name": "Tomato Early Blight",
        "plant": "Tomato",
        "type": "Fungal disease",
        "description": "A fungal disease that can cause dark lesions on tomato foliage.",
        "symptoms": "Dark circular lesions, yellowing and leaf drop.",
        "solution": "Remove affected leaves and improve airflow.",
        "prevention": "Avoid prolonged leaf wetness and maintain plant spacing."
    },

    "Tomato_Late_blight": {
        "name": "Tomato Late Blight",
        "plant": "Tomato",
        "type": "Disease",
        "description": "A disease that can spread rapidly in suitable environmental conditions.",
        "symptoms": "Dark irregular patches and rapid leaf damage.",
        "solution": "Remove affected material and minimize prolonged leaf wetness.",
        "prevention": "Monitor frequently and maintain good airflow."
    },

    "Tomato_Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "plant": "Tomato",
        "type": "Fungal disease",
        "description": "A fungal disease commonly associated with humid conditions.",
        "symptoms": "Yellow areas on leaves and mold-like growth underneath.",
        "solution": "Improve ventilation and remove severely affected leaves.",
        "prevention": "Reduce humidity around foliage and improve airflow."
    },

    "Tomato_Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "plant": "Tomato",
        "type": "Fungal disease",
        "description": "A fungal disease producing many small spots on tomato leaves.",
        "symptoms": "Small circular spots, often with darker edges.",
        "solution": "Remove affected leaves and avoid splashing water onto foliage.",
        "prevention": "Maintain clean growing areas and good airflow."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Tomato Spider Mites",
        "plant": "Tomato",
        "type": "Pest",
        "description": "Tiny pests that feed on plant tissue and cause leaf discoloration.",
        "symptoms": "Fine stippling, yellowing and possible webbing.",
        "solution": "Inspect leaf undersides and use appropriate pest-management practices.",
        "prevention": "Monitor plants regularly and maintain healthy growing conditions."
    },

    "Tomato__Target_Spot": {
        "name": "Tomato Target Spot",
        "plant": "Tomato",
        "type": "Fungal disease",
        "description": "A fungal disease producing circular lesions on tomato leaves.",
        "symptoms": "Circular dark lesions that may resemble target patterns.",
        "solution": "Remove severely affected material and improve airflow.",
        "prevention": "Reduce prolonged leaf wetness and maintain plant spacing."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "plant": "Tomato",
        "type": "Viral disease",
        "description": "A viral disease that can cause yellowing, curling and reduced plant growth.",
        "symptoms": "Curling leaves, yellowing and stunted growth.",
        "solution": "Remove severely affected plants where appropriate and manage insect vectors.",
        "prevention": "Monitor for whiteflies and maintain good garden hygiene."
    },

    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "plant": "Tomato",
        "type": "Viral disease",
        "description": "A viral disease that can produce mosaic-like patterns on leaves.",
        "symptoms": "Mottled or mosaic patterns and possible growth reduction.",
        "solution": "Remove severely affected plants and clean tools carefully.",
        "prevention": "Avoid transferring plant sap between plants and use clean tools."
    },

    "Tomato_healthy": {
        "name": "Healthy Tomato",
        "plant": "Tomato",
        "type": "Healthy",
        "description": "The model detected a healthy tomato leaf pattern.",
        "symptoms": "No major disease pattern detected.",
        "solution": "Continue normal plant care.",
        "prevention": "Maintain good sunlight, watering, nutrition and monitoring."
    }
}


# =========================================================
# QUESTION DATABASE
# =========================================================
qa_database = {
    "yellow": (
        "Yellow leaves can have several causes, including watering problems, "
        "nutrient deficiencies, pests or disease. Check the soil moisture, "
        "leaf undersides and the overall condition of the plant."
    ),

    "water": (
        "Water according to the plant's needs and growing conditions. "
        "Avoid constantly waterlogged soil and avoid unnecessarily wetting "
        "the leaves."
    ),

    "tomato": (
        "Tomatoes generally benefit from strong sunlight, consistent watering, "
        "good airflow and regular inspection for pests and diseases."
    ),

    "potato": (
        "Potatoes should be monitored for early blight, late blight and other "
        "problems. Good airflow, appropriate watering and healthy planting "
        "material can help reduce disease pressure."
    ),

    "pepper": (
        "Pepper plants benefit from sunlight, appropriate watering, healthy "
        "soil and adequate spacing. Inspect leaves regularly."
    ),

    "blight": (
        "Blight diseases can damage foliage quickly. Remove severely affected "
        "material, reduce prolonged leaf wetness, improve airflow and seek "
        "local agricultural advice for serious outbreaks."
    ),

    "spots": (
        "Leaf spots can be caused by fungi, bacteria, pests or environmental "
        "stress. Look at the shape, size and distribution of spots together "
        "with other symptoms."
    ),

    "healthy": (
        "Healthy plants generally have strong growth and leaves without "
        "significant discoloration, lesions or pest damage. Regular monitoring "
        "helps detect problems early."
    ),

    "fungus": (
        "Fungal diseases are often encouraged by moisture and poor airflow. "
        "Good spacing, ventilation and avoiding prolonged leaf wetness can "
        "help reduce disease risk."
    ),

    "pest": (
        "Inspect both sides of leaves for insects, eggs, webbing or feeding "
        "damage. Early detection is important for plant protection."
    )
}


# =========================================================
# SESSION STATE
# =========================================================
if "detections" not in st.session_state:
    st.session_state.detections = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("""
<div class="brand">
    <div class="brand-icon">🌿</div>
    <div class="brand-title">PlantCare AI</div>
    <div class="brand-subtitle">Intelligent Plant Health</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "🏠 Home",
        "🔬 Disease Detection",
        "📚 Disease Library",
        "💬 Ask PlantCare",
        "📊 Dashboard",
        "🤖 About Model"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="text-align:center;">
    <b>🌱 PlantCare AI</b><br>
    <small>Protect your plants with AI</small>
</div>
""", unsafe_allow_html=True)


# =========================================================
# HOME
# =========================================================
if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">AI-POWERED PLANT HEALTH</div>

        <div class="hero-title">
            Know your plant.<br>
            Protect your harvest.
        </div>

        <div class="hero-text">
            PlantCare AI analyzes plant leaf images using deep learning,
            identifies common plant diseases and gives practical guidance
            to help you care for your plants.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">15</div>
            <div class="stat-label">Disease Classes</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">91.3%</div>
            <div class="stat-label">Validation Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">3</div>
            <div class="stat-label">Plant Types</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">AI</div>
            <div class="stat-label">Leaf Analysis</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##")

    st.markdown(
        '<div class="section-title">Everything you need in one place</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Explore PlantCare AI features below.</div>',
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
        <div class="feature">
            <div class="feature-icon">🔬</div>
            <div class="feature-title">Disease Detection</div>
            <div class="feature-text">
                Upload a leaf image and let the trained AI model
                analyze the plant condition.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class="feature">
            <div class="feature-icon">📚</div>
            <div class="feature-title">Disease Library</div>
            <div class="feature-text">
                Explore diseases, symptoms, solutions and prevention
                information.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class="feature">
            <div class="feature-icon">💬</div>
            <div class="feature-title">Plant Assistant</div>
            <div class="feature-text">
                Ask general questions about watering, pests,
                diseases and plant health.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##")

    st.markdown(
        '<div class="section-title">🌱 Supported Plants</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("""
        <div class="card">
            <h2>🍅 Tomato</h2>
            <p class="card-text">
                Detect common tomato diseases and healthy leaves.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="card">
            <h2>🥔 Potato</h2>
            <p class="card-text">
                Detect potato early blight, late blight and healthy leaves.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="card">
            <h2>🌶️ Pepper</h2>
            <p class="card-text">
                Detect pepper bacterial spot and healthy leaves.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# DISEASE DETECTION
# =========================================================
elif page == "🔬 Disease Detection":

    st.markdown(
        '<div class="section-title">🔬 AI Disease Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Upload a clear tomato, potato or pepper leaf image.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "📷 Choose a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1.1, 1])

        with left:
            st.image(
                image,
                caption="Uploaded leaf",
                use_container_width=True
            )

        with right:
            st.markdown("""
            <div class="card">
                <div class="card-title">📷 Image Ready</div>
                <p class="card-text">
                    Your image is ready for analysis.
                    For best results, use a clear image with the leaf
                    visible and reasonably good lighting.
                </p>
            </div>
            """, unsafe_allow_html=True)

            analyze = st.button(
                "🔬 ANALYZE LEAF",
                type="primary",
                use_container_width=True
            )

        if analyze:

            with st.spinner("🤖 AI is analyzing your leaf..."):

                model = load_model()

                img = image.resize((224, 224))
                img_array = np.array(img, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)

                predictions = model.predict(
                    img_array,
                    verbose=0
                )

                predicted_index = int(np.argmax(predictions[0]))
                predicted_class = class_names[predicted_index]
                confidence = float(
                    predictions[0][predicted_index] * 100
                )

            info = disease_info.get(
                predicted_class,
                {
                    "name": predicted_class.replace("_", " "),
                    "plant": "Plant",
                    "type": "Unknown",
                    "description": "The model identified this condition.",
                    "symptoms": "Visual symptoms should be reviewed carefully.",
                    "solution": "Consider consulting a qualified agricultural professional.",
                    "prevention": "Monitor the plant regularly."
                }
            )

            is_healthy = "healthy" in predicted_class.lower()

            if is_healthy:
                health_score = confidence
                health_text = "Healthy pattern detected"
            elif confidence >= 90:
                health_score = 25
                health_text = "High disease risk"
            elif confidence >= 75:
                health_score = 45
                health_text = "Moderate disease risk"
            else:
                health_score = 60
                health_text = "Needs further review"

            st.session_state.detections.append({
                "disease": info["name"],
                "confidence": confidence,
                "health_score": health_score
            })

            st.markdown("---")

            st.markdown(
                '<div class="section-title">🎯 Analysis Result</div>',
                unsafe_allow_html=True
            )

            st.markdown(f"""
            <div class="result-card">

                <div style="color:#718078;font-weight:700;font-size:13px;">
                    DETECTED CONDITION
                </div>

                <div class="result-title">
                    🌿 {info["name"]}
                </div>

                <br>

                <div style="color:#718078;font-weight:700;font-size:13px;">
                    MODEL CONFIDENCE
                </div>

                <div class="confidence">
                    {confidence:.2f}%
                </div>

            </div>
            """, unsafe_allow_html=True)

            st.progress(
                min(confidence / 100, 1.0)
            )

            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric(
                    "Plant",
                    info["plant"]
                )

            with r2:
                st.metric(
                    "Condition",
                    info["type"]
                )

            with r3:
                st.metric(
                    "Health Indicator",
                    f"{health_score:.0f}/100"
                )

            if is_healthy:
                st.success(
                    f"🟢 {health_text}"
                )
            elif confidence >= 90:
                st.error(
                    f"🔴 {health_text}"
                )
            elif confidence >= 75:
                st.warning(
                    f"🟡 {health_text}"
                )
            else:
                st.info(
                    f"🔵 {health_text}"
                )

            st.markdown("---")

            a, b = st.columns(2)

            with a:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">📖 What is it?</div>
                    <p class="card-text">
                        {info["description"]}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with b:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">🔎 Symptoms</div>
                    <p class="card-text">
                        {info["symptoms"]}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            a, b = st.columns(2)

            with a:
                st.markdown(f"""
                <div class="info-box">
                    <b>🛠️ What should you do?</b>
                    <p>{info["solution"]}</p>
                </div>
                """, unsafe_allow_html=True)

            with b:
                st.markdown(f"""
                <div class="info-box">
                    <b>🛡️ How can you prevent it?</b>
                    <p>{info["prevention"]}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="warning-box">
                <b>⚠️ Important</b><br>
                AI predictions are for educational and project purposes.
                They should not replace professional agricultural diagnosis.
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# DISEASE LIBRARY
# =========================================================
elif page == "📚 Disease Library":

    st.markdown(
        '<div class="section-title">📚 Plant Disease Library</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Search and explore conditions recognized by PlantCare AI.'
        '</div>',
        unsafe_allow_html=True
    )

    search = st.text_input(
        "🔎 Search the library",
        placeholder="Search tomato, potato, blight, healthy..."
    )

    results = []

    for key in class_names:

        info = disease_info.get(key)

        if info:

            searchable = (
                info["name"] + " " +
                info["plant"] + " " +
                info["type"] + " " +
                info["description"]
            )

            if search.lower() in searchable.lower():
                results.append(info)

    st.info(f"Showing {len(results)} conditions")

    for info in results:

        with st.expander(
            f"🌿 {info['name']}  •  {info['plant']}"
        ):

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"**Plant:** {info['plant']}")
                st.markdown(f"**Type:** {info['type']}")
                st.write(info["description"])

            with c2:
                st.markdown("**🔎 Symptoms**")
                st.write(info["symptoms"])

                st.markdown("**🛠️ Solution**")
                st.write(info["solution"])

                st.markdown("**🛡️ Prevention**")
                st.write(info["prevention"])


# =========================================================
# ASK PLANTCARE
# =========================================================
elif page == "💬 Ask PlantCare":

    st.markdown(
        '<div class="section-title">💬 Ask PlantCare</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Your plant health question assistant.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">PLANT HEALTH ASSISTANT</div>
        <div class="hero-title" style="font-size:38px;">
            What would you like to know?
        </div>
        <div class="hero-text">
            Ask about plant diseases, yellow leaves, watering,
            pests, blight and prevention.
        </div>
    </div>
    """, unsafe_allow_html=True)

    question = st.text_input(
        "🌱 Your question",
        placeholder="Example: Why are my tomato leaves turning yellow?"
    )

    if st.button(
        "🌿 ASK PLANTCARE",
        type="primary",
        use_container_width=True
    ):

        if question.strip():

            q = question.lower()
            answer = None

            for keyword, response in qa_database.items():

                if keyword in q:
                    answer = response
                    break

            if answer is None:
                answer = (
                    "I can currently answer general questions about "
                    "tomatoes, potatoes, peppers, watering, pests, "
                    "blight, fungal problems and plant health. "
                    "Try asking about a specific symptom."
                )

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer
                }
            )

    for chat in reversed(st.session_state.chat_history):

        st.markdown(
            f"""
            <div class="chat-user">
                <b>You</b><br>
                {chat["question"]}
            </div>

            <div class="chat-ai">
                <b>🌿 PlantCare AI</b><br><br>
                {chat["answer"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 💡 Try asking")

    examples = [
        "Why are my tomato leaves turning yellow?",
        "How can I prevent blight?",
        "How often should I water my plants?",
        "What should I do about plant pests?",
        "How can I keep my potato plants healthy?"
    ]

    for example in examples:
        st.write("🌱 " + example)


# =========================================================
# DASHBOARD
# =========================================================
elif page == "📊 Dashboard":

    st.markdown(
        '<div class="section-title">📊 Plant Health Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Your detection history and plant health indicators.'
        '</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.detections:

        st.markdown("""
        <div class="card">
            <div class="card-title">🌱 No detections yet</div>
            <p class="card-text">
                Go to Disease Detection and analyze your first leaf.
                Your results will appear here automatically.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:

        total = len(st.session_state.detections)

        healthy = sum(
            1 for d in st.session_state.detections
            if "healthy" in d["disease"].lower()
        )

        disease = total - healthy

        average_health = sum(
            d["health_score"]
            for d in st.session_state.detections
        ) / total

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Images Analyzed",
                total
            )

        with c2:
            st.metric(
                "Healthy",
                healthy
            )

        with c3:
            st.metric(
                "Disease",
                disease
            )

        with c4:
            st.metric(
                "Avg Health",
                f"{average_health:.0f}/100"
            )

        st.markdown("### 🌿 Plant Health Indicator")

        st.progress(
            min(average_health / 100, 1.0)
        )

        if average_health >= 75:
            st.success("🟢 Overall plant health indicator is good.")
        elif average_health >= 50:
            st.warning("🟡 Some plant health concerns were detected.")
        else:
            st.error("🔴 Multiple health concerns were detected.")

        st.markdown("### 📈 Detection History")

        for i, detection in enumerate(
            reversed(st.session_state.detections),
            start=1
        ):

            st.markdown(f"""
            <div class="card">
                <b>Detection {i}</b><br>
                🌿 {detection["disease"]}<br>
                🎯 Confidence: {detection["confidence"]:.2f}%<br>
                ❤️ Health indicator: {detection["health_score"]:.0f}/100
            </div>
            """, unsafe_allow_html=True)

        st.caption(
            "The health indicator is a project metric based on the model result; "
            "it is NOT a scientifically validated survival probability."
        )


# =========================================================
# ABOUT MODEL
# =========================================================
elif page == "🤖 About Model":

    st.markdown(
        '<div class="section-title">🤖 About PlantCare AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Technology behind the plant disease detector.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Disease Classes",
            "15"
        )

    with c2:
        st.metric(
            "Validation Accuracy",
            "91.3%"
        )

    with c3:
        st.metric(
            "Input Image",
            "224 × 224"
        )

    st.markdown("##")

    st.markdown("""
    <div class="card">
        <div class="card-title">🧠 Deep Learning Model</div>
        <p class="card-text">
            PlantCare AI uses a TensorFlow/Keras image classification model
            trained to recognize common plant leaf conditions.
            The trained model is hosted on Hugging Face and loaded by the
            Streamlit application.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ How the AI works")

    steps = [
        ("1", "📷", "Upload", "User uploads a plant leaf image."),
        ("2", "🔄", "Preprocess", "Image is resized to 224 × 224 pixels."),
        ("3", "🧠", "Analyze", "The neural network analyzes visual patterns."),
        ("4", "🎯", "Predict", "The highest-probability class is selected."),
        ("5", "📊", "Explain", "The app displays the result and plant-care guidance.")
    ]

    for number, icon, title, description in steps:

        st.markdown(f"""
        <div class="card">
            <span style="font-size:25px;">{icon}</span>
            <b style="font-size:19px;">
                {number}. {title}
            </b>
            <br>
            <span class="card-text">
                {description}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        <b>⚠️ Model limitation</b><br>
        A 91.3% validation accuracy does not mean every real-world
        image will be classified correctly. Lighting, camera quality,
        background and plant variety can affect predictions.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer-custom">
    🌿 <b>PlantCare AI</b><br>
    Intelligent plant disease detection • TensorFlow • Streamlit
</div>
""", unsafe_allow_html=True)
