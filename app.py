import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# =========================================================
# PAGE SETTINGS
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
    background: linear-gradient(135deg, #f4fff8 0%, #eefbf4 50%, #f8fffb 100%);
}

/* Remove Streamlit default menu/footer */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #063b2b 0%, #087443 100%);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar title */
.sidebar-title {
    font-size: 26px;
    font-weight: 800;
    text-align: center;
    padding: 10px 0 5px 0;
}

.sidebar-subtitle {
    text-align: center;
    color: #c9f7df !important;
    font-size: 13px;
    margin-bottom: 20px;
}

/* Main hero */
.hero {
    background: linear-gradient(135deg, #063b2b, #087443, #20a86b);
    padding: 50px;
    border-radius: 30px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 15px 45px rgba(0, 80, 50, 0.18);
}

.hero-small {
    color: #bff5d8;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero-title {
    color: white !important;
    font-size: 48px;
    font-weight: 800;
    line-height: 1.1;
    margin: 12px 0;
}

.hero-text {
    color: #e8fff2 !important;
    font-size: 18px;
    line-height: 1.6;
    max-width: 780px;
}

/* Cards */
.card {
    background: white;
    border: 1px solid #d8eee1;
    border-radius: 22px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 8px 25px rgba(20, 80, 50, 0.07);
}

.card h2,
.card h3 {
    color: #063b2b !important;
}

.card p {
    color: #40564b !important;
    line-height: 1.6;
}

/* Stats */
.stat {
    background: white;
    border: 1px solid #d8eee1;
    border-radius: 20px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(20, 80, 50, 0.06);
}

.stat-number {
    color: #087443 !important;
    font-size: 32px;
    font-weight: 800;
}

.stat-label {
    color: #53685d !important;
    font-size: 13px;
    font-weight: 600;
}

/* Page headings */
h1 {
    color: #063b2b !important;
    font-weight: 800 !important;
}

h2 {
    color: #075c3b !important;
    font-weight: 800 !important;
}

h3 {
    color: #087443 !important;
    font-weight: 700 !important;
}

p, li {
    color: #40564b;
}

/* Detection result */
.result {
    background: linear-gradient(135deg, #ffffff, #effbf4);
    border: 2px solid #a7dfc0;
    border-radius: 25px;
    padding: 30px;
    margin-top: 25px;
    box-shadow: 0 12px 35px rgba(20, 80, 50, 0.08);
}

.result-label {
    color: #668074 !important;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}

.result-name {
    color: #063b2b !important;
    font-size: 32px;
    font-weight: 800;
    margin-top: 8px;
}

.confidence {
    color: #087443 !important;
    font-size: 40px;
    font-weight: 800;
}

/* Plant cards */
.plant-card {
    background: white;
    border-radius: 22px;
    padding: 30px;
    min-height: 190px;
    border: 1px solid #d8eee1;
    box-shadow: 0 8px 25px rgba(20, 80, 50, 0.07);
}

.plant-card h2 {
    font-size: 42px;
    margin: 0;
}

.plant-card h3 {
    color: #063b2b !important;
}

.plant-card p {
    color: #53685d !important;
}

/* Info boxes */
.info-box {
    background: #ecfdf4;
    border-left: 5px solid #0aa35a;
    border-radius: 14px;
    padding: 20px;
    margin: 15px 0;
}

.info-box h3 {
    margin-top: 0;
}

.info-box p {
    color: #365447 !important;
}

/* Upload area */
div[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #8bd2aa;
    border-radius: 20px;
    padding: 15px;
}

/* Buttons */
.stButton > button {
    border-radius: 14px;
    min-height: 48px;
    font-weight: 700;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 40px 0 15px;
    color: #6b7e74 !important;
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
        "type": "Bacterial Disease",
        "description": "A bacterial disease that can cause dark spots and lesions on pepper leaves and fruit.",
        "symptoms": "Small dark spots, leaf lesions and damaged fruit.",
        "solution": "Remove badly affected plant material. Keep foliage dry and improve air circulation.",
        "prevention": "Avoid overhead watering, maintain spacing and clean gardening tools."
    },

    "Pepper__bell___healthy": {
        "name": "Healthy Pepper",
        "plant": "Pepper",
        "type": "Healthy",
        "description": "The model detected a healthy pepper leaf.",
        "symptoms": "No major disease pattern detected.",
        "solution": "Continue normal plant care.",
        "prevention": "Provide good sunlight, watering, nutrition and regular monitoring."
    },

    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "plant": "Potato",
        "type": "Fungal Disease",
        "description": "A fungal disease that commonly causes dark lesions on potato leaves.",
        "symptoms": "Dark circular spots and yellowing around affected areas.",
        "solution": "Remove affected leaves and improve airflow around plants.",
        "prevention": "Avoid prolonged leaf wetness and maintain good plant spacing."
    },

    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "plant": "Potato",
        "type": "Disease",
        "description": "A serious potato disease that can spread quickly under favorable conditions.",
        "symptoms": "Dark irregular leaf lesions, discoloration and rapid plant decline.",
        "solution": "Remove affected plant material and reduce prolonged leaf wetness. For serious outbreaks, consult an agricultural professional.",
        "prevention": "Use healthy planting material, provide airflow and monitor plants regularly."
    },

    "Potato___healthy": {
        "name": "Healthy Potato",
        "plant": "Potato",
        "type": "Healthy",
        "description": "The model detected a healthy potato leaf.",
        "symptoms": "No major disease pattern detected.",
        "solution": "Continue normal plant care.",
        "prevention": "Monitor regularly and maintain appropriate watering and nutrition."
    },

    "Tomato_Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "plant": "Tomato",
        "type": "Bacterial Disease",
        "description": "A bacterial disease that can affect tomato leaves and fruit.",
        "symptoms": "Small dark spots and lesions on leaves or fruit.",
        "solution": "Remove severely affected material and avoid unnecessarily wetting foliage.",
        "prevention": "Use clean tools, maintain airflow and avoid overhead watering."
    },

    "Tomato_Early_blight": {
        "name": "Tomato Early Blight",
        "plant": "Tomato",
        "type": "Fungal Disease",
        "description": "A fungal disease that can cause dark lesions on tomato foliage.",
        "symptoms": "Dark circular lesions, yellowing and leaf drop.",
        "solution": "Remove affected leaves and improve airflow.",
        "prevention": "Avoid prolonged leaf wetness and maintain plant spacing."
    },

    "Tomato_Late_blight": {
        "name": "Tomato Late Blight",
        "plant": "Tomato",
        "type": "Disease",
        "description": "A disease that can spread rapidly under suitable environmental conditions.",
        "symptoms": "Dark irregular patches and rapid leaf damage.",
        "solution": "Remove affected material and minimize prolonged leaf wetness.",
        "prevention": "Monitor frequently and maintain good airflow."
    },

    "Tomato_Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "plant": "Tomato",
        "type": "Fungal Disease",
        "description": "A fungal disease commonly associated with humid conditions.",
        "symptoms": "Yellow areas on upper leaf surfaces and mold-like growth underneath.",
        "solution": "Improve ventilation and remove severely affected leaves.",
        "prevention": "Reduce humidity around foliage and improve airflow."
    },

    "Tomato_Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "plant": "Tomato",
        "type": "Fungal Disease",
        "description": "A fungal disease producing many small spots on tomato leaves.",
        "symptoms": "Small circular spots, often with darker edges.",
        "solution": "Remove affected leaves and avoid splashing water onto foliage.",
        "prevention": "Maintain clean growing areas and good airflow."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Tomato Spider Mites",
        "plant": "Tomato",
        "type": "Pest",
        "description": "Tiny pests that feed on plant tissue and can cause leaf discoloration.",
        "symptoms": "Fine stippling, yellowing and possible webbing.",
        "solution": "Inspect leaf undersides and use appropriate pest-management methods.",
        "prevention": "Monitor plants regularly and maintain healthy growing conditions."
    },

    "Tomato__Target_Spot": {
        "name": "Tomato Target Spot",
        "plant": "Tomato",
        "type": "Fungal Disease",
        "description": "A fungal disease that produces circular lesions on tomato leaves.",
        "symptoms": "Circular dark lesions that may resemble target patterns.",
        "solution": "Remove severely affected material and improve airflow.",
        "prevention": "Reduce prolonged leaf wetness and maintain plant spacing."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "plant": "Tomato",
        "type": "Viral Disease",
        "description": "A viral disease that can cause yellowing, curling and reduced plant growth.",
        "symptoms": "Curling leaves, yellowing and stunted growth.",
        "solution": "Remove severely affected plants where appropriate and manage insect vectors.",
        "prevention": "Monitor for whiteflies and maintain good garden hygiene."
    },

    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "plant": "Tomato",
        "type": "Viral Disease",
        "description": "A viral disease that can produce mosaic-like patterns on leaves.",
        "symptoms": "Mottled or mosaic patterns and possible growth reduction.",
        "solution": "Remove severely affected plants and clean tools carefully.",
        "prevention": "Avoid transferring plant sap between plants and use clean tools."
    },

    "Tomato_healthy": {
        "name": "Healthy Tomato",
        "plant": "Tomato",
        "type": "Healthy",
        "description": "The model detected a healthy tomato leaf.",
        "symptoms": "No major disease pattern detected.",
        "solution": "Continue normal plant care.",
        "prevention": "Maintain good sunlight, watering, nutrition and monitoring."
    }
}


# =========================================================
# SIMPLE PLANT Q&A
# =========================================================

qa_database = {
    "yellow": "Yellow leaves can be caused by watering problems, nutrient issues, pests or disease. Check soil moisture and inspect both sides of the leaves.",
    "water": "Water plants according to their needs. Avoid constantly waterlogged soil and avoid unnecessarily wetting the leaves.",
    "tomato": "Tomatoes generally need good sunlight, consistent watering, airflow and regular inspection for pests and diseases.",
    "potato": "Potatoes should be monitored for early blight and late blight. Good airflow, appropriate watering and healthy planting material can help.",
    "pepper": "Peppers benefit from sunlight, appropriate watering, healthy soil and adequate spacing.",
    "blight": "For blight problems, remove severely affected material, reduce prolonged leaf wetness and improve airflow.",
    "spots": "Leaf spots can have several causes including fungi, bacteria, pests or environmental stress.",
    "healthy": "Healthy plants generally have strong growth and leaves without significant discoloration, lesions or pest damage.",
    "fungus": "Fungal diseases are often encouraged by moisture and poor airflow. Good spacing and ventilation can reduce risk.",
    "pest": "Inspect both sides of leaves for insects, eggs, webbing or feeding damage. Early detection is important."
}


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown(
    '<div class="sidebar-title">🌿 PlantCare AI</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="sidebar-subtitle">Smart Plant Health Assistant</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MENU",
    [
        "🏠 Home",
        "🔬 Disease Detection",
        "📚 Disease Library",
        "💬 Ask PlantCare",
        "🤖 About Model"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "🌱 **Protect your plants with AI**"
)

st.sidebar.caption(
    "Predictions are for educational purposes."
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-small">AI-POWERED PLANT HEALTH</div>
        <div class="hero-title">Know your plant.<br>Protect your harvest.</div>
        <div class="hero-text">
            Upload a leaf image and let PlantCare AI analyze it.
            Discover possible diseases, understand symptoms,
            and learn practical prevention steps.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="stat">
            <div class="stat-number">15</div>
            <div class="stat-label">Disease Classes</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="stat">
            <div class="stat-number">91.3%</div>
            <div class="stat-label">Validation Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="stat">
            <div class="stat-number">3</div>
            <div class="stat-label">Plant Types</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="stat">
            <div class="stat-number">AI</div>
            <div class="stat-label">Leaf Analysis</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 🌱 Your Plant Health Dashboard")

    a, b = st.columns(2)

    with a:
        st.markdown("""
        <div class="card">
            <h2>🔬 Detect a Disease</h2>
            <p>
            Upload a tomato, potato or pepper leaf.
            PlantCare AI will analyze the image and show
            the predicted condition and confidence.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="card">
            <h2>💬 Ask PlantCare</h2>
            <p>
            Ask questions about watering, yellow leaves,
            pests, blight, plant health and prevention.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 🌿 Supported Plants")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("""
        <div class="plant-card">
            <h2>🍅</h2>
            <h3>Tomato</h3>
            <p>Detect common tomato diseases and healthy leaves.</p>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="plant-card">
            <h2>🥔</h2>
            <h3>Potato</h3>
            <p>Detect early blight, late blight and healthy leaves.</p>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="plant-card">
            <h2>🌶️</h2>
            <h3>Pepper</h3>
            <p>Detect bacterial spot and healthy pepper leaves.</p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# DISEASE DETECTION
# =========================================================

elif page == "🔬 Disease Detection":

    st.title("🔬 Disease Detection")

    st.write(
        "Upload a clear image of a tomato, potato or pepper leaf."
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
                caption="Your uploaded leaf",
                use_container_width=True
            )

        with right:

            st.markdown("""
            <div class="card">
                <h2>🌿 Image Ready</h2>
                <p>
                Your image is ready for AI analysis.
                For best results, use a clear image with
                the leaf visible and reasonably well lit.
                </p>
            </div>
            """, unsafe_allow_html=True)

            analyze = st.button(
                "🔬 Analyze My Leaf",
                type="primary",
                use_container_width=True
            )

        if analyze:

            with st.spinner("🤖 PlantCare AI is analyzing the leaf..."):

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
                    "plant": "Unknown",
                    "type": "Unknown",
                    "description": "The model identified this condition.",
                    "symptoms": "See the uploaded image.",
                    "solution": "Consider professional agricultural advice.",
                    "prevention": "Monitor the plant regularly."
                }
            )

            st.markdown("## 🎯 Analysis Result")

            st.markdown(f"""
            <div class="result">
                <div class="result-label">PREDICTED CONDITION</div>
                <div class="result-name">🌿 {info["name"]}</div>
                <br>
                <div class="result-label">MODEL CONFIDENCE</div>
                <div class="confidence">{confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(
                min(confidence / 100, 1.0)
            )

            st.markdown("### 🩺 Plant Health Indicator")

            if "healthy" in predicted_class.lower():

                st.success(
                    "🟢 Healthy pattern detected"
                )

            elif confidence >= 90:

                st.error(
                    "🔴 High disease-risk pattern detected"
                )

            elif confidence >= 75:

                st.warning(
                    "🟡 Moderate disease-risk pattern detected"
                )

            else:

                st.info(
                    "🔵 Result should be reviewed carefully"
                )

            st.markdown("## 📖 Understanding Your Result")

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"""
                <div class="card">
                    <h2>📖 What is it?</h2>
                    <p>{info["description"]}</p>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="card">
                    <h2>🔎 Common Symptoms</h2>
                    <p>{info["symptoms"]}</p>
                </div>
                """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"""
                <div class="info-box">
                    <h3>🛠️ What to Do</h3>
                    <p>{info["solution"]}</p>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="info-box">
                    <h3>🛡️ Prevention</h3>
                    <p>{info["prevention"]}</p>
                </div>
                """, unsafe_allow_html=True)

            st.caption(
                "⚠️ AI predictions are for educational purposes and should not replace professional agricultural diagnosis."
            )


# =========================================================
# DISEASE LIBRARY
# =========================================================

elif page == "📚 Disease Library":

    st.title("📚 Disease Library")

    st.write(
        "Explore diseases and healthy conditions recognized by PlantCare AI."
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
                info["name"] +
                " " +
                info["plant"] +
                " " +
                info["type"]
            )

            if search.lower() in searchable.lower():

                results.append(info)

    st.markdown(
        f"### 🌿 {len(results)} conditions found"
    )

    for info in results:

        with st.expander(
            f"🌿 {info['name']}  •  {info['plant']}"
        ):

            st.markdown(
                f"**Type:** {info['type']}"
            )

            st.markdown("### 📖 Description")
            st.write(info["description"])

            st.markdown("### 🔎 Symptoms")
            st.write(info["symptoms"])

            st.markdown("### 🛠️ What to do")
            st.write(info["solution"])

            st.markdown("### 🛡️ Prevention")
            st.write(info["prevention"])


# =========================================================
# ASK PLANTCARE
# =========================================================

elif page == "💬 Ask PlantCare":

    st.title("💬 Ask PlantCare")

    st.markdown("""
    <div class="hero">
        <div class="hero-small">PLANT HEALTH ASSISTANT</div>
        <div class="hero-title" style="font-size:38px;">
            Ask anything about your plant.
        </div>
        <div class="hero-text">
            Ask about yellow leaves, watering, pests,
            blight, plant diseases and prevention.
        </div>
    </div>
    """, unsafe_allow_html=True)

    question = st.text_input(
        "🔎 Your question",
        placeholder="Why are my tomato leaves turning yellow?"
    )

    if st.button(
        "🌱 Ask PlantCare",
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
                    "I can answer general questions about tomato, "
                    "potato and pepper plants, common diseases, "
                    "symptoms, watering, pests and prevention."
                )

            st.markdown("## 💡 PlantCare Answer")

            st.markdown(f"""
            <div class="card">
                <h3>🌿 Your Question</h3>
                <p>{question}</p>
                <hr>
                <h3>🤖 PlantCare AI</h3>
                <p>{answer}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("## 💡 Try asking")

    examples = [
        "Why are my tomato leaves turning yellow?",
        "How can I prevent blight?",
        "How often should I water my plants?",
        "What should I do about plant pests?"
    ]

    for example in examples:

        st.markdown(
            f"🌱 {example}"
        )

    st.info(
        "This assistant uses a built-in plant knowledge base. It does not perform live web searches."
    )


# =========================================================
# ABOUT MODEL
# =========================================================

elif page == "🤖 About Model":

    st.title("🤖 About PlantCare AI")

    st.markdown("""
    <div class="hero">
        <div class="hero-small">THE TECHNOLOGY</div>
        <div class="hero-title" style="font-size:38px;">
            AI-powered leaf analysis.
        </div>
        <div class="hero-text">
            PlantCare AI uses a TensorFlow/Keras image classification
            model to recognize visual patterns in plant leaves.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="stat">
            <div class="stat-number">15</div>
            <div class="stat-label">Classes</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="stat">
            <div class="stat-number">91.3%</div>
            <div class="stat-label">Validation Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="stat">
            <div class="stat-number">224×224</div>
            <div class="stat-label">Image Input</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## ⚙️ How the AI Works")

    steps = [
        "📷 1. Upload a leaf image",
        "🔄 2. Image is resized to 224 × 224 pixels",
        "🧠 3. TensorFlow analyzes visual patterns",
        "🎯 4. The highest probability class is selected",
        "📊 5. Confidence is shown to the user"
    ]

    for step in steps:

        st.markdown(
            f"""
            <div class="card">
                <h3>{step}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## 🌿 Model Capabilities")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
        <div class="card">
            <h2>🍅 Supported Plants</h2>
            <p>Tomato</p>
            <p>Potato</p>
            <p>Pepper</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="card">
            <h2>🔬 Analysis</h2>
            <p>Disease classification</p>
            <p>Confidence estimation</p>
            <p>Plant-care recommendations</p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="app-footer">
    🌿 <b>PlantCare AI</b><br>
    AI-powered plant disease detection • TensorFlow • Streamlit
</div>
""", unsafe_allow_html=True)
