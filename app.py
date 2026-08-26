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
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PROFESSIONAL UI
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(94, 234, 212, 0.15), transparent 25%),
        radial-gradient(circle at 90% 15%, rgba(134, 239, 172, 0.18), transparent 25%),
        #f6faf7;
}

#MainMenu, footer {
    visibility: hidden;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #063b2b, #0b513b);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.hero {
    padding: 55px 50px;
    border-radius: 30px;
    background: linear-gradient(135deg, #063b2b, #0d704b, #24a36b);
    color: white;
    box-shadow: 0 20px 50px rgba(6, 59, 43, 0.20);
    margin-bottom: 30px;
}

.hero-small {
    color: #c9f7df;
    font-weight: 700;
    letter-spacing: 2px;
    font-size: 13px;
}

.hero-title {
    font-size: 52px;
    line-height: 1.05;
    font-weight: 800;
    margin: 12px 0;
}

.hero-text {
    font-size: 18px;
    line-height: 1.6;
    max-width: 750px;
    color: #e3fff0;
}

.card {
    background: rgba(255,255,255,0.92);
    border: 1px solid #dcebe3;
    border-radius: 22px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 8px 30px rgba(15, 50, 35, 0.06);
}

.card h3 {
    margin-top: 0;
}

.stat {
    background: white;
    border-radius: 20px;
    padding: 22px;
    text-align: center;
    border: 1px solid #deebe4;
    box-shadow: 0 8px 25px rgba(15, 50, 35, 0.05);
}

.stat-number {
    font-size: 32px;
    font-weight: 800;
    color: #087443;
}

.stat-label {
    color: #6b7c73;
    font-size: 13px;
}

.result {
    padding: 30px;
    border-radius: 25px;
    background: linear-gradient(135deg, #ffffff, #effbf4);
    border: 2px solid #a9dfc0;
    box-shadow: 0 12px 35px rgba(15, 70, 45, 0.08);
}

.result-name {
    font-size: 32px;
    font-weight: 800;
    color: #073b2b;
}

.confidence {
    font-size: 40px;
    font-weight: 800;
    color: #087443;
}

.chat-user {
    background: #087443;
    color: white;
    padding: 15px 20px;
    border-radius: 20px 20px 5px 20px;
    margin: 10px 0 10px 20%;
}

.chat-ai {
    background: white;
    border: 1px solid #dcebe3;
    padding: 18px 20px;
    border-radius: 20px 20px 20px 5px;
    margin: 10px 20% 10px 0;
    box-shadow: 0 5px 20px rgba(0,0,0,0.04);
}

.tip {
    background: #ecfdf3;
    border-left: 5px solid #12a05a;
    padding: 16px;
    border-radius: 12px;
}

.warning {
    background: #fff8e6;
    border-left: 5px solid #e5a400;
    padding: 16px;
    border-radius: 12px;
}

.stButton > button {
    border-radius: 13px;
    min-height: 48px;
    font-weight: 700;
}

div[data-testid="stFileUploader"] {
    background: white;
    border-radius: 20px;
    padding: 10px;
    border: 2px dashed #a9dfc0;
}

.footer {
    text-align: center;
    color: #718078;
    padding: 40px 0 10px;
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
# DISEASE DATABASE
# =========================================================
disease_info = {

    "Pepper__bell___Bacterial_spot": {
        "name": "Pepper Bacterial Spot",
        "plant": "Pepper",
        "type": "Bacterial disease",
        "description": "A bacterial disease that can produce dark spots and lesions on pepper leaves and fruit.",
        "symptoms": "Small dark spots, leaf lesions and damaged fruit.",
        "solution": "Remove badly affected plant material, keep foliage dry and improve air circulation.",
        "prevention": "Avoid overhead watering, maintain spacing and use clean gardening tools."
    },

    "Pepper__bell___healthy": {
        "name": "Healthy Pepper",
        "plant": "Pepper",
        "type": "Healthy",
        "description": "The model detected a healthy pepper leaf.",
        "symptoms": "No major disease pattern detected.",
        "solution": "Continue normal plant care.",
        "prevention": "Maintain good sunlight, watering, nutrition and regular monitoring."
    },

    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "plant": "Potato",
        "type": "Fungal disease",
        "description": "A fungal disease that commonly causes dark lesions on potato leaves.",
        "symptoms": "Dark circular spots and yellowing around affected areas.",
        "solution": "Remove affected leaves and improve airflow around plants.",
        "prevention": "Avoid prolonged leaf wetness and maintain good plant spacing."
    },

    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "plant": "Potato",
        "type": "Disease",
        "description": "A serious potato disease that can spread quickly under favorable environmental conditions.",
        "symptoms": "Dark irregular leaf lesions, leaf discoloration and rapid plant decline.",
        "solution": "Remove affected plant material and reduce prolonged leaf wetness. For serious outbreaks, consult a qualified agricultural professional.",
        "prevention": "Use healthy planting material, provide good airflow and monitor plants regularly."
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
        "type": "Bacterial disease",
        "description": "A bacterial disease that can affect tomato leaves and fruit.",
        "symptoms": "Small dark spots and lesions on leaves or fruit.",
        "solution": "Remove severely affected material and avoid wetting foliage unnecessarily.",
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
        "description": "A disease that can spread rapidly under suitable environmental conditions.",
        "symptoms": "Dark irregular patches and rapid leaf damage.",
        "solution": "Remove affected material and minimize prolonged leaf wetness.",
        "prevention": "Monitor frequently and maintain good airflow."
    },

    "Tomato_Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "plant": "Tomato",
        "type": "Fungal disease",
        "description": "A fungal disease commonly associated with humid conditions.",
        "symptoms": "Yellow areas on upper leaf surfaces and mold-like growth underneath.",
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
        "description": "Tiny pests that feed on plant tissue and can cause leaf discoloration.",
        "symptoms": "Fine stippling, yellowing and possible webbing.",
        "solution": "Inspect leaf undersides and use appropriate pest-management methods.",
        "prevention": "Monitor plants regularly and maintain healthy growing conditions."
    },

    "Tomato__Target_Spot": {
        "name": "Tomato Target Spot",
        "plant": "Tomato",
        "type": "Fungal disease",
        "description": "A fungal disease that produces circular lesions on tomato leaves.",
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
        "description": "The model detected a healthy tomato leaf.",
        "symptoms": "No major disease pattern detected.",
        "solution": "Continue normal plant care.",
        "prevention": "Maintain good sunlight, watering, nutrition and monitoring."
    }
}


# =========================================================
# QUESTION ANSWER DATABASE
# =========================================================
qa_database = {
    "yellow": "Yellow leaves can have many causes, including watering problems, nutrient issues, pests or disease. Check the underside of the leaves, soil moisture and the overall condition of the plant before deciding on treatment.",

    "water": "Water plants according to their needs and growing conditions. Avoid keeping soil constantly waterlogged, and avoid unnecessarily wetting the leaves because prolonged leaf wetness can encourage some diseases.",

    "tomato": "Tomatoes generally benefit from good sunlight, consistent watering, airflow and regular inspection for pests and diseases. Avoid overcrowding plants and monitor leaves regularly.",

    "potato": "Potatoes should be monitored for early blight, late blight and other problems. Good airflow, appropriate watering and healthy planting material can help reduce disease pressure.",

    "pepper": "Pepper plants benefit from good sunlight, appropriate watering, healthy soil and adequate spacing. Check leaves regularly for bacterial spots and pests.",

    "blight": "Blight refers to diseases that can damage plant foliage rapidly. Remove severely affected material, reduce prolonged leaf wetness, improve airflow and seek local agricultural advice when an outbreak is serious.",

    "spots": "Leaf spots can be caused by fungi, bacteria, pests or environmental stress. Look at the size, shape and distribution of the spots and check for other symptoms before deciding what is causing them.",

    "healthy": "A healthy plant generally has strong growth and leaves without significant discoloration, lesions or pest damage. Regular monitoring helps detect problems early.",

    "fungus": "Fungal diseases are often encouraged by moisture and poor airflow. Good spacing, ventilation and avoiding prolonged leaf wetness can help reduce risk.",

    "pest": "Inspect both the upper and lower surfaces of leaves for insects, eggs, webbing or feeding damage. Early detection is important for effective pest management."
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
st.sidebar.markdown("# 🌿 PlantCare AI")
st.sidebar.caption("Smart plant health assistant")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔬 Disease Detection",
        "📚 Disease Library",
        "💬 Ask PlantCare",
        "🤖 About Model"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "AI predictions are for educational and project purposes."
)


# =========================================================
# HOME
# =========================================================
if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-small">AI-POWERED PLANT HEALTH</div>
        <div class="hero-title">
            Know your plant.<br>
            Protect your harvest.
        </div>
        <div class="hero-text">
            PlantCare AI uses deep learning to analyze plant leaf images,
            identify common diseases and provide practical plant-care guidance.
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

    st.markdown("##")

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("""
        <div class="card">
        <h3>🌱 Your plant health companion</h3>
        <p>
        Upload a leaf image to identify possible diseases,
        explore the disease library, or ask PlantCare AI
        questions about plant health.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="card">
        <h3>🔬 What you can do</h3>
        <p>
        📷 Analyze leaf images<br>
        📚 Explore diseases<br>
        💬 Ask plant questions<br>
        📊 Track detection results
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 🌿 Supported Plants")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("""
        <div class="card">
        <h2>🍅</h2>
        <h3>Tomato</h3>
        <p>Multiple common tomato diseases and healthy leaves.</p>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="card">
        <h2>🥔</h2>
        <h3>Potato</h3>
        <p>Early blight, late blight and healthy leaves.</p>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="card">
        <h2>🌶️</h2>
        <h3>Pepper</h3>
        <p>Bacterial spot and healthy leaves.</p>
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
        "Upload your leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1.1, 1])

        with col1:
            st.image(
                image,
                caption="Leaf image",
                use_container_width=True
            )

        with col2:

            st.markdown("""
            <div class="card">
            <h3>📷 Image ready</h3>
            <p>
            Make sure the leaf is clearly visible and reasonably well lit.
            </p>
            </div>
            """, unsafe_allow_html=True)

            analyze = st.button(
                "🔬 Analyze Leaf",
                type="primary",
                use_container_width=True
            )

        if analyze:

            with st.spinner("🤖 Analyzing your leaf..."):

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
                    "description": "The model identified this class.",
                    "symptoms": "See the uploaded image.",
                    "solution": "Consider professional agricultural advice.",
                    "prevention": "Monitor the plant regularly."
                }
            )

            # Health/risk indicator
            if "healthy" in predicted_class.lower():
                health_score = confidence
                health_label = "Healthy pattern detected"
            elif confidence >= 90:
                health_score = 25
                health_label = "High disease risk"
            elif confidence >= 75:
                health_score = 45
                health_label = "Moderate disease risk"
            else:
                health_score = 60
                health_label = "Uncertain / needs review"

            st.session_state.detections.append({
                "disease": info["name"],
                "confidence": confidence
            })

            st.markdown("## 🎯 Detection Complete")

            st.markdown(f"""
            <div class="result">
                <div style="color:#688076;font-weight:700;font-size:13px;">
                PREDICTED CONDITION
                </div>

                <div class="result-name">
                🌿 {info["name"]}
                </div>

                <br>

                <div style="color:#688076;font-weight:700;font-size:13px;">
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

            st.markdown("### 📊 Plant Health Indicator")

            if "healthy" in predicted_class.lower():
                st.success(
                    f"🟢 {health_label} — indicator: {health_score:.0f}/100"
                )
            elif confidence >= 90:
                st.error(
                    f"🔴 {health_label} — indicator: {health_score:.0f}/100"
                )
            elif confidence >= 75:
                st.warning(
                    f"🟡 {health_label} — indicator: {health_score:.0f}/100"
                )
            else:
                st.info(
                    f"🔵 {health_label} — indicator: {health_score:.0f}/100"
                )

            st.markdown("---")

            a, b = st.columns(2)

            with a:
                st.markdown(f"""
                <div class="card">
                <h3>📖 What is it?</h3>
                <p>{info["description"]}</p>
                </div>
                """, unsafe_allow_html=True)

            with b:
                st.markdown(f"""
                <div class="card">
                <h3>🔎 Common symptoms</h3>
                <p>{info["symptoms"]}</p>
                </div>
                """, unsafe_allow_html=True)

            a, b = st.columns(2)

            with a:
                st.markdown(f"""
                <div class="tip">
                <h3>🛠️ What to do</h3>
                <p>{info["solution"]}</p>
                </div>
                """, unsafe_allow_html=True)

            with b:
                st.markdown(f"""
                <div class="tip">
                <h3>🛡️ Prevention</h3>
                <p>{info["prevention"]}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            st.caption(
                "Important: Model predictions are not a professional agricultural diagnosis."
            )


# =========================================================
# DISEASE LIBRARY
# =========================================================
elif page == "📚 Disease Library":

    st.title("📚 Disease Library")

    st.write(
        "Explore the conditions included in the PlantCare AI model."
    )

    search = st.text_input(
        "🔎 Search diseases",
        placeholder="Try tomato, potato, blight, healthy..."
    )

    results = []

    for key in class_names:
        info = disease_info.get(key)

        if info:
            text = (
                info["name"] + " " +
                info["plant"] + " " +
                info["type"]
            )

            if search.lower() in text.lower():
                results.append(info)

    st.write(f"Showing **{len(results)}** conditions")

    for info in results:

        with st.expander(
            f"🌿 {info['name']} • {info['plant']}"
        ):

            st.markdown(f"**Type:** {info['type']}")

            st.write(info["description"])

            st.markdown("**🔎 Symptoms**")
            st.write(info["symptoms"])

            st.markdown("**🛠️ What to do**")
            st.write(info["solution"])

            st.markdown("**🛡️ Prevention**")
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
        Get quick general guidance about plant diseases,
        symptoms, watering, pests and prevention.
        </div>
    </div>
    """, unsafe_allow_html=True)

    question = st.text_input(
        "Ask your question",
        placeholder="e.g. Why are my tomato leaves turning yellow?"
    )

    if st.button(
        "Ask PlantCare 🌱",
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
                    "I can help with general questions about tomato, potato "
                    "and pepper plants, common diseases, symptoms, watering, "
                    "pests and prevention. Try asking about a specific plant "
                    "or symptom."
                )

            st.markdown(
                f"""
                <div class="chat-user">
                <b>You</b><br>
                {question}
                </div>

                <div class="chat-ai">
                <b>🌿 PlantCare AI</b><br><br>
                {answer}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "This assistant uses a built-in plant knowledge base and does not perform live web searches."
            )

    st.markdown("### 💡 Try asking")

    examples = [
        "Why are my tomato leaves turning yellow?",
        "How can I prevent blight?",
        "How often should I water my plants?",
        "What should I do about plant pests?"
    ]

    for example in examples:
        st.write("•", example)


# =========================================================
# ABOUT MODEL
# =========================================================
elif page == "🤖 About Model":

    st.title("🤖 About the Model")

    st.markdown("""
    <div class="card">
    <h3>🧠 Deep Learning Plant Classifier</h3>
    <p>
    PlantCare AI uses a TensorFlow/Keras image classification model
    trained to recognize plant leaf conditions.
    </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Classes", "15")

    with c2:
        st.metric("Validation Accuracy", "91.3%")

    with c3:
        st.metric("Input Size", "224 × 224")

    st.markdown("---")

    st.subheader("⚙️ Prediction Pipeline")

    steps = [
        "📷 Leaf image uploaded",
        "🔄 Image resized to 224 × 224",
        "🧠 Neural network analyzes visual patterns",
        "🎯 Highest-probability class selected",
        "📊 Confidence displayed to the user"
    ]

    for step in steps:
        st.markdown(
            f"""
            <div class="card">
            {step}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader("📈 Your Detection Dashboard")

    if len(st.session_state.detections) == 0:

        st.info(
            "Your detection statistics will appear here after you analyze images."
        )

    else:

        total = len(st.session_state.detections)

        healthy = sum(
            1 for d in st.session_state.detections
            if "healthy" in d["disease"].lower()
        )

        disease = total - healthy

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Images analyzed", total)

        with col2:
            st.metric("Healthy detections", healthy)

        with col3:
            st.metric("Disease detections", disease)

        st.markdown("### Detection confidence")

        for d in st.session_state.detections:

            st.write(
                f"**{d['disease']}** — {d['confidence']:.2f}%"
            )

            st.progress(
                min(d["confidence"] / 100, 1.0)
            )


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
🌿 <b>PlantCare AI</b><br>
AI-powered plant disease detection • Built with TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)
