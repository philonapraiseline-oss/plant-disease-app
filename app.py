import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FloraSense",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CONSTANTS
# =========================================================

MODEL_URL = (
    "https://huggingface.co/philona777/plant-disease-model/"
    "resolve/main/plant_disease_model.keras"
)

IMG_SIZE = (224, 224)

EXPECTED_CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background: #f4f8f5;
        color: #17352a;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #123d2b !important;
        font-weight: 750 !important;
    }

    p, li, label, span, div {
        color: #243b32;
    }

    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #123d2b 0%,
            #18583d 55%,
            #0e432e 100%
        );
        min-width: 285px !important;
        width: 285px !important;
    }

    section[data-testid="stSidebar"] > div {
        width: 285px !important;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .sidebar-brand {
        padding: 10px 10px 18px 10px;
        border-bottom: 1px solid rgba(255,255,255,0.18);
        margin-bottom: 18px;
    }

    .sidebar-brand-title {
        font-size: 27px;
        font-weight: 800;
        color: white !important;
        letter-spacing: -0.5px;
    }

    .sidebar-brand-subtitle {
        font-size: 13px;
        color: #d8f1e3 !important;
        margin-top: 4px;
        font-weight: 500;
    }

    .sidebar-section {
        color: #bce4ce !important;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.3px;
        text-transform: uppercase;
        margin: 12px 8px 7px 8px;
    }

    section[data-testid="stSidebar"] .stButton {
        width: 100%;
    }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border: none !important;
        border-radius: 10px !important;
        background: transparent !important;
        color: white !important;
        text-align: left !important;
        padding: 10px 12px !important;
        margin: 2px 0 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.13) !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] .stButton > button:focus {
        background: rgba(255,255,255,0.16) !important;
        color: white !important;
        outline: none !important;
    }

    .sidebar-footer {
        margin-top: 25px;
        padding: 15px 10px;
        border-top: 1px solid rgba(255,255,255,0.16);
        font-size: 12px;
        line-height: 1.7;
    }

    .sidebar-footer strong {
        color: white !important;
        font-size: 13px;
    }

    .sidebar-footer span {
        color: #c7e8d5 !important;
    }

    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        background: linear-gradient(
            135deg,
            #dff5e7 0%,
            #edf9f0 45%,
            #ccebd8 100%
        );
        border-radius: 24px;
        padding: 42px;
        border: 1px solid #c5e5d0;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 48px;
        margin-bottom: 8px;
        color: #103f2b !important;
    }

    .hero p {
        font-size: 18px;
        line-height: 1.7;
        color: #315447 !important;
        max-width: 800px;
    }

    /* =====================================================
       CARDS
       ===================================================== */

    .card {
        background: white;
        border: 1px solid #dce9e1;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow: 0 6px 22px rgba(20, 70, 45, 0.07);
    }

    .card h3 {
        margin-top: 0;
        color: #164b34 !important;
    }

    .card p {
        color: #40584d !important;
        line-height: 1.65;
    }

    .stat-card {
        background: white;
        border: 1px solid #dce9e1;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(20,70,45,0.06);
    }

    .stat-number {
        font-size: 31px;
        font-weight: 800;
        color: #16804c !important;
    }

    .stat-label {
        font-size: 13px;
        color: #53685e !important;
        font-weight: 600;
        margin-top: 4px;
    }

    /* =====================================================
       SECTION TITLES
       ===================================================== */

    .section-title {
        font-size: 26px;
        font-weight: 750;
        color: #123d2b !important;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .section-subtitle {
        color: #52675d !important;
        font-size: 15px;
        margin-bottom: 20px;
    }

    /* =====================================================
       BADGES
       ===================================================== */

    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 700;
        background: #e4f5ea;
        color: #176b42 !important;
        margin-bottom: 10px;
    }

    .badge-danger {
        background: #fde8e8;
        color: #a62f2f !important;
    }

    .badge-warning {
        background: #fff3d8;
        color: #8b6414 !important;
    }

    /* =====================================================
       RESULT CARD
       ===================================================== */

    .result-card {
        background: linear-gradient(
            135deg,
            #e7f8ed,
            #ffffff
        );
        border: 2px solid #a9dcbc;
        border-radius: 20px;
        padding: 28px;
        margin-top: 20px;
    }

    .result-title {
        font-size: 28px;
        font-weight: 800;
        color: #145c38 !important;
    }

    .confidence {
        font-size: 22px;
        font-weight: 750;
        color: #16804c !important;
    }

    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid #198754 !important;
        background: #198754 !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 9px 20px !important;
    }

    .stButton > button:hover {
        background: #146c43 !important;
        border-color: #146c43 !important;
        color: white !important;
    }

    /* =====================================================
       INPUTS
       ===================================================== */

    .stTextInput input,
    .stTextArea textarea {
        background: white !important;
        color: #17352a !important;
        border: 1px solid #bcd1c4 !important;
        border-radius: 10px !important;
    }

    .stSelectbox div[data-baseweb="select"] {
        background: white !important;
    }

    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 15px;
        padding: 10px;
        border: 1px dashed #9bc7aa;
    }

    /* =====================================================
       TABLE
       ===================================================== */

    .info-table {
        width: 100%;
        border-collapse: collapse;
        background: white;
        border-radius: 12px;
        overflow: hidden;
    }

    .info-table th {
        background: #e2f3e8;
        color: #174a34 !important;
        padding: 13px;
        text-align: left;
    }

    .info-table td {
        border-top: 1px solid #e1ebe5;
        padding: 13px;
        color: #344b40 !important;
    }

    /* =====================================================
       CHAT
       ===================================================== */

    .chat-user {
        background: #dff4e7;
        border-radius: 15px 15px 3px 15px;
        padding: 14px 17px;
        margin: 8px 0 8px 18%;
        color: #174a34 !important;
    }

    .chat-ai {
        background: white;
        border: 1px solid #d8e7de;
        border-radius: 15px 15px 15px 3px;
        padding: 14px 17px;
        margin: 8px 18% 8px 0;
        color: #344b40 !important;
    }

    /* =====================================================
       MOBILE / SMALL LAPTOP
       ===================================================== */

    @media (max-width: 900px) {
        .hero {
            padding: 25px;
        }

        .hero h1 {
            font-size: 36px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# DISEASE DATABASE
# =========================================================

disease_info = {

    "Pepper__bell___Bacterial_spot": {
        "name": "Pepper Bacterial Spot",
        "plant": "Pepper",
        "type": "Bacterial disease",
        "description": "A bacterial disease that causes spots on pepper leaves and fruit.",
        "symptoms": [
            "Small dark leaf spots",
            "Yellowing around infected areas",
            "Spots may enlarge over time",
            "Fruit lesions can appear"
        ],
        "what_to_do": [
            "Remove heavily affected leaves",
            "Avoid working with wet plants",
            "Improve air circulation",
            "Use appropriate bacterial disease management practices"
        ],
        "prevention": [
            "Use clean seeds",
            "Avoid overhead watering",
            "Keep foliage dry",
            "Remove infected plant material"
        ]
    },

    "Pepper__bell___healthy": {
        "name": "Healthy Pepper",
        "plant": "Pepper",
        "type": "Healthy",
        "description": "The model identifies this image as a healthy pepper leaf.",
        "symptoms": [
            "Healthy green color",
            "No obvious disease spots",
            "Normal leaf structure"
        ],
        "what_to_do": [
            "Continue regular plant care",
            "Monitor leaves regularly",
            "Maintain appropriate watering"
        ],
        "prevention": [
            "Provide good sunlight",
            "Avoid overwatering",
            "Inspect plants regularly"
        ]
    },

    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "plant": "Potato",
        "type": "Fungal disease",
        "description": "A fungal disease commonly associated with dark lesions on potato leaves.",
        "symptoms": [
            "Dark circular spots",
            "Concentric ring patterns",
            "Yellowing around lesions",
            "Older leaves may be affected first"
        ],
        "what_to_do": [
            "Remove severely affected foliage",
            "Improve air circulation",
            "Avoid overhead watering",
            "Follow appropriate fungicide guidance where needed"
        ],
        "prevention": [
            "Rotate crops",
            "Remove infected plant debris",
            "Maintain plant spacing",
            "Avoid prolonged leaf wetness"
        ]
    },

    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "plant": "Potato",
        "type": "Fungal-like pathogen disease",
        "description": "A serious potato disease that can spread quickly under cool and wet conditions.",
        "symptoms": [
            "Dark irregular leaf lesions",
            "Rapid leaf damage",
            "Brown or black affected areas",
            "White growth may appear under humid conditions"
        ],
        "what_to_do": [
            "Remove severely infected material",
            "Avoid overhead irrigation",
            "Improve air circulation",
            "Seek local agricultural guidance for treatment"
        ],
        "prevention": [
            "Use healthy planting material",
            "Avoid excessive moisture",
            "Monitor plants frequently",
            "Remove infected debris"
        ]
    },

    "Potato___healthy": {
        "name": "Healthy Potato",
        "plant": "Potato",
        "type": "Healthy",
        "description": "The model identifies this potato leaf as healthy.",
        "symptoms": [
            "Healthy green foliage",
            "No obvious lesions",
            "Normal leaf appearance"
        ],
        "what_to_do": [
            "Continue normal care",
            "Monitor for changes",
            "Maintain balanced watering"
        ],
        "prevention": [
            "Use healthy planting material",
            "Keep the growing area clean",
            "Monitor regularly"
        ]
    },

    "Tomato_Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "plant": "Tomato",
        "type": "Bacterial disease",
        "description": "A bacterial disease that produces small dark spots on tomato leaves and fruit.",
        "symptoms": [
            "Small dark leaf spots",
            "Yellow halos",
            "Leaf damage",
            "Fruit spots"
        ],
        "what_to_do": [
            "Remove severely affected leaves",
            "Avoid handling plants when wet",
            "Improve airflow",
            "Use appropriate disease management"
        ],
        "prevention": [
            "Use clean seeds",
            "Avoid overhead watering",
            "Remove plant debris",
            "Provide adequate spacing"
        ]
    },

    "Tomato_Early_blight": {
        "name": "Tomato Early Blight",
        "plant": "Tomato",
        "type": "Fungal disease",
        "description": "A fungal disease that often creates dark circular lesions on tomato leaves.",
        "symptoms": [
            "Brown or dark circular spots",
            "Concentric rings",
            "Yellowing",
            "Lower leaves often affected first"
        ],
        "what_to_do": [
            "Remove badly affected leaves",
            "Keep foliage dry",
            "Improve air circulation",
            "Use appropriate treatment if recommended"
        ],
        "prevention": [
            "Rotate crops",
            "Remove infected debris",
            "Water at soil level",
            "Maintain good spacing"
        ]
    },

    "Tomato_Late_blight": {
        "name": "Tomato Late Blight",
        "plant": "Tomato",
        "type": "Fungal-like pathogen disease",
        "description": "A rapidly developing disease favored by cool and humid conditions.",
        "symptoms": [
            "Large dark lesions",
            "Rapid leaf browning",
            "Stem lesions",
            "Severe foliage damage"
        ],
        "what_to_do": [
            "Remove severely affected tissue",
            "Reduce leaf wetness",
            "Improve ventilation",
            "Seek agricultural advice for treatment"
        ],
        "prevention": [
            "Monitor plants frequently",
            "Avoid overhead watering",
            "Remove infected material",
            "Maintain spacing"
        ]
    },

    "Tomato_Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "plant": "Tomato",
        "type": "Fungal disease",
        "description": "A fungal disease commonly associated with high humidity.",
        "symptoms": [
            "Yellow patches on upper leaf surfaces",
            "Fuzzy growth underneath leaves",
            "Leaf yellowing",
            "Premature leaf drop"
        ],
        "what_to_do": [
            "Improve ventilation",
            "Reduce humidity",
            "Remove affected leaves",
            "Avoid wetting leaves"
        ],
        "prevention": [
            "Provide adequate spacing",
            "Improve greenhouse ventilation",
            "Water at soil level",
            "Remove plant debris"
        ]
    },

    "Tomato_Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "plant": "Tomato",
        "type": "Fungal disease",
        "description": "A fungal disease that produces numerous small spots on tomato leaves.",
        "symptoms": [
            "Small circular spots",
            "Dark borders",
            "Light centers",
            "Lower leaves may be affected first"
        ],
        "what_to_do": [
            "Remove infected leaves",
            "Keep foliage dry",
            "Improve airflow",
            "Clean infected plant debris"
        ],
        "prevention": [
            "Crop rotation",
            "Avoid overhead watering",
            "Maintain spacing",
            "Sanitize gardening tools"
        ]
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Tomato Spider Mites",
        "plant": "Tomato",
        "type": "Pest",
        "description": "Spider mites are tiny pests that feed on plant tissues and can cause leaf damage.",
        "symptoms": [
            "Fine speckling",
            "Yellow or bronze leaves",
            "Leaf drying",
            "Fine webbing in severe infestations"
        ],
        "what_to_do": [
            "Inspect leaf undersides",
            "Wash foliage with water when appropriate",
            "Remove heavily affected leaves",
            "Use suitable pest management guidance"
        ],
        "prevention": [
            "Monitor plants regularly",
            "Reduce excessive plant stress",
            "Encourage beneficial insects",
            "Keep plants adequately watered"
        ]
    },

    "Tomato__Target_Spot": {
        "name": "Tomato Target Spot",
        "plant": "Tomato",
        "type": "Fungal disease",
        "description": "A fungal disease that produces target-like spots on tomato foliage.",
        "symptoms": [
            "Circular brown spots",
            "Target-like rings",
            "Leaf yellowing",
            "Possible fruit lesions"
        ],
        "what_to_do": [
            "Remove severely affected leaves",
            "Improve airflow",
            "Avoid wet foliage",
            "Follow suitable disease management practices"
        ],
        "prevention": [
            "Maintain plant spacing",
            "Reduce humidity",
            "Remove plant debris",
            "Use clean planting material"
        ]
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "plant": "Tomato",
        "type": "Viral disease",
        "description": "A viral disease that can cause curling, yellowing and reduced growth.",
        "symptoms": [
            "Upward curling leaves",
            "Yellowing",
            "Reduced plant growth",
            "Small or poorly developing plants"
        ],
        "what_to_do": [
            "Remove severely infected plants where appropriate",
            "Control whitefly populations",
            "Separate affected plants",
            "Monitor nearby plants"
        ],
        "prevention": [
            "Control insect vectors",
            "Use healthy seedlings",
            "Remove infected plant material",
            "Monitor plants early"
        ]
    },

    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "plant": "Tomato",
        "type": "Viral disease",
        "description": "A viral disease that may produce mottled or mosaic patterns on leaves.",
        "symptoms": [
            "Mosaic leaf patterns",
            "Uneven green coloration",
            "Leaf distortion",
            "Reduced plant vigor"
        ],
        "what_to_do": [
            "Remove affected plants when appropriate",
            "Avoid spreading plant sap",
            "Clean tools",
            "Monitor surrounding plants"
        ],
        "prevention": [
            "Use clean seeds",
            "Sanitize tools",
            "Wash hands after handling plants",
            "Remove infected material"
        ]
    },

    "Tomato_healthy": {
        "name": "Healthy Tomato",
        "plant": "Tomato",
        "type": "Healthy",
        "description": "The model identifies this tomato leaf as healthy.",
        "symptoms": [
            "Healthy green appearance",
            "No obvious disease lesions",
            "Normal leaf structure"
        ],
        "what_to_do": [
            "Continue normal plant care",
            "Monitor leaves regularly",
            "Maintain appropriate watering"
        ],
        "prevention": [
            "Provide good sunlight",
            "Avoid excessive watering",
            "Maintain airflow",
            "Inspect regularly"
        ]
    }
}

# =========================================================
# Q&A DATABASE
# =========================================================

qa_database = {
    "yellow leaves": (
        "Yellow leaves can have several causes, including overwatering, "
        "nutrient problems, natural aging, pests, or disease. Check the "
        "soil moisture, inspect both sides of the leaves, and look for spots "
        "or insects before deciding on treatment."
    ),

    "watering": (
        "Water plants when the soil needs it rather than following a rigid "
        "schedule. Check the upper layer of soil and make sure containers "
        "have good drainage. Avoid keeping roots constantly waterlogged."
    ),

    "blight": (
        "Blight is a general term used for several plant diseases. Early "
        "blight and late blight can affect tomato and potato plants. Look "
        "for dark lesions, yellowing, rapid spread, and environmental "
        "conditions that favor disease."
    ),

    "leaf spots": (
        "Leaf spots can be caused by fungi, bacteria, pests, or other stress. "
        "Inspect the shape, color, location and progression of the spots. "
        "Avoid wetting foliage and remove severely affected leaves when "
        "appropriate."
    ),

    "fungi": (
        "Many fungal diseases become worse when foliage stays wet for long "
        "periods. Good airflow, adequate spacing, clean plant debris and "
        "watering at soil level can help reduce disease pressure."
    ),

    "pests": (
        "Inspect both the upper and lower surfaces of leaves. Common signs "
        "include speckling, curling, sticky residue, holes or webbing. "
        "Early identification helps prevent pests from spreading."
    ),

    "healthy plants": (
        "Healthy plants generally have strong growth, appropriate leaf "
        "color for the species, good root-zone drainage and no obvious "
        "pest or disease symptoms. Regular inspection is one of the best "
        "ways to catch problems early."
    ),

    "tomato": (
        "Tomatoes need good sunlight, appropriate watering, airflow and "
        "regular inspection for fungal diseases, bacterial diseases, "
        "viruses and pests."
    ),

    "potato": (
        "Potatoes should be monitored for early blight and late blight. "
        "Avoid prolonged leaf wetness, provide adequate spacing and remove "
        "infected plant material when appropriate."
    ),

    "pepper": (
        "Pepper plants benefit from good sunlight, well-draining soil and "
        "consistent care. Watch for bacterial spots, pests and signs of "
        "water stress."
    )
}

# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():
    model_path = tf.keras.utils.get_file(
        "plant_disease_model.keras",
        MODEL_URL
    )
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_class_names():
    possible_paths = [
        "class_names.json",
        os.path.join(os.path.dirname(__file__), "class_names.json")
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    names = json.load(f)

                if isinstance(names, dict):
                    return list(names.values())

                if isinstance(names, list):
                    return names

            except Exception:
                pass

    return EXPECTED_CLASSES


# =========================================================
# LOAD MODEL
# =========================================================

try:
    model = load_model()
    class_names = load_class_names()
    model_error = None

except Exception as e:
    model = None
    class_names = EXPECTED_CLASSES
    model_error = str(e)


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "history" not in st.session_state:
    st.session_state.history = []

if "plants" not in st.session_state:
    st.session_state.plants = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">🌿 FloraSense</div>
            <div class="sidebar-brand-subtitle">
                Detect. Diagnose. Protect.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">Navigation</div>',
        unsafe_allow_html=True
    )

    navigation = [
        ("🏠 Home", "Home"),
        ("📷 AI Scanner", "AI Scanner"),
        ("🩺 Diagnosis", "Diagnosis"),
        ("🌱 My Plants", "My Plants"),
        ("🦠 Disease Library", "Disease Library"),
        ("🌿 Plant Library", "Plant Library"),
        ("📊 History", "History"),
        ("📚 Learn", "Learn"),
        ("💧 Care", "Care"),
        ("🌦️ Weather", "Weather"),
        ("🤖 FloraSense AI", "FloraSense AI"),
        ("🧪 How AI Works", "How AI Works"),
        ("📈 Model Performance", "Model Performance"),
        ("⚙️ Settings", "Settings"),
        ("ℹ️ About", "About")
    ]

    for label, page_name in navigation:

        if st.button(
            label,
            key="nav_" + page_name,
            use_container_width=True
        ):
            st.session_state.page = page_name
            st.rerun()

    st.markdown(
        """
        <div class="sidebar-footer">
            <strong>FloraSense</strong><br>
            <span>AI Plant Health Assistant</span><br>
            <span>v1.0</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_disease_data(predicted_class):
    return disease_info.get(
        predicted_class,
        {
            "name": predicted_class.replace("_", " "),
            "plant": "Plant",
            "type": "Unknown",
            "description": "No additional information is available.",
            "symptoms": ["Refer to the image and prediction carefully."],
            "what_to_do": ["Monitor the plant and seek expert advice if needed."],
            "prevention": ["Maintain good plant hygiene and regular monitoring."]
        }
    )


def friendly_class_name(name):
    if name in disease_info:
        return disease_info[name]["name"]

    return name.replace("_", " ").replace("__", " ")


def run_prediction(uploaded_file):

    image = Image.open(uploaded_file).convert("RGB")

    resized = image.resize(IMG_SIZE)

    image_array = np.array(resized).astype("float32") / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(np.argmax(probabilities))

    confidence = float(probabilities[predicted_index] * 100)

    if predicted_index < len(class_names):
        predicted_class = class_names[predicted_index]
    else:
        predicted_class = EXPECTED_CLASSES[predicted_index]

    return image, predicted_class, confidence


def show_disease_details(predicted_class, confidence):

    info = get_disease_data(predicted_class)

    st.markdown(
        """
        <div class="result-card">
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="badge">AI ANALYSIS RESULT</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="result-title">{info["name"]}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<p><strong>Plant:</strong> {info["plant"]}</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<p><strong>Type:</strong> {info["type"]}</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="confidence">Confidence: {confidence:.2f}%</div>',
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if "healthy" in predicted_class.lower():
        st.success("🌱 Health indicator: The plant appears healthy according to the model.")
    elif confidence >= 85:
        st.warning("🩺 Health indicator: A disease was detected with high model confidence.")
    elif confidence >= 60:
        st.warning("⚠️ Health indicator: The prediction should be checked carefully.")
    else:
        st.info("🔎 Health indicator: The model has lower confidence. Consider another clear image.")

    st.markdown("### What is it?")
    st.write(info["description"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔍 Common Symptoms")

        for item in info["symptoms"]:
            st.markdown(f"- {item}")

    with col2:
        st.markdown("### 🛠️ What to Do")

        for item in info["what_to_do"]:
            st.markdown(f"- {item}")

    st.markdown("### 🛡️ Prevention")

    for item in info["prevention"]:
        st.markdown(f"- {item}")


# =========================================================
# HOME
# =========================================================

if st.session_state.page == "Home":

    st.markdown(
        """
        <div class="hero">
            <div class="badge">AI-POWERED PLANT HEALTH</div>
            <h1>🌿 Welcome to FloraSense</h1>
            <p>
                Detect plant diseases, understand symptoms and learn how
                to protect your plants with AI-powered leaf analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-number">15</div>
                <div class="stat-label">Disease Classes</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-number">91.3%</div>
                <div class="stat-label">Validation Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-number">3</div>
                <div class="stat-label">Plant Types</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-number">224²</div>
                <div class="stat-label">Image Analysis</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">🌱 Supported Plants</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(
            """
            <div class="card">
                <h3>🍅 Tomato</h3>
                <p>
                    Detect common tomato diseases including blight,
                    leaf mold, bacterial spot and viral diseases.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p2:
        st.markdown(
            """
            <div class="card">
                <h3>🥔 Potato</h3>
                <p>
                    Analyze potato leaves for early blight,
                    late blight and healthy foliage.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p3:
        st.markdown(
            """
            <div class="card">
                <h3>🌶️ Pepper</h3>
                <p>
                    Identify pepper bacterial spot and healthy
                    pepper leaves.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">🚀 How FloraSense Helps</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            """
            <div class="card">
                <h3>📷 Scan</h3>
                <p>
                    Upload a clear photo of a plant leaf and let
                    the AI analyze it.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b:
        st.markdown(
            """
            <div class="card">
                <h3>🩺 Diagnose</h3>
                <p>
                    Get the predicted disease and confidence
                    score along with useful information.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c:
        st.markdown(
            """
            <div class="card">
                <h3>🛡️ Protect</h3>
                <p>
                    Learn symptoms, possible actions and
                    prevention practices.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# AI SCANNER
# =========================================================

elif st.session_state.page == "AI Scanner":

    st.title("📷 AI Plant Scanner")

    st.write(
        "Upload a clear JPG, JPEG or PNG image of a tomato, potato or pepper leaf."
    )

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        key="scanner_upload"
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                image,
                caption="Uploaded plant image",
                use_container_width=True
            )

        with col2:
            st.markdown(
                """
                <div class="card">
                    <h3>🔬 Ready for Analysis</h3>
                    <p>
                        FloraSense will resize the image to 224 × 224
                        pixels and analyze it using the trained neural network.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if model is None:
                st.error(
                    "The AI model could not be loaded. "
                    "Please check the deployment logs."
                )
            else:

                if st.button(
                    "🔍 Analyze Plant",
                    key="analyze_button",
                    use_container_width=True
                ):

                    with st.spinner("Analyzing the leaf image..."):

                        try:

                            analyzed_image, predicted_class, confidence = run_prediction(
                                uploaded_file
                            )

                            st.session_state.last_prediction = predicted_class
                            st.session_state.last_confidence = confidence

                            st.session_state.history.append(
                                {
                                    "disease": friendly_class_name(predicted_class),
                                    "confidence": confidence
                                }
                            )

                            st.success("Analysis completed!")

                            show_disease_details(
                                predicted_class,
                                confidence
                            )

                        except Exception as e:

                            st.error(
                                f"Prediction failed: {str(e)}"
                            )


# =========================================================
# DIAGNOSIS
# =========================================================

elif st.session_state.page == "Diagnosis":

    st.title("🩺 Plant Health Diagnosis")

    st.write(
        "Review the latest AI diagnosis from the scanner."
    )

    if "last_prediction" not in st.session_state:

        st.info(
            "No diagnosis is available yet. Go to 📷 AI Scanner and upload a leaf image."
        )

    else:

        predicted_class = st.session_state.last_prediction
        confidence = st.session_state.last_confidence

        show_disease_details(
            predicted_class,
            confidence
        )


# =========================================================
# MY PLANTS
# =========================================================

elif st.session_state.page == "My Plants":

    st.title("🌱 My Plants")

    st.write(
        "Keep a simple list of the plants you want to monitor."
    )

    plant_name = st.text_input(
        "Plant name",
        placeholder="Example: Tomato Plant 1"
    )

    plant_type = st.selectbox(
        "Plant type",
        ["Tomato", "Potato", "Pepper"]
    )

    if st.button("➕ Add Plant"):

        if plant_name.strip():

            st.session_state.plants.append(
                {
                    "name": plant_name.strip(),
                    "type": plant_type
                }
            )

            st.success("Plant added successfully!")

    st.markdown("### Your Plants")

    if not st.session_state.plants:

        st.info(
            "You have not added any plants yet."
        )

    else:

        for index, plant in enumerate(st.session_state.plants):

            st.markdown(
                f"""
                <div class="card">
                    <h3>🌱 {plant["name"]}</h3>
                    <p>
                        Plant type: <strong>{plant["type"]}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# DISEASE LIBRARY
# =========================================================

elif st.session_state.page == "Disease Library":

    st.title("🦠 Disease Library")

    st.write(
        "Explore the diseases recognized by the FloraSense AI model."
    )

    search = st.text_input(
        "🔎 Search diseases",
        placeholder="Search tomato, blight, bacterial, healthy..."
    ).lower()

    filtered = []

    for key, info in disease_info.items():

        searchable = (
            info["name"] + " "
            + info["plant"] + " "
            + info["type"] + " "
            + info["description"]
        ).lower()

        if not search or search in searchable:
            filtered.append((key, info))

    st.write(f"Showing {len(filtered)} disease classes.")

    for key, info in filtered:

        with st.container():

            st.markdown(
                f"""
                <div class="card">
                    <div class="badge">{info["plant"]}</div>
                    <h3>{info["name"]}</h3>
                    <p>
                        <strong>Type:</strong> {info["type"]}
                    </p>
                    <p>{info["description"]}</p>

                    <p>
                        <strong>Symptoms:</strong>
                        {" • ".join(info["symptoms"])}
                    </p>

                    <p>
                        <strong>Prevention:</strong>
                        {" • ".join(info["prevention"])}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# PLANT LIBRARY
# =========================================================

elif st.session_state.page == "Plant Library":

    st.title("🌿 Plant Library")

    st.write(
        "Learn about the three plant groups supported by your AI model."
    )

    plants = [
        (
            "🍅 Tomato",
            "Tomatoes are warm-season plants that benefit from sunlight, "
            "consistent watering and good airflow."
        ),
        (
            "🥔 Potato",
            "Potatoes require well-draining soil and should be monitored "
            "carefully for blight symptoms."
        ),
        (
            "🌶️ Pepper",
            "Pepper plants prefer warm conditions, sunlight and consistent "
            "moisture without waterlogging."
        )
    ]

    for name, description in plants:

        st.markdown(
            f"""
            <div class="card">
                <h3>{name}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# HISTORY
# =========================================================

elif st.session_state.page == "History":

    st.title("📊 Detection History")

    st.write(
        "Previous predictions from this browser session appear here."
    )

    if not st.session_state.history:

        st.info(
            "No detection history yet."
        )

    else:

        for index, item in enumerate(
            reversed(st.session_state.history),
            start=1
        ):

            st.markdown(
                f"""
                <div class="card">
                    <h3>#{index} — {item["disease"]}</h3>
                    <p>
                        Confidence:
                        <strong>{item["confidence"]:.2f}%</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# LEARN
# =========================================================

elif st.session_state.page == "Learn":

    st.title("📚 Learn About Plant Health")

    st.write(
        "Simple guides for understanding common plant problems."
    )

    topics = [
        (
            "🔍 Inspecting Leaves",
            "Check both sides of leaves regularly for spots, discoloration, "
            "pests, curling or unusual growth."
        ),
        (
            "💧 Watering",
            "Avoid both extreme dryness and constant waterlogging. "
            "Check soil moisture before watering."
        ),
        (
            "🌬️ Airflow",
            "Good spacing and airflow can help reduce prolonged leaf wetness "
            "and disease pressure."
        ),
        (
            "🧹 Plant Hygiene",
            "Remove severely infected plant debris and clean gardening tools "
            "to reduce opportunities for disease spread."
        )
    ]

    for title, text in topics:

        st.markdown(
            f"""
            <div class="card">
                <h3>{title}</h3>
                <p>{text}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# CARE
# =========================================================

elif st.session_state.page == "Care":

    st.title("💧 Plant Care Guide")

    st.write(
        "General care guidance for the plants supported by FloraSense."
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            <div class="card">
                <h3>💧 Watering</h3>
                <p>
                    Check the soil before watering. Avoid keeping the
                    root zone continuously waterlogged.
                </p>
            </div>

            <div class="card">
                <h3>☀️ Light</h3>
                <p>
                    Provide suitable sunlight for the plant species
                    and growing environment.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="card">
                <h3>🌬️ Airflow</h3>
                <p>
                    Give plants enough space for air movement around
                    the leaves.
                </p>
            </div>

            <div class="card">
                <h3>🔎 Inspection</h3>
                <p>
                    Inspect plants regularly so disease and pest
                    problems can be noticed early.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# WEATHER
# =========================================================

elif st.session_state.page == "Weather":

    st.title("🌦️ Plant Weather Guide")

    st.write(
        "This page provides general guidance for adjusting plant care "
        "based on weather conditions."
    )

    weather = st.selectbox(
        "Current condition",
        [
            "☀️ Sunny and hot",
            "🌧️ Rainy",
            "☁️ Cloudy",
            "🌬️ Windy",
            "🌤️ Mild"
        ]
    )

    advice = {
        "☀️ Sunny and hot":
            "Monitor soil moisture and avoid unnecessary leaf wetting during hot periods.",

        "🌧️ Rainy":
            "Watch for prolonged leaf wetness and poor drainage, which can increase disease pressure.",

        "☁️ Cloudy":
            "Monitor humidity and airflow, especially around dense foliage.",

        "🌬️ Windy":
            "Check plants for physical damage and make sure containers or supports are stable.",

        "🌤️ Mild":
            "Continue normal monitoring and maintain balanced watering."
    }

    st.markdown(
        f"""
        <div class="card">
            <h3>🌱 Care Recommendation</h3>
            <p>{advice[weather]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Weather information on this page is general guidance and is not a live weather forecast."
    )


# =========================================================
# FLORASENSE AI
# =========================================================

elif st.session_state.page == "FloraSense AI":

    st.title("🤖 FloraSense AI")

    st.write(
        "Ask FloraSense about common plant-health questions."
    )

    question = st.text_input(
        "Ask your plant question",
        placeholder="Example: Why are my tomato leaves yellow?"
    )

    if st.button("💬 Ask FloraSense"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            q = question.lower()

            answer = None

            for keyword, response in qa_database.items():

                if keyword in q:
                    answer = response
                    break

            if answer is None:

                if "tomato" in q:
                    answer = qa_database["tomato"]

                elif "potato" in q:
                    answer = qa_database["potato"]

                elif "pepper" in q:
                    answer = qa_database["pepper"]

                else:
                    answer = (
                        "I can help with common questions about tomato, "
                        "potato and pepper plants, including yellow leaves, "
                        "watering, blight, leaf spots, fungi, pests and "
                        "healthy plant care."
                    )

            st.markdown(
                f"""
                <div class="chat-user">
                    <strong>You</strong><br>
                    {question}
                </div>

                <div class="chat-ai">
                    <strong>🌿 FloraSense AI</strong><br>
                    {answer}
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# HOW AI WORKS
# =========================================================

elif st.session_state.page == "How AI Works":

    st.title("🧪 How AI Works")

    st.write(
        "FloraSense uses a TensorFlow/Keras image-classification model "
        "trained to recognize 15 plant-health classes."
    )

    steps = [
        (
            "1",
            "📷 Upload Image",
            "You provide a clear image of a plant leaf."
        ),
        (
            "2",
            "📐 Resize",
            "The image is resized to 224 × 224 pixels."
        ),
        (
            "3",
            "🧠 Neural Network",
            "The trained neural network analyzes visual patterns in the leaf."
        ),
        (
            "4",
            "📊 Probabilities",
            "The model produces probability scores for the available classes."
        ),
        (
            "5",
            "🩺 Prediction",
            "The class with the highest probability becomes the displayed prediction."
        )
    ]

    for number, title, description in steps:

        st.markdown(
            f"""
            <div class="card">
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif st.session_state.page == "Model Performance":

    st.title("📈 Model Performance")

    st.write(
        "Key information about the trained plant disease classification model."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-number">91.3%</div>
                <div class="stat-label">Validation Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-number">15</div>
                <div class="stat-label">Classes</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-number">224×224</div>
                <div class="stat-label">Input Size</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="card">
            <h3>Model Summary</h3>
            <p>
                FloraSense uses a TensorFlow/Keras image classification
                model trained on plant leaf images. The model recognizes
                15 classes covering tomato, potato and pepper plants.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Recognized Classes")

    for index, class_name in enumerate(class_names, start=1):

        st.markdown(
            f"{index}. **{friendly_class_name(class_name)}**"
        )


# =========================================================
# SETTINGS
# =========================================================

elif st.session_state.page == "Settings":

    st.title("⚙️ Settings")

    st.write(
        "Application preferences."
    )

    st.checkbox(
        "Show confidence percentage",
        value=True,
        key="show_confidence"
    )

    st.checkbox(
        "Show prevention guidance",
        value=True,
        key="show_prevention"
    )

    st.selectbox(
        "Interface language",
        ["English"],
        key="language"
    )

    st.info(
        "Settings currently apply only to this browser session."
    )


# =========================================================
# ABOUT
# =========================================================

elif st.session_state.page == "About":

    st.title("ℹ️ About FloraSense")

    st.markdown(
        """
        <div class="hero">
            <div class="badge">FLORASENSE v1.0</div>
            <h1>🌿 FloraSense</h1>
            <p>
                Detect. Diagnose. Protect.
            </p>
            <p>
                FloraSense is an AI-powered plant health assistant
                designed to help identify common diseases affecting
                tomato, potato and pepper leaves.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🎯 Project Purpose")

    st.write(
        "The goal of FloraSense is to combine machine learning with "
        "simple plant-health information so users can better understand "
        "potential problems affecting their plants."
    )

    st.markdown("### 🧠 Technology")

    st.markdown(
        """
        - TensorFlow / Keras
        - Image classification
        - 224 × 224 image input
        - 15 disease and health classes
        - Streamlit interface
        """
    )

    st.markdown("### ⚠️ Important Note")

    st.info(
        "AI predictions are informational and should not replace advice "
        "from a qualified agricultural professional. Image quality and "
        "plant conditions can affect prediction accuracy."
    )
