import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FloraSense",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS
# ============================================================

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

# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "history" not in st.session_state:
    st.session_state.history = []

if "plants" not in st.session_state:
    st.session_state.plants = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0

if "show_confidence" not in st.session_state:
    st.session_state.show_confidence = True

if "show_prevention" not in st.session_state:
    st.session_state.show_prevention = True

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   GLOBAL
   ========================================================== */

.stApp {
    background:
        radial-gradient(circle at top right, rgba(94, 196, 132, 0.12), transparent 28%),
        radial-gradient(circle at bottom left, rgba(46, 125, 79, 0.08), transparent 30%),
        #f5f8f6;
    color: #19352a;
}

.main .block-container {
    max-width: 1450px;
    padding: 2rem 2.5rem 4rem 2.5rem;
}

h1, h2, h3, h4 {
    color: #123b29 !important;
    font-weight: 800 !important;
}

p, li {
    color: #40564c !important;
    line-height: 1.7;
}

hr {
    border-color: #dce8e0 !important;
}

/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0b3323 0%,
            #12452f 45%,
            #0c3826 100%
        ) !important;

    min-width: 290px !important;
    width: 290px !important;
}

section[data-testid="stSidebar"] > div {
    width: 290px !important;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

.sidebar-brand {
    padding: 10px 10px 20px 10px;
    border-bottom: 1px solid rgba(255,255,255,0.16);
    margin-bottom: 17px;
}

.sidebar-title {
    color: #ffffff !important;
    font-size: 29px;
    font-weight: 850;
    letter-spacing: -0.7px;
}

.sidebar-tagline {
    color: #bfe6ce !important;
    font-size: 12px;
    margin-top: 4px;
    font-weight: 600;
}

.sidebar-section-title {
    color: #91c9a7 !important;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 16px 8px 7px 8px;
}

section[data-testid="stSidebar"] .stButton {
    width: 100%;
}

section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    min-height: 42px !important;

    background: transparent !important;
    color: #ffffff !important;

    border: 1px solid transparent !important;
    border-radius: 11px !important;

    text-align: left !important;

    padding: 9px 12px !important;
    margin: 2px 0 !important;

    font-size: 14px !important;
    font-weight: 650 !important;

    box-shadow: none !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.12) !important;
    border-color: rgba(255,255,255,0.10) !important;
}

section[data-testid="stSidebar"] .stButton > button:focus {
    background: rgba(255,255,255,0.16) !important;
    color: white !important;
}

.sidebar-footer {
    margin-top: 25px;
    padding: 17px 10px;
    border-top: 1px solid rgba(255,255,255,0.15);
}

.sidebar-footer-title {
    color: white !important;
    font-size: 13px;
    font-weight: 750;
}

.sidebar-footer-text {
    color: #abd8bb !important;
    font-size: 11px;
    line-height: 1.7;
}

/* ==========================================================
   HERO
   ========================================================== */

.hero {
    position: relative;
    overflow: hidden;

    background:
        linear-gradient(
            135deg,
            #d9f2e2 0%,
            #edf9f1 45%,
            #c6e9d2 100%
        );

    border: 1px solid #b9dfc6;
    border-radius: 28px;

    padding: 46px 48px;
    margin-bottom: 26px;

    box-shadow: 0 15px 40px rgba(25, 88, 52, 0.08);
}

.hero:after {
    content: "🌿";
    position: absolute;
    right: 45px;
    bottom: -25px;
    font-size: 150px;
    opacity: 0.12;
}

.hero-eyebrow {
    display: inline-block;

    background: #ffffff;
    color: #177045 !important;

    padding: 7px 13px;
    border-radius: 30px;

    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
}

.hero h1 {
    color: #103b28 !important;
    font-size: 48px !important;
    line-height: 1.1;
    margin: 16px 0 10px 0;
}

.hero p {
    color: #3a594a !important;
    font-size: 17px;
    max-width: 820px;
}

/* ==========================================================
   PAGE HEADER
   ========================================================== */

.page-header {
    background: white;
    border: 1px solid #dce8e0;
    border-radius: 20px;

    padding: 25px 28px;
    margin-bottom: 24px;

    box-shadow: 0 7px 25px rgba(20, 70, 45, 0.05);
}

.page-header h1 {
    margin: 0;
    color: #123b29 !important;
}

.page-header p {
    margin: 7px 0 0 0;
    color: #5b7065 !important;
}

/* ==========================================================
   STAT CARDS
   ========================================================== */

.stat-card {
    background: #ffffff;

    border: 1px solid #dce8e0;
    border-radius: 18px;

    padding: 22px;

    min-height: 125px;

    box-shadow: 0 7px 24px rgba(25, 70, 45, 0.055);
}

.stat-icon {
    font-size: 25px;
}

.stat-number {
    color: #167348 !important;
    font-size: 29px;
    font-weight: 850;
    margin-top: 5px;
}

.stat-label {
    color: #65786e !important;
    font-size: 12px;
    font-weight: 650;
}

/* ==========================================================
   CARDS
   ========================================================== */

.card {
    background: #ffffff;

    border: 1px solid #dce8e0;
    border-radius: 19px;

    padding: 25px;

    margin-bottom: 18px;

    box-shadow: 0 8px 28px rgba(20, 70, 45, 0.055);
}

.card h3 {
    margin-top: 0;
    color: #174c34 !important;
}

.card p {
    color: #4b6257 !important;
}

/* ==========================================================
   FEATURE CARDS
   ========================================================== */

.feature-card {
    background: white;

    border: 1px solid #dbe9df;
    border-radius: 19px;

    padding: 24px;
    min-height: 205px;

    box-shadow: 0 8px 25px rgba(20,70,45,0.05);
}

.feature-icon {
    font-size: 32px;
    margin-bottom: 10px;
}

.feature-title {
    color: #164b33 !important;
    font-size: 18px;
    font-weight: 800;
}

.feature-text {
    color: #5a6f65 !important;
    font-size: 14px;
    line-height: 1.65;
}

/* ==========================================================
   PLANT CARDS
   ========================================================== */

.plant-card {
    background: linear-gradient(
        145deg,
        #ffffff,
        #f3faf5
    );

    border: 1px solid #d6e9dc;
    border-radius: 20px;

    padding: 27px;

    min-height: 235px;

    box-shadow: 0 8px 27px rgba(20,70,45,0.055);
}

.plant-icon {
    font-size: 43px;
}

.plant-name {
    font-size: 22px;
    font-weight: 800;
    color: #154b32 !important;
    margin-top: 10px;
}

.plant-text {
    color: #566d61 !important;
    font-size: 14px;
}

/* ==========================================================
   BADGES
   ========================================================== */

.badge {
    display: inline-block;

    padding: 6px 11px;

    border-radius: 30px;

    background: #e3f4e9;
    color: #176b42 !important;

    font-size: 11px;
    font-weight: 800;
}

.badge-yellow {
    background: #fff2cf;
    color: #8a6413 !important;
}

.badge-red {
    background: #fde5e5;
    color: #9d3030 !important;
}

/* ==========================================================
   RESULT
   ========================================================== */

.result-box {
    background:
        linear-gradient(
            135deg,
            #e4f7eb,
            #ffffff
        );

    border: 2px solid #a9d8b8;
    border-radius: 22px;

    padding: 30px;

    margin-top: 20px;
}

.result-heading {
    font-size: 29px;
    font-weight: 850;
    color: #145b37 !important;
}

.result-confidence {
    font-size: 22px;
    font-weight: 800;
    color: #17804b !important;
}

/* ==========================================================
   PROGRESS
   ========================================================== */

.progress-background {
    background: #dcebe1;
    border-radius: 20px;
    height: 12px;
    overflow: hidden;
}

.progress-fill {
    background: #23965b;
    height: 100%;
    border-radius: 20px;
}

/* ==========================================================
   PIPELINE
   ========================================================== */

.pipeline {
    background: white;

    border: 1px solid #dce8e0;
    border-radius: 18px;

    padding: 22px;

    text-align: center;

    min-height: 175px;

    box-shadow: 0 7px 22px rgba(20,70,45,0.05);
}

.pipeline-number {
    width: 45px;
    height: 45px;

    display: inline-flex;
    align-items: center;
    justify-content: center;

    background: #dff3e6;
    color: #157045 !important;

    border-radius: 50%;

    font-weight: 850;
    font-size: 18px;
}

.pipeline-title {
    margin-top: 12px;

    font-weight: 800;
    color: #184d35 !important;
}

.pipeline-text {
    font-size: 13px;
    color: #60736a !important;
}

/* ==========================================================
   CHAT
   ========================================================== */

.chat-container {
    background: #edf7f0;
    border: 1px solid #d7e9dc;
    border-radius: 20px;
    padding: 20px;
}

.chat-user {
    background: #ccebd7;
    color: #174b34 !important;

    border-radius: 17px 17px 4px 17px;

    padding: 14px 18px;

    margin: 12px 0 12px 18%;
}

.chat-ai {
    background: white;
    color: #40564c !important;

    border: 1px solid #d9e7de;

    border-radius: 17px 17px 17px 4px;

    padding: 15px 18px;

    margin: 12px 18% 12px 0;
}

/* ==========================================================
   STREAMLIT INPUTS
   ========================================================== */

.stTextInput input,
.stTextArea textarea {
    background: white !important;
    color: #19352a !important;

    border: 1px solid #bcd2c3 !important;
    border-radius: 11px !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #7a8c82 !important;
}

.stSelectbox div[data-baseweb="select"] {
    background: white !important;
    color: #19352a !important;
}

.stSelectbox div[data-baseweb="select"] * {
    color: #19352a !important;
}

[data-testid="stFileUploader"] {
    background: white !important;

    border: 2px dashed #a8cfb5 !important;
    border-radius: 17px !important;

    padding: 12px !important;
}

/* ==========================================================
   BUTTONS
   ========================================================== */

.stButton > button {
    background: #198754 !important;
    color: white !important;

    border: 1px solid #198754 !important;
    border-radius: 11px !important;

    font-weight: 750 !important;

    padding: 10px 20px !important;

    transition: 0.2s;
}

.stButton > button:hover {
    background: #146c43 !important;
    border-color: #146c43 !important;
}

/* ==========================================================
   INFO / SUCCESS
   ========================================================== */

div[data-testid="stAlert"] {
    border-radius: 13px !important;
}

/* ==========================================================
   FOOTER
   ========================================================== */

.app-footer {
    text-align: center;

    margin-top: 45px;
    padding-top: 25px;

    border-top: 1px solid #dce8e0;

    color: #708278 !important;
    font-size: 12px;
}

/* ==========================================================
   RESPONSIVE
   ========================================================== */

@media (max-width: 1000px) {

    .main .block-container {
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }

    .hero {
        padding: 30px;
    }

    .hero h1 {
        font-size: 37px !important;
    }

}

</style>
""",
    unsafe_allow_html=True
)

# ============================================================
# DISEASE DATABASE
# ============================================================

disease_info = {

    "Pepper__bell___Bacterial_spot": {
        "name": "Pepper Bacterial Spot",
        "plant": "Pepper",
        "type": "Bacterial Disease",
        "severity": "Moderate",
        "description": "A bacterial disease that can produce dark spots on pepper leaves and fruit.",
        "symptoms": [
            "Small dark spots",
            "Yellow areas around lesions",
            "Leaf damage",
            "Possible fruit lesions"
        ],
        "action": [
            "Remove severely affected leaves",
            "Avoid working with wet foliage",
            "Improve air circulation",
            "Monitor nearby plants"
        ],
        "prevention": [
            "Use clean planting material",
            "Avoid overhead watering",
            "Maintain good plant spacing",
            "Remove infected debris"
        ]
    },

    "Pepper__bell___healthy": {
        "name": "Healthy Pepper",
        "plant": "Pepper",
        "type": "Healthy",
        "severity": "Healthy",
        "description": "The model identifies the pepper leaf as healthy.",
        "symptoms": [
            "Healthy green appearance",
            "No obvious disease lesions",
            "Normal leaf structure"
        ],
        "action": [
            "Continue normal care",
            "Monitor regularly",
            "Maintain appropriate watering"
        ],
        "prevention": [
            "Provide suitable sunlight",
            "Avoid overwatering",
            "Inspect plants regularly"
        ]
    },

    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "plant": "Potato",
        "type": "Fungal Disease",
        "severity": "Moderate",
        "description": "A fungal disease associated with dark lesions and target-like rings on potato leaves.",
        "symptoms": [
            "Dark circular spots",
            "Concentric rings",
            "Yellowing",
            "Older leaves may be affected first"
        ],
        "action": [
            "Remove badly affected foliage",
            "Improve air circulation",
            "Avoid prolonged leaf wetness",
            "Follow appropriate local treatment guidance"
        ],
        "prevention": [
            "Rotate crops",
            "Remove infected debris",
            "Maintain plant spacing",
            "Water near the soil"
        ]
    },

    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "plant": "Potato",
        "type": "Fungal-like Pathogen",
        "severity": "High",
        "description": "A serious potato disease that can develop rapidly under cool and wet conditions.",
        "symptoms": [
            "Dark irregular lesions",
            "Rapid foliage damage",
            "Brown or black affected areas",
            "Possible pale growth under humid conditions"
        ],
        "action": [
            "Remove severely affected material",
            "Reduce prolonged leaf wetness",
            "Improve airflow",
            "Seek local agricultural guidance"
        ],
        "prevention": [
            "Use healthy planting material",
            "Monitor plants frequently",
            "Avoid excessive moisture",
            "Remove infected debris"
        ]
    },

    "Potato___healthy": {
        "name": "Healthy Potato",
        "plant": "Potato",
        "type": "Healthy",
        "severity": "Healthy",
        "description": "The model identifies this potato leaf as healthy.",
        "symptoms": [
            "Healthy green foliage",
            "No obvious lesions",
            "Normal leaf structure"
        ],
        "action": [
            "Continue normal care",
            "Monitor leaves",
            "Maintain balanced watering"
        ],
        "prevention": [
            "Use healthy planting material",
            "Keep growing areas clean",
            "Inspect plants regularly"
        ]
    },

    "Tomato_Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "plant": "Tomato",
        "type": "Bacterial Disease",
        "severity": "Moderate",
        "description": "A bacterial disease that can cause small dark spots on tomato leaves and fruit.",
        "symptoms": [
            "Small dark leaf spots",
            "Yellow halos",
            "Leaf damage",
            "Fruit spots"
        ],
        "action": [
            "Remove severely affected leaves",
            "Avoid handling wet plants",
            "Improve airflow",
            "Monitor surrounding plants"
        ],
        "prevention": [
            "Use clean seeds",
            "Avoid overhead watering",
            "Remove infected debris",
            "Maintain spacing"
        ]
    },

    "Tomato_Early_blight": {
        "name": "Tomato Early Blight",
        "plant": "Tomato",
        "type": "Fungal Disease",
        "severity": "Moderate",
        "description": "A fungal disease that commonly produces dark circular lesions with ring patterns.",
        "symptoms": [
            "Brown circular spots",
            "Concentric rings",
            "Yellowing",
            "Lower leaves often affected first"
        ],
        "action": [
            "Remove badly affected leaves",
            "Keep foliage dry",
            "Improve air circulation",
            "Follow appropriate treatment guidance"
        ],
        "prevention": [
            "Rotate crops",
            "Remove plant debris",
            "Water at soil level",
            "Maintain spacing"
        ]
    },

    "Tomato_Late_blight": {
        "name": "Tomato Late Blight",
        "plant": "Tomato",
        "type": "Fungal-like Pathogen",
        "severity": "High",
        "description": "A rapidly developing disease favored by cool and humid conditions.",
        "symptoms": [
            "Large dark lesions",
            "Rapid leaf browning",
            "Stem lesions",
            "Severe foliage damage"
        ],
        "action": [
            "Remove severely affected tissue",
            "Reduce leaf wetness",
            "Improve ventilation",
            "Seek agricultural advice"
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
        "type": "Fungal Disease",
        "severity": "Moderate",
        "description": "A fungal disease commonly associated with high humidity and poor air circulation.",
        "symptoms": [
            "Yellow patches",
            "Fuzzy growth underneath leaves",
            "Leaf yellowing",
            "Premature leaf drop"
        ],
        "action": [
            "Improve ventilation",
            "Reduce excessive humidity",
            "Remove affected leaves",
            "Avoid wetting foliage"
        ],
        "prevention": [
            "Maintain plant spacing",
            "Improve airflow",
            "Water at soil level",
            "Remove plant debris"
        ]
    },

    "Tomato_Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "plant": "Tomato",
        "type": "Fungal Disease",
        "severity": "Moderate",
        "description": "A fungal disease producing numerous small spots on tomato foliage.",
        "symptoms": [
            "Small circular spots",
            "Dark borders",
            "Light centers",
            "Lower leaves affected first"
        ],
        "action": [
            "Remove infected leaves",
            "Keep foliage dry",
            "Improve airflow",
            "Clean infected debris"
        ],
        "prevention": [
            "Rotate crops",
            "Avoid overhead watering",
            "Maintain spacing",
            "Clean gardening tools"
        ]
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Tomato Two-Spotted Spider Mite",
        "plant": "Tomato",
        "type": "Pest",
        "severity": "Moderate",
        "description": "Tiny pests that feed on plant tissue and may cause speckling and leaf damage.",
        "symptoms": [
            "Fine speckling",
            "Yellow or bronze leaves",
            "Drying leaves",
            "Fine webbing in severe cases"
        ],
        "action": [
            "Inspect leaf undersides",
            "Wash foliage where appropriate",
            "Remove heavily affected leaves",
            "Follow suitable pest management guidance"
        ],
        "prevention": [
            "Monitor plants regularly",
            "Reduce plant stress",
            "Encourage beneficial insects",
            "Maintain appropriate watering"
        ]
    },

    "Tomato__Target_Spot": {
        "name": "Tomato Target Spot",
        "plant": "Tomato",
        "type": "Fungal Disease",
        "severity": "Moderate",
        "description": "A fungal disease producing target-like spots on tomato foliage.",
        "symptoms": [
            "Circular brown spots",
            "Target-like rings",
            "Leaf yellowing",
            "Possible fruit lesions"
        ],
        "action": [
            "Remove severely affected leaves",
            "Improve airflow",
            "Avoid wet foliage",
            "Follow appropriate disease management"
        ],
        "prevention": [
            "Maintain spacing",
            "Reduce humidity",
            "Remove plant debris",
            "Use clean planting material"
        ]
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "plant": "Tomato",
        "type": "Viral Disease",
        "severity": "High",
        "description": "A viral disease that can cause leaf curling, yellowing and reduced plant growth.",
        "symptoms": [
            "Upward curling leaves",
            "Yellowing",
            "Reduced growth",
            "Small or poorly developing plants"
        ],
        "action": [
            "Remove severely infected plants where appropriate",
            "Control whitefly populations",
            "Monitor nearby plants",
            "Use healthy seedlings"
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
        "type": "Viral Disease",
        "severity": "High",
        "description": "A viral disease that can cause mottled or mosaic patterns on tomato leaves.",
        "symptoms": [
            "Mosaic patterns",
            "Uneven green coloration",
            "Leaf distortion",
            "Reduced plant vigor"
        ],
        "action": [
            "Remove affected plants when appropriate",
            "Avoid spreading plant sap",
            "Clean tools",
            "Monitor nearby plants"
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
        "severity": "Healthy",
        "description": "The model identifies this tomato leaf as healthy.",
        "symptoms": [
            "Healthy green appearance",
            "No obvious disease lesions",
            "Normal leaf structure"
        ],
        "action": [
            "Continue normal plant care",
            "Monitor leaves regularly",
            "Maintain appropriate watering"
        ],
        "prevention": [
            "Provide good sunlight",
            "Avoid excessive watering",
            "Maintain airflow",
            "Inspect plants regularly"
        ]
    }
}

# ============================================================
# Q&A KNOWLEDGE
# ============================================================

qa_database = {
    "yellow": (
        "Yellow leaves can have several causes, including overwatering, "
        "nutrient problems, natural aging, pests or disease. Check soil "
        "moisture and inspect both sides of the leaves before deciding "
        "on treatment."
    ),

    "water": (
        "Water based on soil moisture and plant needs rather than a rigid "
        "schedule. Good drainage is important because constantly waterlogged "
        "soil can damage roots."
    ),

    "blight": (
        "Blight is a general term for several plant diseases. Tomato and "
        "potato plants can be affected by early or late blight. Look for "
        "dark lesions, yellowing and rapid disease progression."
    ),

    "spot": (
        "Leaf spots can result from fungal diseases, bacterial diseases, "
        "pests or environmental stress. Examine the shape, color and "
        "location of the spots and monitor whether they are spreading."
    ),

    "fung": (
        "Fungal diseases often become worse when leaves remain wet for "
        "long periods. Good airflow, plant spacing and watering near "
        "the soil can help reduce disease pressure."
    ),

    "pest": (
        "Inspect the upper and lower surfaces of leaves. Speckling, holes, "
        "curling, sticky residue and webbing can indicate pest activity."
    ),

    "tomato": (
        "Tomatoes benefit from sunlight, appropriate watering, good airflow "
        "and regular monitoring for fungal, bacterial, viral and pest problems."
    ),

    "potato": (
        "Potatoes should be monitored carefully for early and late blight. "
        "Good drainage, plant spacing and avoiding prolonged leaf wetness "
        "can help reduce disease pressure."
    ),

    "pepper": (
        "Peppers benefit from sunlight, well-draining soil and consistent "
        "care. Monitor leaves for bacterial spots, pests and signs of water stress."
    )
}

# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = tf.keras.utils.get_file(
        "plant_disease_model.keras",
        MODEL_URL
    )

    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_class_names():

    paths = [
        "class_names.json",
        os.path.join(
            os.path.dirname(__file__),
            "class_names.json"
        )
    ]

    for path in paths:

        if os.path.exists(path):

            try:

                with open(path, "r") as file:
                    data = json.load(file)

                if isinstance(data, dict):
                    return list(data.values())

                if isinstance(data, list):
                    return data

            except Exception:
                pass

    return EXPECTED_CLASSES


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()
    class_names = load_class_names()

    model_error = None

except Exception as e:

    model = None
    class_names = EXPECTED_CLASSES
    model_error = str(e)


# ============================================================
# HELPERS
# ============================================================

def friendly_name(class_name):

    if class_name in disease_info:
        return disease_info[class_name]["name"]

    return (
        class_name
        .replace("__", " ")
        .replace("_", " ")
    )


def get_info(class_name):

    return disease_info.get(
        class_name,
        {
            "name": friendly_name(class_name),
            "plant": "Unknown",
            "type": "Unknown",
            "severity": "Unknown",
            "description": "Additional information is not available.",
            "symptoms": [],
            "action": [],
            "prevention": []
        }
    )


def predict_image(uploaded_file):

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    resized = image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        resized
    ).astype("float32") / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_index] * 100
    )

    if predicted_index < len(class_names):
        predicted_class = class_names[predicted_index]
    else:
        predicted_class = EXPECTED_CLASSES[predicted_index]

    top_indices = np.argsort(
        predictions
    )[-3:][::-1]

    top_predictions = []

    for index in top_indices:

        if index < len(class_names):

            top_predictions.append(
                (
                    friendly_name(class_names[index]),
                    float(predictions[index] * 100)
                )
            )

    return image, predicted_class, confidence, top_predictions


def save_history(predicted_class, confidence):

    st.session_state.history.append(
        {
            "disease": friendly_name(predicted_class),
            "confidence": confidence
        }
    )


def page_header(title, subtitle):

    st.markdown(
        f"""
        <div class="page-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title, subtitle=None):

    st.markdown(
        f'<h2 style="margin-top:30px;">{title}</h2>',
        unsafe_allow_html=True
    )

    if subtitle:

        st.markdown(
            f'<p style="color:#62766b;">{subtitle}</p>',
            unsafe_allow_html=True
        )


def disease_result(predicted_class, confidence, top_predictions=None):

    info = get_info(predicted_class)

    if info["severity"] == "Healthy":

        badge = "🌱 HEALTHY"
        badge_class = "badge"

    elif info["severity"] == "High":

        badge = "⚠️ HIGH ATTENTION"
        badge_class = "badge badge-red"

    else:

        badge = "🩺 NEEDS ATTENTION"
        badge_class = "badge badge-yellow"

    st.markdown(
        f"""
        <div class="result-box">

            <div class="{badge_class}">
                {badge}
            </div>

            <div class="result-heading">
                {info["name"]}
            </div>

            <p>
                <strong>Plant:</strong> {info["plant"]}
                &nbsp;&nbsp; | &nbsp;&nbsp;
                <strong>Category:</strong> {info["type"]}
            </p>

            <div class="result-confidence">
                AI Confidence: {confidence:.2f}%
            </div>

            <div style="margin-top:15px;">
                <div class="progress-background">
                    <div
                        class="progress-fill"
                        style="width:{min(confidence,100)}%;">
                    </div>
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🔎 What does this mean?")

    st.write(
        info["description"]
    )

    left, right = st.columns(2)

    with left:

        st.markdown("### 🩺 Common Symptoms")

        for symptom in info["symptoms"]:
            st.markdown(
                f"- {symptom}"
            )

    with right:

        st.markdown("### 🛠️ Recommended Actions")

        for action in info["action"]:
            st.markdown(
                f"- {action}"
            )

    if st.session_state.show_prevention:

        st.markdown("### 🛡️ Prevention")

        prevention_cols = st.columns(
            len(info["prevention"])
            if len(info["prevention"]) <= 4
            else 4
        )

        for i, item in enumerate(
            info["prevention"]
        ):

            with prevention_cols[
                i % len(prevention_cols)
            ]:

                st.markdown(
                    f"""
                    <div class="card">
                        <strong>✓ {item}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    if top_predictions:

        st.markdown("### 📊 Other Possible Predictions")

        for name, score in top_predictions:

            st.write(
                f"**{name}** — {score:.2f}%"
            )

            st.progress(
                min(score / 100, 1.0)
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <div class="sidebar-title">
                🌿 FloraSense
            </div>

            <div class="sidebar-tagline">
                Detect. Diagnose. Protect.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section-title">Main</div>',
        unsafe_allow_html=True
    )

    navigation = [

        ("🏠 Home", "Home"),

        ("📷 AI Scanner", "AI Scanner"),

        ("🩺 Diagnosis", "Diagnosis"),

        ("🌱 My Plants", "My Plants"),

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
        '<div class="sidebar-section-title">Explore</div>',
        unsafe_allow_html=True
    )

    navigation = [

        ("🦠 Disease Library", "Disease Library"),

        ("🌿 Plant Library", "Plant Library"),

        ("📊 History", "History"),

        ("📚 Learn", "Learn"),

        ("💧 Care", "Care"),

        ("🌦️ Weather", "Weather"),

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
        '<div class="sidebar-section-title">AI & Model</div>',
        unsafe_allow_html=True
    )

    navigation = [

        ("🤖 FloraSense AI", "FloraSense AI"),

        ("🧪 How AI Works", "How AI Works"),

        ("📈 Model Performance", "Model Performance"),

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
        '<div class="sidebar-section-title">Application</div>',
        unsafe_allow_html=True
    )

    navigation = [

        ("⚙️ Settings", "Settings"),

        ("ℹ️ About", "About"),

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

            <div class="sidebar-footer-title">
                FloraSense
            </div>

            <div class="sidebar-footer-text">
                AI Plant Health Assistant<br>
                Intelligent Leaf Analysis<br>
                v1.0
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-eyebrow">
                AI-POWERED PLANT HEALTH PLATFORM
            </div>

            <h1>
                🌿 Welcome to FloraSense
            </h1>

            <p>
                Your intelligent plant health assistant for detecting
                common diseases, understanding symptoms and learning
                better plant-care practices.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    stats = st.columns(4)

    stat_data = [
        ("🦠", "15", "Disease Classes"),
        ("🎯", "91.3%", "Validation Accuracy"),
        ("🌱", "3", "Supported Plants"),
        ("📐", "224×224", "AI Image Input")
    ]

    for col, data in zip(stats, stat_data):

        with col:

            st.markdown(
                f"""
                <div class="stat-card">

                    <div class="stat-icon">
                        {data[0]}
                    </div>

                    <div class="stat-number">
                        {data[1]}
                    </div>

                    <div class="stat-label">
                        {data[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    section_title(
        "🌱 What can FloraSense do?",
        "A complete plant-health workspace built around AI-powered image analysis."
    )

    features = [

        (
            "📷",
            "AI Leaf Scanner",
            "Upload a plant leaf image and receive an AI-generated disease prediction with a confidence score."
        ),

        (
            "🩺",
            "Detailed Diagnosis",
            "Understand what the predicted condition means, its symptoms, possible actions and prevention."
        ),

        (
            "🦠",
            "Disease Library",
            "Explore the complete set of diseases and healthy classes recognized by your model."
        ),

        (
            "🌱",
            "Plant Monitoring",
            "Create a simple personal collection of plants that you want to monitor."
        ),

        (
            "📚",
            "Plant Education",
            "Learn about plant care, disease prevention, watering, airflow and common problems."
        ),

        (
            "🤖",
            "Plant AI Assistant",
            "Ask common plant-health questions using FloraSense's built-in knowledge assistant."
        )

    ]

    feature_cols = st.columns(3)

    for index, feature in enumerate(features):

        with feature_cols[index % 3]:

            st.markdown(
                f"""
                <div class="feature-card">

                    <div class="feature-icon">
                        {feature[0]}
                    </div>

                    <div class="feature-title">
                        {feature[1]}
                    </div>

                    <div class="feature-text">
                        {feature[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    section_title(
        "🌿 Supported Plants",
        "The trained model currently recognizes three major plant groups."
    )

    plants = [

        (
            "🍅",
            "Tomato",
            "Detect common tomato conditions including early blight, late blight, bacterial spot, leaf mold, septoria, target spot, viral diseases and spider mites."
        ),

        (
            "🥔",
            "Potato",
            "Analyze potato leaves for early blight, late blight and healthy foliage."
        ),

        (
            "🌶️",
            "Pepper",
            "Recognize pepper bacterial spot and healthy pepper leaves."
        )

    ]

    plant_cols = st.columns(3)

    for col, plant in zip(
        plant_cols,
        plants
    ):

        with col:

            st.markdown(
                f"""
                <div class="plant-card">

                    <div class="plant-icon">
                        {plant[0]}
                    </div>

                    <div class="plant-name">
                        {plant[1]}
                    </div>

                    <div class="plant-text">
                        {plant[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    section_title(
        "⚡ How FloraSense Works",
        "From an image to an understandable plant-health result."
    )

    pipeline = [

        (
            "1",
            "Upload",
            "Choose a clear plant leaf image."
        ),

        (
            "2",
            "Process",
            "The image is resized to 224 × 224."
        ),

        (
            "3",
            "Analyze",
            "The TensorFlow model analyzes visual patterns."
        ),

        (
            "4",
            "Predict",
            "The highest-probability class is selected."
        ),

        (
            "5",
            "Understand",
            "FloraSense presents the result and guidance."
        )

    ]

    pipeline_cols = st.columns(5)

    for col, step in zip(
        pipeline_cols,
        pipeline
    ):

        with col:

            st.markdown(
                f"""
                <div class="pipeline">

                    <div class="pipeline-number">
                        {step[0]}
                    </div>

                    <div class="pipeline-title">
                        {step[1]}
                    </div>

                    <div class="pipeline-text">
                        {step[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        """
        <div class="app-footer">
            🌿 FloraSense · AI Plant Health Assistant · v1.0
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# AI SCANNER
# ============================================================

elif st.session_state.page == "AI Scanner":

    page_header(
        "📷 AI Plant Scanner",
        "Upload a clear leaf image and let FloraSense analyze it."
    )

    if model is None:

        st.error(
            "The AI model could not be loaded."
        )

        st.code(
            model_error or "Unknown model loading error"
        )

    else:

        left, right = st.columns(
            [1.05, 0.95]
        )

        with left:

            st.markdown(
                """
                <div class="card">

                    <h3>📸 Upload Leaf Image</h3>

                    <p>
                        For the best prediction, upload a clear image
                        where the leaf is visible and well illuminated.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            uploaded_file = st.file_uploader(
                "Choose JPG, JPEG or PNG",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ],
                key="main_scanner"
            )

            if uploaded_file:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.image(
                    image,
                    caption="Uploaded leaf",
                    use_container_width=True
                )

                analyze = st.button(
                    "🔍 Analyze Leaf",
                    use_container_width=True
                )

                if analyze:

                    with st.spinner(
                        "FloraSense is analyzing the leaf..."
                    ):

                        try:

                            (
                                analyzed_image,
                                predicted_class,
                                confidence,
                                top_predictions
                            ) = predict_image(
                                uploaded_file
                            )

                            st.session_state.last_prediction = predicted_class
                            st.session_state.last_confidence = confidence

                            save_history(
                                predicted_class,
                                confidence
                            )

                            st.success(
                                "Analysis completed successfully."
                            )

                            disease_result(
                                predicted_class,
                                confidence,
                                top_predictions
                            )

                        except Exception as e:

                            st.error(
                                f"Prediction failed: {e}"
                            )

        with right:

            st.markdown(
                """
                <div class="card">

                    <h3>💡 Tips for Better Results</h3>

                    <p>
                        Better images can help the model recognize
                        visual patterns more reliably.
                    </p>

                    <p>✓ Use a clear image</p>
                    <p>✓ Keep the leaf visible</p>
                    <p>✓ Avoid extremely dark images</p>
                    <p>✓ Avoid excessive blur</p>
                    <p>✓ Try to photograph the affected area</p>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="card">

                    <h3>🧠 AI Model</h3>

                    <p>
                        <strong>Architecture:</strong>
                        TensorFlow / Keras
                    </p>

                    <p>
                        <strong>Input:</strong>
                        224 × 224 RGB image
                    </p>

                    <p>
                        <strong>Classes:</strong>
                        15
                    </p>

                    <p>
                        <strong>Validation accuracy:</strong>
                        approximately 91.3%
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# DIAGNOSIS
# ============================================================

elif st.session_state.page == "Diagnosis":

    page_header(
        "🩺 Plant Health Diagnosis",
        "Review the latest AI-generated plant health assessment."
    )

    if st.session_state.last_prediction is None:

        st.info(
            "No diagnosis is available yet."
        )

        st.markdown(
            """
            <div class="card">

                <h3>📷 Start with the AI Scanner</h3>

                <p>
                    Go to AI Scanner from the left navigation,
                    upload a plant leaf image and run the analysis.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        disease_result(
            st.session_state.last_prediction,
            st.session_state.last_confidence
        )

        st.markdown(
            """
            <div class="card">

                <h3>⚠️ Important</h3>

                <p>
                    An AI prediction is an informational assessment.
                    Plant symptoms can look similar across different
                    diseases, pests and environmental conditions.
                    Consider professional agricultural advice for
                    important crop decisions.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# MY PLANTS
# ============================================================

elif st.session_state.page == "My Plants":

    page_header(
        "🌱 My Plants",
        "Create a simple personal plant-monitoring dashboard."
    )

    left, right = st.columns(
        [1, 1.5]
    )

    with left:

        st.markdown(
            """
            <div class="card">

                <h3>➕ Add a Plant</h3>

                <p>
                    Add plants that you want to keep track of.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        plant_name = st.text_input(
            "Plant name",
            placeholder="Example: Balcony Tomato"
        )

        plant_type = st.selectbox(
            "Plant type",
            [
                "Tomato",
                "Potato",
                "Pepper"
            ]
        )

        if st.button(
            "🌱 Add Plant",
            use_container_width=True
        ):

            if plant_name.strip():

                st.session_state.plants.append(
                    {
                        "name": plant_name.strip(),
                        "type": plant_type
                    }
                )

                st.success(
                    "Plant added to your collection."
                )

            else:

                st.warning(
                    "Please enter a plant name."
                )

    with right:

        st.markdown(
            "### 🌿 Your Plant Collection"
        )

        if not st.session_state.plants:

            st.info(
                "Your collection is empty. Add your first plant."
            )

        else:

            for index, plant in enumerate(
                st.session_state.plants
            ):

                icon = {
                    "Tomato": "🍅",
                    "Potato": "🥔",
                    "Pepper": "🌶️"
                }.get(
                    plant["type"],
                    "🌱"
                )

                st.markdown(
                    f"""
                    <div class="plant-card">

                        <div class="plant-icon">
                            {icon}
                        </div>

                        <div class="plant-name">
                            {plant["name"]}
                        </div>

                        <div class="plant-text">
                            Type: {plant["type"]}
                        </div>

                        <div class="badge">
                            MONITORING
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# DISEASE LIBRARY
# ============================================================

elif st.session_state.page == "Disease Library":

    page_header(
        "🦠 Disease Library",
        "Explore the 15 classes recognized by the FloraSense model."
    )

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:

        search = st.text_input(
            "🔎 Search",
            placeholder="Search disease or plant..."
        ).lower()

    with filter_col2:

        plant_filter = st.selectbox(
            "🌱 Filter by plant",
            [
                "All",
                "Tomato",
                "Potato",
                "Pepper"
            ]
        )

    results = []

    for key, info in disease_info.items():

        searchable = (
            info["name"]
            + " "
            + info["plant"]
            + " "
            + info["type"]
        ).lower()

        matches_search = (
            not search
            or search in searchable
        )

        matches_plant = (
            plant_filter == "All"
            or info["plant"] == plant_filter
        )

        if matches_search and matches_plant:

            results.append(
                (key, info)
            )

    st.markdown(
        f"### Showing {len(results)} classes"
    )

    for key, info in results:

        severity_class = (
            "badge"
            if info["severity"] == "Healthy"
            else "badge badge-yellow"
        )

        if info["severity"] == "High":
            severity_class = "badge badge-red"

        st.markdown(
            f"""
            <div class="card">

                <div class="{severity_class}">
                    {info["severity"].upper()}
                </div>

                <h3>
                    {info["name"]}
                </h3>

                <p>
                    <strong>Plant:</strong>
                    {info["plant"]}
                </p>

                <p>
                    <strong>Type:</strong>
                    {info["type"]}
                </p>

                <p>
                    {info["description"]}
                </p>

                <hr>

                <p>
                    <strong>Symptoms:</strong>
                    {" • ".join(info["symptoms"])}
                </p>

                <p>
                    <strong>Recommended action:</strong>
                    {" • ".join(info["action"])}
                </p>

                <p>
                    <strong>Prevention:</strong>
                    {" • ".join(info["prevention"])}
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# PLANT LIBRARY
# ============================================================

elif st.session_state.page == "Plant Library":

    page_header(
        "🌿 Plant Library",
        "Explore the plants currently supported by the FloraSense model."
    )

    plant_details = [

        {
            "icon": "🍅",
            "name": "Tomato",
            "description": "Tomato is the most extensively represented plant group in the current model.",
            "conditions": "10 recognized classes",
            "care": [
                "Provide good sunlight",
                "Maintain appropriate soil moisture",
                "Allow good airflow",
                "Inspect leaves regularly"
            ]
        },

        {
            "icon": "🥔",
            "name": "Potato",
            "description": "Potato leaves can show symptoms associated with early blight and late blight.",
            "conditions": "3 recognized classes",
            "care": [
                "Use healthy planting material",
                "Avoid prolonged leaf wetness",
                "Maintain spacing",
                "Monitor for blight"
            ]
        },

        {
            "icon": "🌶️",
            "name": "Pepper",
            "description": "Pepper plants are currently represented by bacterial spot and healthy classes.",
            "conditions": "2 recognized classes",
            "care": [
                "Provide sunlight",
                "Use well-draining soil",
                "Avoid excessive watering",
                "Inspect leaves for spots"
            ]
        }

    ]

    for plant in plant_details:

        st.markdown(
            f"""
            <div class="plant-card">

                <div class="plant-icon">
                    {plant["icon"]}
                </div>

                <div class="plant-name">
                    {plant["name"]}
                </div>

                <div class="plant-text">
                    {plant["description"]}
                </div>

                <p>
                    <strong>Model coverage:</strong>
                    {plant["conditions"]}
                </p>

                <p>
                    <strong>Care priorities:</strong>
                    {" • ".join(plant["care"])}
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# HISTORY
# ============================================================

elif st.session_state.page == "History":

    page_header(
        "📊 Detection History",
        "Review predictions made during your current app session."
    )

    if not st.session_state.history:

        st.info(
            "No predictions have been recorded yet."
        )

        st.markdown(
            """
            <div class="card">

                <h3>📷 Your history starts here</h3>

                <p>
                    Use AI Scanner to analyze a leaf. Each completed
                    analysis will appear on this page.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        total = len(
            st.session_state.history
        )

        average = sum(
            item["confidence"]
            for item in st.session_state.history
        ) / total

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-icon">🔬</div>
                    <div class="stat-number">{total}</div>
                    <div class="stat-label">Total Analyses</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-icon">🎯</div>
                    <div class="stat-number">{average:.1f}%</div>
                    <div class="stat-label">Average Confidence</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            healthy_count = sum(
                1
                for item in st.session_state.history
                if "healthy" in item["disease"].lower()
            )

            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-icon">🌱</div>
                    <div class="stat-number">{healthy_count}</div>
                    <div class="stat-label">Healthy Results</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### Recent Analyses")

        for index, item in enumerate(
            reversed(st.session_state.history)
        ):

            st.markdown(
                f"""
                <div class="card">

                    <span class="badge">
                        ANALYSIS {len(st.session_state.history)-index}
                    </span>

                    <h3>
                        {item["disease"]}
                    </h3>

                    <p>
                        AI confidence:
                        <strong>
                            {item["confidence"]:.2f}%
                        </strong>
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# LEARN
# ============================================================

elif st.session_state.page == "Learn":

    page_header(
        "📚 Learn About Plant Health",
        "Build a better understanding of plant symptoms and prevention."
    )

    lessons = [

        (
            "🔍",
            "How to Inspect a Leaf",
            "Look at both sides of the leaf. Check for spots, discoloration, "
            "curling, holes, webbing and unusual growth. Compare affected "
            "leaves with healthy leaves on the same plant."
        ),

        (
            "💧",
            "Understanding Watering",
            "Watering requirements vary by plant and environment. Check "
            "soil moisture instead of relying only on a fixed schedule. "
            "Good drainage is essential."
        ),

        (
            "🌬️",
            "Why Airflow Matters",
            "Dense foliage and prolonged moisture can create conditions "
            "that favor some diseases. Appropriate spacing and airflow "
            "can help keep foliage drier."
        ),

        (
            "🦠",
            "Disease Prevention",
            "Regular inspection, clean planting material, removing infected "
            "debris and avoiding unnecessary leaf wetness can help reduce "
            "disease pressure."
        ),

        (
            "🐛",
            "Recognizing Pest Damage",
            "Speckling, holes, curling, sticky residue and webbing can be "
            "signs of pest activity. Inspect leaf undersides carefully."
        ),

        (
            "🌱",
            "Healthy Plant Basics",
            "Healthy plants generally show strong growth, appropriate "
            "color and good root-zone conditions. Regular observation "
            "helps detect problems earlier."
        )

    ]

    lesson_cols = st.columns(2)

    for index, lesson in enumerate(lessons):

        with lesson_cols[index % 2]:

            st.markdown(
                f"""
                <div class="feature-card">

                    <div class="feature-icon">
                        {lesson[0]}
                    </div>

                    <div class="feature-title">
                        {lesson[1]}
                    </div>

                    <div class="feature-text">
                        {lesson[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# CARE
# ============================================================

elif st.session_state.page == "Care":

    page_header(
        "💧 Plant Care Center",
        "Practical guidance for maintaining healthier plants."
    )

    care_sections = [

        (
            "💧",
            "Watering",
            "Check soil moisture before watering. Avoid keeping roots "
            "constantly waterlogged. Container plants should have adequate drainage."
        ),

        (
            "☀️",
            "Light",
            "Provide suitable sunlight for the specific plant. If plants "
            "show stress, consider whether their current growing location "
            "matches their needs."
        ),

        (
            "🌬️",
            "Airflow",
            "Avoid overcrowding plants. Good airflow can help reduce "
            "prolonged moisture around foliage."
        ),

        (
            "🧹",
            "Plant Hygiene",
            "Remove severely affected plant material where appropriate "
            "and clean tools after working with diseased plants."
        ),

        (
            "🌱",
            "Soil & Drainage",
            "Well-draining growing media can help prevent excessive moisture "
            "around roots."
        ),

        (
            "🔎",
            "Regular Monitoring",
            "Inspect plants frequently. Early observation can help identify "
            "changes before a problem becomes severe."
        )

    ]

    cols = st.columns(3)

    for index, section in enumerate(
        care_sections
    ):

        with cols[index % 3]:

            st.markdown(
                f"""
                <div class="feature-card">

                    <div class="feature-icon">
                        {section[0]}
                    </div>

                    <div class="feature-title">
                        {section[1]}
                    </div>

                    <div class="feature-text">
                        {section[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# WEATHER
# ============================================================

elif st.session_state.page == "Weather":

    page_header(
        "🌦️ Weather & Plant Care",
        "Use general weather conditions to think about plant-care adjustments."
    )

    condition = st.selectbox(
        "Select the current condition",
        [
            "☀️ Hot and Sunny",
            "🌧️ Rainy",
            "☁️ Cloudy",
            "🌬️ Windy",
            "🌤️ Mild"
        ]
    )

    recommendations = {

        "☀️ Hot and Sunny": (
            "Monitor soil moisture more frequently. Protect plants from "
            "unusual heat stress and avoid unnecessary leaf wetting during "
            "very hot conditions."
        ),

        "🌧️ Rainy": (
            "Pay attention to drainage and prolonged leaf wetness. Wet "
            "conditions can increase disease pressure for some pathogens."
        ),

        "☁️ Cloudy": (
            "Monitor humidity and airflow. Avoid unnecessary watering if "
            "the growing medium is already moist."
        ),

        "🌬️ Windy": (
            "Check plants for physical damage and make sure tall plants "
            "or containers are properly supported."
        ),

        "🌤️ Mild": (
            "Continue normal monitoring, watering and plant-care routines."
        )

    }

    st.markdown(
        f"""
        <div class="hero">

            <div class="hero-eyebrow">
                PLANT CARE ADVISORY
            </div>

            <h1>
                {condition}
            </h1>

            <p>
                {recommendations[condition]}
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "This is general plant-care guidance, not a live weather forecast."
    )


# ============================================================
# FLORASENSE AI
# ============================================================

elif st.session_state.page == "FloraSense AI":

    page_header(
        "🤖 FloraSense AI",
        "Ask questions about common tomato, potato and pepper plant problems."
    )

    st.markdown(
        """
        <div class="chat-container">

            <div class="chat-ai">

                <strong>🌿 FloraSense AI</strong><br><br>

                Hello! I can help you understand common plant-health
                topics such as yellow leaves, watering, blight, leaf spots,
                fungal problems, pests and healthy plant care.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 💡 Suggested Questions")

    suggestions = [
        "Why are my tomato leaves yellow?",
        "What is blight?",
        "How often should I water plants?",
        "How can I prevent fungal diseases?",
        "What are common potato diseases?",
        "How do I identify plant pests?"
    ]

    suggestion_cols = st.columns(3)

    for index, question in enumerate(
        suggestions
    ):

        with suggestion_cols[index % 3]:

            st.markdown(
                f"""
                <div class="card">
                    <strong>💬 {question}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )

    question = st.text_input(
        "Ask FloraSense",
        placeholder="Type your plant-health question..."
    )

    if st.button(
        "🤖 Ask FloraSense",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please type a question first."
            )

        else:

            q = question.lower()

            answer = None

            for keyword, response in qa_database.items():

                if keyword in q:

                    answer = response
                    break

            if answer is None:

                answer = (
                    "I currently focus on common questions about "
                    "tomatoes, potatoes, peppers, yellow leaves, "
                    "watering, blight, leaf spots, fungi, pests "
                    "and healthy plant care."
                )

            st.markdown(
                f"""
                <div class="chat-container">

                    <div class="chat-user">
                        <strong>You</strong><br>
                        {question}
                    </div>

                    <div class="chat-ai">
                        <strong>🌿 FloraSense AI</strong><br><br>
                        {answer}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# HOW AI WORKS
# ============================================================

elif st.session_state.page == "How AI Works":

    page_header(
        "🧪 How FloraSense AI Works",
        "Understand the complete image-classification pipeline."
    )

    steps = [

        (
            "1",
            "📷 Upload Image",
            "The user uploads a JPG, JPEG or PNG image containing a plant leaf."
        ),

        (
            "2",
            "📐 Resize",
            "The image is converted to RGB and resized to 224 × 224 pixels."
        ),

        (
            "3",
            "🧠 Neural Network",
            "The TensorFlow/Keras model analyzes visual features from the image."
        ),

        (
            "4",
            "📊 Probability Scores",
            "The model generates probability scores for the 15 recognized classes."
        ),

        (
            "5",
            "🎯 Select Prediction",
            "The class with the highest probability becomes the predicted class."
        ),

        (
            "6",
            "🩺 Explain",
            "FloraSense presents the prediction, confidence, symptoms and guidance."
        )

    ]

    cols = st.columns(3)

    for index, step in enumerate(
        steps
    ):

        with cols[index % 3]:

            st.markdown(
                f"""
                <div class="pipeline">

                    <div class="pipeline-number">
                        {step[0]}
                    </div>

                    <div class="pipeline-title">
                        {step[1]}
                    </div>

                    <div class="pipeline-text">
                        {step[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    section_title(
        "🔬 Technical Configuration"
    )

    tech1, tech2, tech3 = st.columns(3)

    technical = [

        (
            "Framework",
            "TensorFlow / Keras"
        ),

        (
            "Image Input",
            "224 × 224 RGB"
        ),

        (
            "Classification",
            "15 classes"
        )

    ]

    for col, item in zip(
        [tech1, tech2, tech3],
        technical
    ):

        with col:

            st.markdown(
                f"""
                <div class="stat-card">

                    <div class="stat-label">
                        {item[0]}
                    </div>

                    <div class="stat-number"
                         style="font-size:22px;">
                        {item[1]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif st.session_state.page == "Model Performance":

    page_header(
        "📈 Model Performance",
        "Key information about the trained FloraSense classification model."
    )

    c1, c2, c3, c4 = st.columns(4)

    performance = [

        ("🎯", "91.3%", "Validation Accuracy"),

        ("🦠", "15", "Classes"),

        ("🌱", "3", "Plant Groups"),

        ("📐", "224×224", "Input Resolution")

    ]

    for col, data in zip(
        [c1, c2, c3, c4],
        performance
    ):

        with col:

            st.markdown(
                f"""
                <div class="stat-card">

                    <div class="stat-icon">
                        {data[0]}
                    </div>

                    <div class="stat-number">
                        {data[1]}
                    </div>

                    <div class="stat-label">
                        {data[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    section_title(
        "🧠 Model Overview"
    )

    st.markdown(
        """
        <div class="card">

            <h3>TensorFlow / Keras Image Classifier</h3>

            <p>
                FloraSense uses a trained image-classification model
                designed to recognize visual patterns associated with
                plant diseases and healthy leaves.
            </p>

            <p>
                The model accepts RGB leaf images resized to
                <strong>224 × 224 pixels</strong> and produces
                probability scores across <strong>15 classes</strong>.
            </p>

            <p>
                The reported validation accuracy from training was
                approximately <strong>91.3%</strong>.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    section_title(
        "🦠 Recognized Classes"
    )

    class_columns = st.columns(3)

    for index, class_name in enumerate(
        class_names
    ):

        with class_columns[index % 3]:

            st.markdown(
                f"""
                <div class="card">

                    <strong>
                        {index + 1}. {friendly_name(class_name)}
                    </strong>

                </div>
                """,
                unsafe_allow_html=True
            )

    section_title(
        "⚠️ Model Limitations"
    )

    st.markdown(
        """
        <div class="card">

            <p>
                • Performance can vary with image quality.
            </p>

            <p>
                • Real-world leaves may look different from training images.
            </p>

            <p>
                • Similar symptoms can occur across different conditions.
            </p>

            <p>
                • Predictions should be treated as an informational
                machine-learning assessment rather than a guaranteed diagnosis.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SETTINGS
# ============================================================

elif st.session_state.page == "Settings":

    page_header(
        "⚙️ Settings",
        "Customize the information displayed by FloraSense."
    )

    st.markdown(
        """
        <div class="card">

            <h3>🎛️ Display Preferences</h3>

            <p>
                These preferences apply to your current browser session.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.session_state.show_confidence = st.checkbox(
        "Show AI confidence percentage",
        value=st.session_state.show_confidence
    )

    st.session_state.show_prevention = st.checkbox(
        "Show prevention guidance",
        value=st.session_state.show_prevention
    )

    language = st.selectbox(
        "🌐 Language",
        ["English"]
    )

    st.success(
        "Settings updated for this session."
    )

    st.markdown(
        """
        <div class="card">

            <h3>🔐 Privacy</h3>

            <p>
                Images uploaded to the scanner are processed by the
                application for prediction. This interface does not
                provide a permanent cloud-based personal image gallery.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ABOUT
# ============================================================

elif st.session_state.page == "About":

    page_header(
        "ℹ️ About FloraSense",
        "An AI-powered plant health project built with machine learning."
    )

    st.markdown(
        """
        <div class="hero">

            <div class="hero-eyebrow">
                FLORASENSE · VERSION 1.0
            </div>

            <h1>
                🌿 Detect. Diagnose. Protect.
            </h1>

            <p>
                FloraSense combines image classification and plant-health
                information to create an accessible plant disease detection
                platform.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            <div class="card">

                <h3>🎯 Project Objective</h3>

                <p>
                    The objective of FloraSense is to help users identify
                    common plant-health conditions from leaf images using
                    a trained machine-learning model.
                </p>

                <p>
                    The application also provides educational information
                    about symptoms, prevention and general plant care.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="card">

                <h3>🛠️ Technologies</h3>

                <p>
                    <strong>Frontend:</strong> Streamlit
                </p>

                <p>
                    <strong>Machine Learning:</strong> TensorFlow / Keras
                </p>

                <p>
                    <strong>Image Processing:</strong> PIL / NumPy
                </p>

                <p>
                    <strong>Model Hosting:</strong> Hugging Face
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    section_title(
        "🌱 FloraSense at a Glance"
    )

    overview_cols = st.columns(4)

    overview = [

        ("15", "AI Classes"),
        ("3", "Plant Types"),
        ("91.3%", "Validation Accuracy"),
        ("224×224", "Input Image")

    ]

    for col, item in zip(
        overview_cols,
        overview
    ):

        with col:

            st.markdown(
                f"""
                <div class="stat-card">

                    <div class="stat-number">
                        {item[0]}
                    </div>

                    <div class="stat-label">
                        {item[1]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        """
        <div class="app-footer">
            🌿 FloraSense · AI Plant Health Assistant · v1.0
        </div>
        """,
        unsafe_allow_html=True
    )
