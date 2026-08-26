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

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #f3fff7 0%,
        #eefaf3 50%,
        #f8fffb 100%
    );
}

/* Hide Streamlit default menu/footer */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #043b2a 0%,
        #076b43 55%,
        #0b8955 100%
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-title {
    font-size: 28px;
    font-weight: 800;
    text-align: center;
    padding-top: 12px;
}

.sidebar-subtitle {
    color: #c9f7df !important;
    text-align: center;
    font-size: 13px;
    margin-bottom: 20px;
}

/* ================= HEADINGS ================= */

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

p {
    color: #40564b;
}

/* ================= HERO ================= */

.hero {
    background: linear-gradient(
        135deg,
        #063b2b,
        #087443,
        #21a96d
    );
    padding: 50px;
    border-radius: 30px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 15px 45px rgba(0, 80, 50, 0.18);
}

.hero-small {
    color: #bff5d8 !important;
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
    max-width: 800px;
}

/* ================= CARDS ================= */

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

/* ================= STATS ================= */

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

/* ================= PLANT CARDS ================= */

.plant-card {
    background: white;
