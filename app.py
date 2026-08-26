import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL DESIGN
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 5% 5%, rgba(52, 211, 153, 0.14), transparent 22%),
        radial-gradient(circle at 95% 10%, rgba(45, 212, 191, 0.12), transparent 22%),
        linear-gradient(135deg, #f7fbf8 0%, #eef8f2 100%);
}

/* Hide Streamlit branding but keep sidebar controls */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #063b2b 0%,
        #07553d 45%,
        #08734b 100%
    );
    min-width: 280px;
    max-width: 280px;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-brand {
    text-align: center;
    padding: 12px 5px 25px 5px;
}

.sidebar-logo {
    font-size: 55px;
}

.sidebar-title {
    font-size: 25px;
    font-weight: 800;
    margin-top: 5px;
}

.sidebar-subtitle {
    font-size: 12px;
    opacity: 0.75;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.18);
}

section[data-testid="stSidebar"] .stRadio label {
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* MAIN TITLES */

.page-title {
    font-size: 42px;
    font-weight: 800;
    color: #073b2b;
    margin-bottom: 5px;
}

.page-subtitle {
    color: #63766d;
    font-size: 16px;
    margin-bottom: 30px;
}

/* HERO */

.hero {
    padding: 55px 55px;
    border-radius: 32px;
    background:
        linear-gradient(
            135deg,
            #063b2b 0%,
            #08734b 50%,
            #13a36b 100%
        );
    color: white;
    box-shadow: 0 20px 55px rgba(6, 59, 43, 0.22);
    margin-bottom: 30px;
}

.hero-label {
    color: #a9f5d0;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero-title {
    font-size: 52px;
    line-height: 1.05;
    font-weight: 800;
    margin: 15px 0;
}

.hero-text {
    color: #e1fff0;
    font-size: 18px;
    line-height: 1.6;
    max-width: 780px;
}

/* CARDS */

.card {
    background: rgba(255,255,255,0.96);
    border: 1px solid #dcebe3;
    border-radius: 24px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 10px 35px rgba(15, 60, 40, 0.07);
    color: #17352a;
}

.card h2,
.card h3 {
    color: #073b2b;
}

.card p {
    color: #52665d;
    line-height: 1.65;
}

/* STAT CARDS */

.stat-card {
    background: white;
    border-radius: 22px;
    padding: 25px 15px;
    text-align: center;
    border: 1px solid #dcebe3;
    box-shadow: 0 8px 25px rgba(15, 60, 40, 0.06);
    min-height: 125px;
}

.stat-icon {
    font-size: 28px;
}

.stat-number {
    font-size: 30px;
    font-weight: 800;
    color: #087443;
}

.stat-label {
    color: #718078;
    font-size: 13px;
    font-weight: 600;
}

/* FEATURE CARDS */

.feature {
    background: white;
    border-radius: 25px;
    padding: 28px;
    border: 1px solid #dcebe3;
    min-height: 190px;
    box-shadow: 0 10px 30px rgba(15,60,40,0.06);
}

.feature-icon {
    font-size: 38px;
}

.feature-title {
    font-size: 20px;
    font-weight: 800;
    color: #073b2b;
    margin: 12px 0 7px 0;
}

.feature-text {
    color: #61746b;
    line-height: 1.55;
}

/* PLANT CARDS */

.plant-card {
    background: white;
    border-radius: 25px;
    padding: 28px;
    text-align: center;
    border: 1px solid #dcebe3;
    box-shadow: 0 8px 25px rgba(15,60,40,0.06);
}

.plant-emoji {
    font-size: 58px;
}

/* RESULT */

.result-card {
    background: linear-gradient(
        135deg,
        #ffffff,
        #effbf4
    );
    border: 2px solid #9bdab7;
    border-radius: 28px;
    padding: 35px;
    box-shadow: 0 15px 40px rgba(15,70,45,0.08);
}

.result-label {
    color: #718078;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.result-name {
    color: #073b2b;
    font-size: 34px;
    font-weight: 800;
    margin: 8px 0 18px 0;
}

.confidence {
    color: #087443;
    font-size: 42px;
    font-weight: 800;
}

/* HEALTH SCORE */

.health-card {
    background: white;
    border-radius: 25px;
    padding: 28px;
    border: 1px solid #dcebe3;
    box-shadow: 0 10px 30px rgba(15,60,40,0.06);
}

.health-number {
    font-size: 48px;
    font-weight: 800;
    color: #087443;
}

/* TIPS */

.tip-card {
    background: linear-gradient(135deg, #ecfdf4, #f
