"""
Configuration and constants for the AI Cultural Tourism Engine
"""
import streamlit as st
import uuid

# =========================
# STREAMLIT PAGE CONFIG
# =========================
def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="AI Cultural Tourism Engine",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# =========================
# SESSION STATE INITIALIZATION
# =========================
def initialize_session_state():
    """Initialize all session state variables"""
    session_defaults = {
        'ranked_results': None,
        'user_input': None,
        'session_id': str(uuid.uuid4()),
        'firebase_doc_id': None,
        'show_itinerary_form': False,
        'personalization_complete': False,
        'pdf_buffer': None,
        'current_itinerary': None,
        'current_city': None,
        'current_user_input': None,
    }
    
    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# =========================
# CONSTANTS
# =========================
INTEREST_OPTIONS = [
    "Culture & History",
    "Adventure & Nature",
    "Beach & Relaxation",
    "Food & Culinary",
    "Art & Museums",
    "Shopping & Urban",
    "Wellness & Spa",
    "Photography"
]

BUDGET_LEVELS = ["Budget", "Mid-Range", "Luxury"]
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
WEATHER_PREFS = ["Cold", "Pleasant", "Warm"]
AGE_GROUPS = ["18-25", "26-35", "36-50", "50+"]

# =========================
# API KEYS & SECRETS
# =========================
def get_gemini_api_key():
    """Get Gemini API key from secrets"""
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return None

def get_firebase_credentials():
    """Get Firebase credentials from secrets"""
    if "FIREBASE_CREDENTIALS" in st.secrets:
        return dict(st.secrets["FIREBASE_CREDENTIALS"])
    return None
