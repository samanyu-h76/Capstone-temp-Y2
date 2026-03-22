import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime
import os
import hashlib
import firebase_admin
from firebase_admin import credentials, firestore
import json
import uuid
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from io import BytesIO
import requests
from PIL import Image as PILImage
from moviepy import (
    ImageClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
    ColorClip
)
import tempfile
import re
import random

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Cultural Tourism Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# FEATURE 2: PROPER STREAMLIT SESSION STATE INITIALIZATION
# THIS ENSURES DATA PERSISTS ACROSS PAGE RELOADS
# =========================
def initialize_session_state():
    """Initialize all session state variables properly - this runs on every page load"""
    
    # Authentication variables
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
    
    # Basic session variables
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Firebase variables
    if 'firebase_doc_id' not in st.session_state:
        st.session_state.firebase_doc_id = None
    
    # Personalization variables
    if 'personalization_complete' not in st.session_state:
        st.session_state.personalization_complete = False
    
    # FEATURE 2: PERSONALIZATION CACHE - PERSISTENT DATA
    if 'cached_user_input' not in st.session_state:
        st.session_state.cached_user_input = {
            'age': 30,
            'interest': 'Culture',
            'duration': 5,
            'weather': 'Pleasant',
            'season': 'Summer',
            'budget': 'Mid-range',
            'continent': 'All Continents',
            'country': 'All Countries'
        }
    
    # Itinerary preferences (moved from personalization)
    if 'itinerary_preferences' not in st.session_state:
        st.session_state.itinerary_preferences = {
            'dietary': 'Omnivore',
            'accommodation': 'Hotel',
            'accessibility': 'No specific needs'
        }
    
    # Initialize tracking for preference changes
    if 'last_user_preferences' not in st.session_state:
        st.session_state.last_user_preferences = None
    
    # FEATURE 2: CACHE RECOMMENDATIONS - PERSISTENT DATA
    if 'cached_ranked_results' not in st.session_state:
        st.session_state.cached_ranked_results = None
    
    # FEATURE 2: CACHE ITINERARIES - PERSISTENT DATA
    if 'cached_itineraries' not in st.session_state:
        st.session_state.cached_itineraries = {}
    
    # Current page variables
    if 'ranked_results' not in st.session_state:
        st.session_state.ranked_results = None
    
    if 'user_input' not in st.session_state:
        st.session_state.user_input = None
    
    if 'show_itinerary_form' not in st.session_state:
        st.session_state.show_itinerary_form = False
    
    if 'pdf_buffer' not in st.session_state:
        st.session_state.pdf_buffer = None
    
    if 'current_itinerary' not in st.session_state:
        st.session_state.current_itinerary = None
    
    if 'current_city' not in st.session_state:
        st.session_state.current_city = None
    
    if 'current_user_input' not in st.session_state:
        st.session_state.current_user_input = None
    
    if 'video_buffer' not in st.session_state:
        st.session_state.video_buffer = None
    
    if 'video_generated' not in st.session_state:
        st.session_state.video_generated = False
    
    # Chatbot variables
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'chat_language' not in st.session_state:
        st.session_state.chat_language = 'English'
    
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []

# Initialize session state on every page load
initialize_session_state()

# -------------------------
# Firebase setup
# -------------------------
FIREBASE_AVAILABLE = False

def initialize_firebase():
    """Initialize Firebase with proper error handling"""
    global FIREBASE_AVAILABLE
    
    try:
        if firebase_admin._apps:
            FIREBASE_AVAILABLE = True
            return firestore.client()
        
        if "FIREBASE_CREDENTIALS" not in st.secrets:
            return None
        
        firebase_creds = dict(st.secrets["FIREBASE_CREDENTIALS"])
        
        if "private_key" in firebase_creds:
            firebase_creds["private_key"] = str(firebase_creds["private_key"])
        
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
        
        FIREBASE_AVAILABLE = True
        return firestore.client()
        
    except Exception as e:
        return None

db = initialize_firebase()

# -------------------------
# Gemini setup
# -------------------------
GEMINI_AVAILABLE = False
gemini_error_message = ""

def initialize_gemini():
    """Initialize Gemini with proper error handling"""
    global GEMINI_AVAILABLE, gemini_error_message
    
    try:
        if "GEMINI_API_KEY" not in st.secrets:
            gemini_error_message = "GEMINI_API_KEY not found in secrets"
            return False
        
        api_key = st.secrets["GEMINI_API_KEY"]
        
        if not api_key or len(api_key) < 10:
            gemini_error_message = "Invalid API key format"
            return False
        
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content("Say 'OK' if you can read this.")
        
        if response and response.text:
            GEMINI_AVAILABLE = True
            return True
        else:
            gemini_error_message = "Gemini responded but with empty text"
            return False
            
    except Exception as e:
        gemini_error_message = f"Gemini initialization error: {str(e)}"
        return False

initialize_gemini()

# -------------------------
# FIREBASE AUTHENTICATION SETUP (REST API)
# -------------------------
FIREBASE_AUTH_AVAILABLE = False
FIREBASE_API_KEY = None
FIREBASE_PROJECT_ID = None

def initialize_firebase_auth():
    global FIREBASE_AUTH_AVAILABLE, FIREBASE_API_KEY, FIREBASE_PROJECT_ID

    FIREBASE_API_KEY = None
    FIREBASE_PROJECT_ID = None
    FIREBASE_AUTH_AVAILABLE = False

    try:
        # ✅ SAFE retrieval
        FIREBASE_API_KEY = st.secrets.get("FIREBASE_API_KEY", None)
        FIREBASE_PROJECT_ID = st.secrets.get("FIREBASE_PROJECT_ID", None)

        st.write("DEBUG API KEY:", FIREBASE_API_KEY)
        st.write("DEBUG PROJECT ID:", FIREBASE_PROJECT_ID)

        # ✅ CHECK ACTUAL VALUES (not key existence)
        if FIREBASE_API_KEY and len(FIREBASE_API_KEY) > 10:
            if FIREBASE_PROJECT_ID and len(FIREBASE_PROJECT_ID) > 3:
                FIREBASE_AUTH_AVAILABLE = True
                return True

        st.error("Firebase keys exist but validation failed")
        return False

    except Exception as e:
        st.error(f"Firebase init error: {str(e)}")
        return False

# Initialize Firebase Auth on startup
initialize_firebase_auth()

# -------------------------
# DATASET LOADING
# -------------------------
@st.cache_data
def load_datasets():
    """Load master destinations and user patterns datasets"""
    try:
        # Load master destinations dataset
        master = pd.read_csv("master_destinations_v2.csv")
        
        # Load user preference patterns if available
        try:
            patterns = pd.read_csv("user_preference_patterns.csv")
        except:
            patterns = None
        
        return master, patterns
    except FileNotFoundError:
        st.error("❌ Dataset files not found. Please ensure master_destinations_v2.csv is in the project root.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading datasets: {str(e)}")
        return None, None

master, patterns = load_datasets()

# -------------------------
# Pexels setup
# -------------------------
PEXELS_AVAILABLE = False
pexels_error_message = ""

def initialize_pexels():
    """Initialize Pexels API with proper error handling"""
    global PEXELS_AVAILABLE, pexels_error_message
    
    try:
        if "PEXELS_API_KEY" not in st.secrets:
            pexels_error_message = "PEXELS_API_KEY not found in secrets"
            return False
        
        api_key = st.secrets["PEXELS_API_KEY"]
        
        if not api_key or len(api_key) < 10:
            pexels_error_message = "Invalid API key format"
            return False
        
        headers = {"Authorization": api_key}
        params = {"query": "test", "per_page": 1}
        response = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params)
        
        if response.status_code == 200:
            PEXELS_AVAILABLE = True
            return True
        else:
            pexels_error_message = f"Pexels API Error: {response.status_code}"
            return False
            
    except Exception as e:
        pexels_error_message = f"Pexels initialization error: {str(e)}"
        return False

initialize_pexels()

# =========================
# UNIFIED FEEDBACK SYSTEM (FIRESTORE ONLY)
# =========================
def save_feedback_to_firebase(module, feedback_type, target, value, metadata=None):
    try:
        if not FIREBASE_AVAILABLE or not db:
            print("Firebase not available")
            return False
        
        user_id = st.session_state.get("user_id")

        # DEBUG LINE
        print("DEBUG USER_ID:", user_id)

        if not user_id or user_id == "":
            st.error("❌ DEBUG: user_id missing. Feedback NOT saved.")
            print("ERROR: user_id missing in session_state")
            return False

        feedback_doc = {
            "user_id": user_id,
            "session_id": st.session_state.get("session_id", "unknown"),
            "module": module,
            "type": feedback_type,
            "target": target,
            "value": value,
            "metadata": metadata or {},
            "timestamp": firestore.SERVER_TIMESTAMP
        }

        db.collection("feedback").add(feedback_doc)

        print(f"[FEEDBACK SAVED] {user_id} | {module} | {target}")
        return True

    except Exception as e:
        print(f"Feedback save error: {str(e)}")
        return False

def save_feedback(city, feedback):
    os.makedirs("feedback", exist_ok=True)
    path = "feedback/feedback.csv"

    user_id = st.session_state.get("user_id", "anonymous")

    row = {
        "user_id": user_id,
        "session_id": st.session_state.get("session_id", "unknown"),
        "city": city,
        "feedback": feedback,
        "timestamp": datetime.now().isoformat()
    }

    try:
        df = pd.read_csv(path)
    except:
        df = pd.DataFrame(columns=row.keys())

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)

    if FIREBASE_AVAILABLE and db is not None:
        try:
            db.collection("user_feedback").add({
                "user_id": user_id,
                "session_id": st.session_state.get("session_id", "unknown"),
                "city": city,
                "feedback": feedback,
                "timestamp": firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            print(f"Firebase feedback error: {str(e)}")

# =========================
# CHATBOT CSV LOGGING
# =========================
def log_chatbot_interaction_to_csv(query, response, rating=None):
    import csv
    import os
    
    csv_file = "chatbot_interactions.csv"
    file_exists = os.path.exists(csv_file)

    user_id = st.session_state.get('user_id', 'anonymous')

    try:
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    'timestamp',
                    'user_id',
                    'session_id',
                    'query',
                    'response',
                    'rating'
                ])

            writer.writerow([
                datetime.now().isoformat(),
                user_id,
                st.session_state.get('session_id', 'unknown'),
                query[:200] if query else '',
                response[:500] if response else '',
                rating
            ])

        print(f"[CHATBOT LOG] {user_id} | {rating}")
        return True

    except Exception as e:
        print(f"CSV logging error: {str(e)}")
        return False

# =========================
# ADAPTIVE FEEDBACK ADJUSTMENT
# =========================
def get_user_destination_feedback_adjustment(city, user_id=None):
    """
    Get feedback adjustment for a destination based on user's previous feedback.
    Returns a score modifier (-0.15 to +0.15) based on feedback trends.
    
    - Negative feedback (low ratings, thumbs down) = reduce score
    - Positive feedback (high ratings, thumbs up) = boost score
    """
    if not FIREBASE_AVAILABLE or not db:
        return 0.0
    
    try:
        user_id = user_id or st.session_state.get('user_id')
        if not user_id:
            return 0.0
        
        # Query user's feedback for this destination - direct chaining
        feedback_docs = db.collection("feedback")\
            .where("user_id", "==", user_id)\
            .where("target", "==", city)\
            .stream()
        
        total_adjustment = 0.0
        feedback_count = 0
        
        for doc in feedback_docs:
            data = doc.to_dict()
            feedback_type = data.get('type', '')
            value = data.get('value')
            
            if feedback_type == 'rating' and value:
                # Rating 1-5: convert to adjustment (-0.1 to +0.1)
                rating = float(value)
                adjustment = (rating - 3) * 0.05  # 1=-0.1, 3=0, 5=+0.1
                total_adjustment += adjustment
                feedback_count += 1
                
            elif feedback_type == 'like':
                # Thumbs up/down: +0.05 or -0.05
                if value == 'up':
                    total_adjustment += 0.05
                elif value == 'down':
                    total_adjustment -= 0.05
                feedback_count += 1
        
        if feedback_count == 0:
            return 0.0
        
        # Average adjustment, capped at +/- 0.15
        avg_adjustment = total_adjustment / feedback_count
        return max(-0.15, min(0.15, avg_adjustment))
        
    except Exception as e:
        print(f"Feedback adjustment error: {str(e)}")
        return 0.0

# NOTE: V2 dataset already loaded at line 205 via load_datasets()
# DO NOT load old dataset - it would override the v2 data
# The load_datasets() function at line 185 is the active data loader

# =========================
# UTILITIES
# =========================
def clean_text_for_pdf(text):
    """Remove emojis and special characters that cause PDF rendering issues"""
    if not text:
        return text
    
    # Remove common emoji patterns and special characters
    import re
    
    # Remove emoji characters (Unicode range)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+",
        flags=re.UNICODE
    )
    
    # Remove emojis
    text = emoji_pattern.sub(r'', text)
    
    # Replace problematic characters
    text = text.replace('•', '-')
    text = text.replace('→', '>')
    text = text.replace('✓', 'Yes')
    text = text.replace('✗', 'No')
    text = text.replace('★', '*')
    text = text.replace('☆', '*')
    
    return text.strip()

# =========================
# AUTHENTICATION FUNCTIONS
# =========================
def sign_up(email, password, name):
    """Sign up a new user with Firebase REST API"""
    try:
        print(f"[v0] DEBUG: Sign up attempt for {email}")
        print(f"[v0] DEBUG: FIREBASE_AUTH_AVAILABLE={FIREBASE_AUTH_AVAILABLE}, FIREBASE_API_KEY set={bool(FIREBASE_API_KEY)}")
        
        if not FIREBASE_AUTH_AVAILABLE or not FIREBASE_API_KEY:
            msg = "Firebase not configured. Add FIREBASE_API_KEY to .streamlit/secrets.toml"
            print(f"[v0] DEBUG: {msg}")
            return False, msg
        
        # Firebase REST API endpoint for sign up
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
        
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        
        print(f"[v0] DEBUG: Calling Firebase signup endpoint")
        response = requests.post(url, json=payload, timeout=10)
        print(f"[v0] DEBUG: Firebase response status: {response.status_code}")
        
        # Show debug info on UI
        st.write("DEBUG STATUS:", response.status_code)
        st.write("DEBUG TEXT:", response.text)
        
        # Try to parse JSON response
        try:
            data = response.json()
        except:
            st.error(f"Non-JSON response: {response.text}")
            return False, f"Non-JSON response: {response.text}"
        
        print(f"[v0] DEBUG: Firebase response: {data}")
        
        if response.status_code == 200:
            user_id = data['localId']
            id_token = data['idToken']
            
            # Store user profile in Firestore
            if db:
                try:
                    db.collection('users').document(user_id).set({
                        'email': email,
                        'name': name,
                        'created_at': datetime.now(),
                        'user_id': user_id
                    })
                except Exception as firestore_error:
                    pass  # Continue even if Firestore fails
            
            return True, "Sign up successful!"
        else:
            error_msg = data.get('error', {}).get('message', 'Unknown error')
            
            if 'EMAIL_EXISTS' in error_msg:
                return False, "Email already exists. Try logging in instead."
            elif 'WEAK_PASSWORD' in error_msg:
                return False, "Password is too weak. Use at least 6 characters."
            elif 'INVALID_EMAIL' in error_msg:
                return False, "Invalid email format"
            
            return False, f"Sign up failed: {error_msg}"
            
    except Exception as e:
        return False, f"An error occurred: {str(e)}"

def sign_in(email, password):
    """Sign in user with Firebase REST API"""
    try:
        if not FIREBASE_AUTH_AVAILABLE or not FIREBASE_API_KEY:
            return False, None, None, "Firebase not configured. Add FIREBASE_API_KEY to .streamlit/secrets.toml"
        
        # Firebase REST API endpoint for sign in
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
        
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        # Show debug info on UI
        st.write("DEBUG STATUS:", response.status_code)
        st.write("DEBUG TEXT:", response.text)
        
        # Try to parse JSON response
        try:
            data = response.json()
        except:
            return False, None, None, f"Non-JSON response: {response.text}"
        
        if response.status_code == 200:
            user_id = data['localId']
            user_email = data['email']
            id_token = data['idToken']
            
            return True, user_id, user_email, "Sign in successful!"
        else:
            error_msg = data.get('error', {}).get('message', 'Unknown error')
            
            if 'INVALID_LOGIN_CREDENTIALS' in error_msg:
                return False, None, None, "Invalid email or password"
            elif 'USER_DISABLED' in error_msg:
                return False, None, None, "This account has been disabled"
            elif 'OPERATION_NOT_ALLOWED' in error_msg:
                return False, None, None, "Email/password sign in is not enabled"
            
            return False, None, None, f"Sign in failed: {error_msg}"
            
    except Exception as e:
        return False, None, None, f"Sign in error: {str(e)}"

def sign_out():
    """Sign out the current user"""
    st.session_state.user = None
    st.session_state.user_id = None
    st.session_state.user_email = None
    st.session_state.is_authenticated = False
    st.rerun()

def get_age_group(age):
    if age <= 25:
        return "18-25"
    elif age <= 35:
        return "26-35"
    elif age <= 45:
        return "36-45"
    elif age <= 55:
        return "46-55"
    else:
        return "56+"

def get_user_pattern(patterns, interest, age_group):
    if patterns is None:
        return None
    row = patterns[
        (patterns["interest"] == interest) &
        (patterns["age_group"] == age_group)
    ]
    return row.iloc[0] if len(row) > 0 else None

def get_dynamic_weights(pattern_row):
    if pattern_row is None:
        return {"experience": 0.6, "rating": 0.25, "duration": 0.15}
    return {"experience": 0.6, "rating": 0.25, "duration": 0.15}

# =========================
# FEATURE 3: ADVANCED SIMILARITY ENGINE (IMPROVED)
# =========================

def map_budget_to_numeric(budget):
    return {
        "Budget": 1,
        "Mid-range": 2,
        "Luxury": 3
    }.get(budget, 2)

def map_weather_to_temp(weather):
    return {
        "Cold": 8,
        "Pleasant": 20,
        "Warm": 28
    }.get(weather, 20)

def compute_similarity(master_df, user, patterns):
    """
    IMPROVED SIMILARITY ENGINE with:
    - Gaussian functions for better matching
    - Exponential weighting for ratings
    - Dynamic weights based on interest
    - Support for new interests like Cuisine
    """
    df = master_df.copy()

    user_budget_num = map_budget_to_numeric(user["budget"])
    user_temp_target = map_weather_to_temp(user["weather"])
    user_duration = user["duration"]
    season = user["season"].lower()
    interest = user["interest"].lower()

    # Experience similarity (handles Cuisine and other interests)
    col_name = f"{interest}_norm"
    if col_name in df.columns:
        df["experience_sim"] = df[col_name]
    else:
        df["experience_sim"] = 0.5

    # Rating similarity (IMPROVED with exponential boost)
    if "rating_norm" in df.columns:
        df["rating_sim"] = df["rating_norm"]
    elif "rating" in df.columns:
        df["rating_sim"] = (df["rating"] / 5.0) ** 1.2
    else:
        df["rating_sim"] = 0.5

    # Duration similarity (IMPROVED with Gaussian function)
    if "ideal_duration_days" in df.columns:
        df["duration_sim"] = np.exp(-((df["ideal_duration_days"] - user_duration) ** 2) / (2 * 3 ** 2))
    else:
        df["duration_sim"] = 0.5

    # Budget similarity (IMPROVED with exponential)
    if "budget_numeric" in df.columns:
        df["budget_sim"] = np.exp(-abs(df["budget_numeric"] - user_budget_num) / 2)
    else:
        df["budget_sim"] = 0.5

    # Climate similarity (IMPROVED)
    season_temp_col = f"{season}_avg_temp"
    if season_temp_col in df.columns:
        df["climate_sim"] = np.exp(-abs(df[season_temp_col] - user_temp_target) / 30)
    else:
        df["climate_sim"] = 0.5

    # DYNAMIC WEIGHTS BASED ON INTEREST
    age_group = get_age_group(user["age"])
    pattern_row = get_user_pattern(patterns, user["interest"], age_group)

    if interest == "cuisine":
        weights = {
            "experience": 0.40,
            "rating": 0.30,
            "duration": 0.10,
            "budget": 0.15,
            "climate": 0.05
        }
    elif interest == "adventure":
        weights = {
            "experience": 0.40,
            "rating": 0.20,
            "duration": 0.20,
            "budget": 0.15,
            "climate": 0.05
        }
    elif interest == "nature":
        weights = {
            "experience": 0.35,
            "rating": 0.20,
            "duration": 0.15,
            "budget": 0.15,
            "climate": 0.15
        }
    else:
        weights = {
            "experience": 0.35,
            "rating": 0.2,
            "duration": 0.15,
            "budget": 0.15,
            "climate": 0.15
        }

    df["final_score"] = (
        weights["experience"] * df["experience_sim"] +
        weights["rating"] * df["rating_sim"] +
        weights["duration"] * df["duration_sim"] +
        weights["budget"] * df["budget_sim"] +
        weights["climate"] * df["climate_sim"]
    )
    
    # ADAPTIVE IMPROVEMENT: Apply user feedback adjustment to scores
    # This modifies scores based on the user's previous feedback for each destination
    user_id = st.session_state.get('user_id')
    if user_id:
        df["feedback_adjustment"] = df["city"].apply(
            lambda city: get_user_destination_feedback_adjustment(city, user_id)
        )
        df["final_score"] = df["final_score"] + df["feedback_adjustment"]
        # Clamp scores to valid range
        df["final_score"] = df["final_score"].clip(lower=0, upper=1)
    
    return df.sort_values("final_score", ascending=False)

# =========================
# FIREBASE FUNCTIONS
# =========================
def save_to_firebase(user_input, ranked_results, session_id):
    """Save recommendations to Firebase"""
    if not FIREBASE_AVAILABLE or db is None:
        return None
    
    try:
        recommendations = []
        for _, row in ranked_results.iterrows():
            recommendations.append({
                "city": row["city"],
                "country": row["country"],
                "continent": row["continent"],
                "rating": float(row["avg_rating"]),
                "match_score": float(row["final_score"]),
                "budget_numeric": float(row["budget_numeric"]),
                "ideal_duration": int(row["ideal_duration_days"]),
                "description": row["description"],
                "culture_norm": float(row.get("culture_norm", 0)),
                "adventure_norm": float(row.get("adventure_norm", 0)),
                "nature_norm": float(row.get("nature_norm", 0)),
                "beach_norm": float(row.get("beach_norm", 0)),
                "cuisine_norm": float(row.get("cuisine_norm", 0))
            })
        
        doc_data = {
            "session_id": session_id,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "user_preferences": {
                "age": user_input["age"],
                "interest": user_input["interest"],
                "duration": user_input["duration"],
                "weather": user_input["weather"],
                "season": user_input["season"],
                "budget": user_input["budget"],
                "travel_style": user_input.get("travel_style", "Solo"),
                "activity_level": user_input.get("activity_level", "Moderate"),
                "accommodation": user_input.get("accommodation", "Hotel"),
                "dietary": user_input.get("dietary", "Omnivore"),
                "visa_requirement": user_input.get("visa_requirement", "No preference")
            },
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "itinerary_generated": False
        }
        
        doc_ref = db.collection("tourism_recommendations").add(doc_data)
        return doc_ref[1].id
        
    except Exception as e:
        return None

# =========================
# GEMINI FUNCTIONS
# =========================
@st.cache_data
def generate_destination_description(city, country):
    """Generate a detailed description for a destination using Gemini"""
    if not GEMINI_AVAILABLE:
        return f"{city}, {country} is a wonderful travel destination with unique culture, cuisine, and attractions."
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""Write a compelling 3-4 sentence travel description for {city}, {country}. 
        Include what makes it unique, what travelers can experience there, and why it's worth visiting.
        Keep it engaging but concise."""
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
            )
        )
        
        if response and response.text:
            return response.text.strip()
        else:
            return f"{city}, {country} is a wonderful travel destination with unique culture, cuisine, and attractions."
    except Exception as e:
        return f"{city}, {country} is a wonderful travel destination with unique culture, cuisine, and attractions."

def gemini_verify_recommendations(top_recommendations, user_input):
    """Verify recommendations using Gemini and provide confidence scores with explanations"""
    if not GEMINI_AVAILABLE:
        return top_recommendations.copy()
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        cities_list = ", ".join([f"{row['city']}, {row['country']}" for _, row in top_recommendations.iterrows()])
        
        prompt = f"""You are a travel expert. Given this traveler profile and recommended destinations, verify the relevance of each destination.

TRAVELER PROFILE:
- Age: {user_input['age']}
- Primary Interest: {user_input['interest']}
- Travel Style: {user_input.get('travel_style', 'Solo')}
- Activity Level: {user_input.get('activity_level', 'Moderate')}
- Budget: {user_input['budget']}
- Duration: {user_input['duration']} days
- Accommodation: {user_input.get('accommodation', 'Hotel')}
- Dietary: {user_input.get('dietary', 'Omnivore')}

RECOMMENDED CITIES: {cities_list}

For EACH city, provide in this exact format (one line per city):
[CITY], [COUNTRY] | Confidence: [0-100]% | Why: [2 sentence explanation of why this matches]

Be specific and reference the traveler's interests and preferences."""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=400,
            )
        )
        
        if response and response.text:
            verifications = {}
            lines = response.text.strip().split('\n')
            for line in lines:
                if '|' in line and 'Confidence:' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        try:
                            city_country = parts[0].strip()
                            confidence_str = parts[1].strip().replace('Confidence:', '').strip().replace('%', '')
                            confidence = int(float(confidence_str))
                            explanation = parts[2].strip().replace('Why:', '').strip()
                            
                            for _, row in top_recommendations.iterrows():
                                if row['city'].lower() in city_country.lower():
                                    verifications[row['city']] = {
                                        'confidence': confidence,
                                        'explanation': explanation
                                    }
                                    break
                        except:
                            pass
            
            result = top_recommendations.copy()
            result['ai_confidence'] = result['city'].map(
                lambda x: verifications.get(x, {}).get('confidence', 75)
            )
            result['ai_explanation'] = result['city'].map(
                lambda x: verifications.get(x, {}).get('explanation', 'Great match based on your preferences')
            )
            return result
        else:
            return top_recommendations.copy()
            
    except Exception as e:
        return top_recommendations.copy()

def gemini_weather_advice(city, climate, season, interest):
    """Generate detailed weather-based travel advice using Gemini"""
    fallback = f"{city} offers a {climate.lower()} climate during {season}, suitable for {interest.lower()} activities. Pack appropriately and stay hydrated."
    
    if not GEMINI_AVAILABLE:
        return fallback

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""You are a helpful travel assistant. 

City: {city}
Climate: {climate}
Season: {season}
Traveler Interest: {interest}

Provide 3-4 sentences with:
1. What the weather is typically like and what to expect
2. 2-3 specific activities best suited for this weather
3. Packing recommendations for the climate
4. One health/wellness tip (sun protection, hydration, etc.)

Keep it informative and practical."""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
            )
        )
        
        if response and response.text:
            return response.text.strip()
        else:
            return fallback
            
    except Exception as e:
        return fallback

def gemini_translate(text, language):
    """Translate text using Gemini"""
    if language == "English" or not GEMINI_AVAILABLE:
        return text

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""Translate the following text to {language}. 
Only provide the translation, nothing else.

Text to translate:
{text}"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2048,
            )
        )
        
        if response and response.text:
            return response.text.strip()
        else:
            return text
            
    except Exception as e:
        return text

def generate_itinerary(city, country, duration, user_input, city_row):
    """Generate complete itinerary"""
    if not GEMINI_AVAILABLE:
        return "Itinerary generation requires Gemini API"
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""You are an expert travel itinerary planner. Create a complete, detailed {duration}-day itinerary for {city}, {country}.

TRAVELER PROFILE:
- Primary Interest: {user_input['interest']}
- Budget Level: {user_input['budget']}
- Season: {user_input['season']}
- Weather Preference: {user_input['weather']}
- Age: {user_input['age']}
- Accommodation Type: {user_input.get('accommodation', 'Hotel')}
- Dietary Preferences: {user_input.get('dietary', 'Omnivore')}
- Accessibility Needs: {user_input.get('accessibility', 'No specific needs')}

INSTRUCTIONS - FOLLOW EXACTLY:
1. Start with a 3-4 sentence personalized introduction
2. For EACH day (1 to {duration}), provide COMPLETE details in this EXACT format:

**Day X - [Compelling Day Title]**
Morning: [Specific activity with exact location name, detailed description 100-150 words]
Lunch: [Specific restaurant name, cuisine type, recommended dishes 50-80 words]
Afternoon: [Specific activity with location and duration 100-150 words]
Evening: [Specific activity/dinner venue with details 100-150 words]
Budget Tip: [Estimated costs and money-saving tips 40-60 words]
Local Insight: [Cultural tip or insider knowledge 50-80 words]

CRITICAL RULES:
- Be EXTREMELY detailed with real location names and restaurants
- Include specific addresses or neighborhoods where relevant
- Provide realistic time durations for each activity
- Suggest actual dining establishments that exist in {city}
- Make recommendations match the {user_input['interest']} interest and {user_input['budget']} budget
- Respect dietary preference: {user_input.get('dietary', 'Omnivore')} (suggest {user_input.get('dietary', 'any')} restaurants and food options)
- Suggest accommodations matching {user_input.get('accommodation', 'Hotel')} preference
- Ensure all activities and locations accommodate: {user_input.get('accessibility', 'No specific needs')}
- Include practical logistics (opening hours, transportation, booking tips)
- Write COMPLETE sentences, not abbreviations
- DO NOT TRUNCATE or cut off any information
- DO NOT use emojis or special characters in the itinerary
- Ensure ALL {duration} days are COMPLETE with no missing sections

GENERATE THE FULL ITINERARY NOW:"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.8,
                max_output_tokens=32000,
                top_p=0.95,
                top_k=40
            )
        )
        
        if response and response.text:
            full_itinerary = response.text.strip()
            
            if len(full_itinerary) > 800:
                return full_itinerary
            else:
                return generate_itinerary_retry(city, country, duration, user_input)
        else:
            return "Failed to generate itinerary. Please try again."
            
    except Exception as e:
        return f"Error: {str(e)}"

def generate_itinerary_retry(city, country, duration, user_input):
    """Retry itinerary generation with a simpler prompt"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""Create a {duration}-day detailed travel itinerary for {city}, {country} for a {user_input['age']}-year-old interested in {user_input['interest']}.

For EACH day from 1 to {duration}, provide:
Day [number] - [Title]
Morning: [what to do and where]
Lunch: [restaurant and food]
Afternoon: [activities]
Evening: [dining and activities]
Tips: [practical information and costs]

WRITE EVERYTHING IN FULL DETAIL. INCLUDE ALL {duration} DAYS COMPLETELY."""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.8,
                max_output_tokens=16000,
                top_p=0.95,
                top_k=40
            )
        )
        
        if response and response.text:
            return response.text.strip()
        else:
            return "Unable to generate itinerary."
            
    except Exception as e:
        return f"Error in retry: {str(e)}"

# =========================
# VIDEO GENERATION FUNCTIONS
# =========================

def generate_video_caption(section_text, day_num, time_period):
    """Use Gemini to generate a short natural subtitle for one itinerary section"""
    fallback = f"Explore the highlights of the {time_period.lower()}"

    def clean_caption(text):
        if not text:
            return fallback

        text = text.strip().replace("\n", " ")
        text = text.replace('"', '').replace("'", "")
        text = re.sub(r'\s+', ' ', text).strip()

        # remove labels if Gemini adds them
        text = re.sub(r'^(caption|subtitle|summary)\s*:\s*', '', text, flags=re.IGNORECASE)

        # strip bullets / numbering
        text = re.sub(r'^[\-\*\d\.\)\s]+', '', text).strip()

        # keep caption readable but not too long
        if len(text) > 70:
            cut = text[:67].rstrip()
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            text = cut + "..."

        if len(text) < 12:
            return fallback

        return text

    if not GEMINI_AVAILABLE:
        return fallback

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
You are creating a subtitle for a travel recap video.

Task:
Write one short, natural caption that summarizes this itinerary section.

Rules:
- Write exactly one sentence
- Make it feel like a travel recap subtitle
- Summarize the experience, do not copy the itinerary text directly
- Keep it between 8 and 16 words
- No hashtags
- No emojis
- No quotation marks
- No labels like "Caption:"
- No bullet points

Return only the caption sentence.

Day: {day_num}
Time period: {time_period}

Itinerary section:
{section_text}
"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=60,
            )
        )

        if response and response.text:
            caption = clean_caption(response.text)
            return caption

        return fallback

    except Exception:
        return fallback
        
def parse_itinerary_into_days(itinerary_text):
    """Parse itinerary text into day-wise video data with AI-generated captions"""
    days_data = []

    day_pattern = r'\*\*Day\s+(\d+)\s*-\s*([^*]+)\*\*'
    day_matches = list(re.finditer(day_pattern, itinerary_text))

    if not day_matches:
        alt_pattern = r'Day\s+(\d+)\s*-\s*(.+)'
        alt_matches = list(re.finditer(alt_pattern, itinerary_text))
        if not alt_matches:
            return []
        day_matches = alt_matches

    video_periods = ['Morning:', 'Lunch:', 'Afternoon:', 'Evening:']

    def clean_text(text):
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.replace('*', '').replace('#', '')
        return text

    def extract_location_name(content, fallback):
        content = clean_text(content)
        first_sentence = content.split('.')[0].strip() if '.' in content else content
        first_sentence = re.sub(r'\([^)]*\)', '', first_sentence).strip()

        words = first_sentence.split()
        if len(words) >= 4:
            return " ".join(words[:4]).strip()
        elif first_sentence:
            return first_sentence[:40].strip()
        else:
            return fallback

    for i, match in enumerate(day_matches):
        day_num = match.group(1)
        day_title = match.group(2).strip()

        start = match.end()
        end = day_matches[i + 1].start() if i + 1 < len(day_matches) else len(itinerary_text)
        day_content = itinerary_text[start:end].strip()

        locations = []

        for period in video_periods:
            if period in day_content:
                start_idx = day_content.find(period)
                next_period_idx = len(day_content)

                for next_period in video_periods + ['Budget Tip:', 'Local Insight:', 'Tips:']:
                    found_idx = day_content.find(next_period, start_idx + len(period))
                    if found_idx != -1 and found_idx < next_period_idx:
                        next_period_idx = found_idx

                content = day_content[start_idx + len(period):next_period_idx].strip()

                if content:
                    location_name = extract_location_name(content, day_title)
                    ai_caption = generate_video_caption(content, day_num, period.replace(':', ''))

                    locations.append({
                        "time_period": period.replace(':', ''),
                        "location": location_name,
                        "caption": ai_caption,
                        "description": content
                    })

        if not locations:
            fallback_caption = generate_video_caption(day_content or day_title, day_num, "Day Recap")
            locations = [{
                "time_period": "All Day",
                "location": day_title,
                "caption": fallback_caption,
                "description": day_content
            }]

        days_data.append({
            "day_num": day_num,
            "day_title": day_title,
            "locations": locations
        })

    return days_data

def fetch_pexels_image(query, filename, page=1):
    """Fetch image from Pexels API and save locally"""
    try:
        if not PEXELS_AVAILABLE:
            return None

        pexels_key = st.secrets.get("PEXELS_API_KEY")
        if not pexels_key:
            return None

        headers = {"Authorization": pexels_key}
        params = {
            "query": query,
            "per_page": 3,
            "page": page,
            "orientation": "landscape"
        }

        response = requests.get(
            "https://api.pexels.com/v1/search",
            headers=headers,
            params=params,
            timeout=10
        )

        if response.status_code != 200:
            return None

        data = response.json()
        photos = data.get("photos", [])
        if not photos:
            return None

        photo = photos[0]
        image_url = photo["src"].get("medium") or photo["src"].get("large") or photo["src"].get("landscape")
        if not image_url:
            return None

        img_response = requests.get(image_url, timeout=10)
        if img_response.status_code != 200:
            return None

        with open(filename, "wb") as f:
            f.write(img_response.content)

        return filename

    except Exception:
        return None

# FEATURE 5: PEXELS IMAGE INTEGRATION
def get_city_image_pexels(city, width=800, height=500):
    try:
        if not PEXELS_AVAILABLE:
            return get_city_image(city)
        
        pexels_key = st.secrets.get("PEXELS_API_KEY")
        if not pexels_key:
            return get_city_image(city)
        
        headers = {"Authorization": pexels_key}
        params = {
            "query": f"{city} travel destination",
            "per_page": 1,
            "orientation": "landscape"
        }
        
        response = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("photos"):
                image_url = data["photos"][0]["src"]["landscape"]
                return image_url
        
        return get_city_image(city)
        
    except Exception as e:
        return get_city_image(city)

def generate_itinerary_video(itinerary_text, city, country, user_input):
    """Generate travel recap video with synced subtitles for every image clip"""
    if not PEXELS_AVAILABLE:
        st.error("Pexels API not configured.")
        return None

    temp_dir = None
    final_video = None

    try:
        days_data = parse_itinerary_into_days(itinerary_text)

        if not days_data:
            st.error("Could not parse itinerary into day-wise sections.")
            return None

        temp_dir = tempfile.mkdtemp()
        st.info(f"Generating video with {len(days_data)} day(s)...")

        day_clips = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        video_width = 960
        video_height = 540
        subtitle_bar_height = 90
        clip_duration = 1.8

        for day_idx, day_data in enumerate(days_data):
            status_text.write(f"Processing Day {day_data['day_num']} of {len(days_data)}...")

            locations = day_data.get("locations", [])
            image_clips = []

            if not locations:
                locations = [{
                    "location": f"{city} travel",
                    "time_period": "All Day",
                    "caption": f"Day {day_data['day_num']} - {day_data['day_title']}"
                }]

            for loc_idx, location in enumerate(locations):
                search_query = f"{location.get('location', city)} {city} travel"
                image_file = os.path.join(
                    temp_dir,
                    f"day_{day_data['day_num']}_loc_{loc_idx}.jpg"
                )

                success = fetch_pexels_image(search_query, image_file, page=1)

                if not success:
                    placeholder = PILImage.new("RGB", (video_width, video_height), color=(70, 130, 180))
                    placeholder.save(image_file)

                try:
                    img = PILImage.open(image_file).convert("RGB")
                    img = img.resize((video_width, video_height))
                    frame = np.array(img)

                    base_clip = ImageClip(frame).with_duration(clip_duration)

                    subtitle_text = location.get("caption", "").strip()
                    if not subtitle_text:
                        subtitle_text = f"Exploring the best of Day {day_data['day_num']}"
                    
                    subtitle_text = re.sub(r'\s+', ' ', subtitle_text).strip()

                    if len(subtitle_text) > 70:
                        cut = subtitle_text[:67].rstrip()
                        if " " in cut:
                            cut = cut.rsplit(" ", 1)[0]
                        subtitle_text = cut + "..."

                    subtitle_text = location.get("caption", "").strip()
                    if not subtitle_text:
                        subtitle_text = f"Exploring the best of Day {day_data['day_num']}"

                    subtitle_text = re.sub(r'\s+', ' ', subtitle_text).strip()
                    
                    if len(subtitle_text) > 70:
                        cut = subtitle_text[:67].rstrip()
                        if " " in cut:
                            cut = cut.rsplit(" ", 1)[0]
                        subtitle_text = cut + "..."
                    
                    try:
                        subtitle_bg = (
                            ColorClip(size=(video_width, subtitle_bar_height), color=(0, 0, 0))
                            .with_duration(clip_duration)
                            .with_opacity(0.65)
                            .with_position((0, video_height - subtitle_bar_height))
                        )

                        subtitle = (
                            TextClip(
                                text=subtitle_text,
                                font_size=24,
                                color="white",
                                method="caption",
                                size=(video_width - 100, subtitle_bar_height - 24)
                            )
                            .with_duration(clip_duration)
                            .with_position(("center", video_height - subtitle_bar_height + 12))
                        )

                        clip = CompositeVideoClip(
                            [base_clip, subtitle_bg, subtitle],
                            size=(video_width, video_height)
                        )
                    except Exception:
                        clip = base_clip

                    image_clips.append(clip)

                except Exception as clip_error:
                    st.warning(f"Skipping one image clip: {clip_error}")
                    continue

            if not image_clips:
                st.warning(f"No usable clips created for Day {day_data['day_num']}")
                continue

            try:
                day_video = concatenate_videoclips(image_clips, method="compose")
            except Exception as day_error:
                st.warning(f"Could not build Day {day_data['day_num']}: {day_error}")
                continue

            # Day intro card
            try:
                day_intro_bg = ColorClip(size=(video_width, video_height), color=(20, 20, 20)).with_duration(1.5)

                day_intro_text = (
                    TextClip(
                        text=f"Day {day_data['day_num']} - {day_data['day_title']}",
                        font_size=36,
                        color="white",
                        method="caption",
                        size=(video_width - 80, None)
                    )
                    .with_duration(1.5)
                    .with_position(("center", "center"))
                )

                day_intro = CompositeVideoClip(
                    [day_intro_bg, day_intro_text],
                    size=(video_width, video_height)
                )

                full_day_video = concatenate_videoclips([day_intro, day_video], method="compose")
            except Exception:
                full_day_video = day_video

            day_clips.append(full_day_video)
            progress_bar.progress((day_idx + 1) / len(days_data))

        if not day_clips:
            st.error("No valid day clips were created.")
            return None

        status_text.write("Merging all day clips...")
        final_video = concatenate_videoclips(day_clips, method="compose")

        # Main title intro
        try:
            title_bg = ColorClip(size=(video_width, video_height), color=(25, 25, 112)).with_duration(2)

            title_text = (
                TextClip(
                    text=f"{city}, {country}\nTravel Recap",
                    font_size=40,
                    color="white",
                    method="caption",
                    size=(video_width - 100, None)
                )
                .with_duration(2)
                .with_position(("center", "center"))
            )

            title_clip = CompositeVideoClip([title_bg, title_text], size=(video_width, video_height))
            final_video = concatenate_videoclips([title_clip, final_video], method="compose")
        except Exception:
            pass

        output_path = os.path.join(temp_dir, "output.mp4")
        status_text.write("Rendering MP4 file...")

        final_video.write_videofile(
            output_path,
            fps=12,
            codec="libx264",
            audio=False,
            preset="ultrafast",
            threads=1,
            logger=None
        )

        output_buffer = BytesIO()
        with open(output_path, "rb") as f:
            output_buffer.write(f.read())
        output_buffer.seek(0)

        status_text.write("Video generated successfully!")
        st.success("Video generated successfully!")
        return output_buffer

    except Exception as e:
        st.error(f"Video generation error: {str(e)}")
        return None

    finally:
        try:
            if final_video is not None:
                final_video.close()
        except:
            pass

        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception:
                pass

# =========================
# FEEDBACK
# =========================
def save_feedback(city, feedback):
    os.makedirs("feedback", exist_ok=True)
    path = "feedback/feedback.csv"

    row = {
        "city": city,
        "feedback": feedback,
        "timestamp": datetime.now()
    }

    try:
        df = pd.read_csv(path)
    except:
        df = pd.DataFrame(columns=row.keys())

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    
    if FIREBASE_AVAILABLE and db is not None:
        try:
            db.collection("user_feedback").add({
                "session_id": st.session_state.session_id,
                "city": city,
                "feedback": feedback,
                "timestamp": firestore.SERVER_TIMESTAMP
            })
        except:
            pass

# =========================
# IMAGE
# =========================
def get_city_image(city):
    """Fallback to Picsum"""
    city_hash = int(hashlib.md5(city.encode()).hexdigest(), 16)
    image_id = city_hash % 1000
    return f"https://picsum.photos/seed/{image_id}/800/500"

# =========================
# PDF GENERATION
# =========================
def create_weather_icon(weather_type):
    weather_icons = {
        "Cold": "❄️",
        "Pleasant": "🌤️",
        "Warm": "☀️"
    }
    return weather_icons.get(weather_type, "🌤️")

def generate_itinerary_pdf(city, country, weather, season, itinerary_text, city_row, user_input, language="English"):
    """Generate PDF from itinerary"""
    
    try:
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2d5aa6'),
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        )
        
        day_style = ParagraphStyle(
            'DayHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=8,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            leading=13
        )
        
        bullet_style = ParagraphStyle(
            'BulletStyle',
            parent=styles['Normal'],
            fontSize=9,
            alignment=TA_LEFT,
            spaceAfter=4,
            leading=12,
            leftIndent=20
        )
        
        info_style = ParagraphStyle(
            'InfoText',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#555555'),
            spaceAfter=6
        )
        
        content = []
        
        title = Paragraph(f"🌍 {city}, {country}", title_style)
        content.append(title)
        content.append(Spacer(1, 0.2*inch))
        
        weather_icon = create_weather_icon(weather)
        info_data = [
            ["📊 Destination Information", ""],
            ["Location:", f"{city}, {country}"],
            ["Weather:", f"{weather_icon} {weather}"],
            ["Season:", f"🗓️ {season}"],
            ["Rating:", f"⭐ {city_row['avg_rating']}/5.0"],
            ["Match Score:", f"🎯 {city_row['final_score']:.2f}"],
            ["Ideal Duration:", f"📅 {city_row['ideal_duration_days']} days"],
            ["Budget Level:", f"💰 {user_input['budget']}"],
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#e8f0f8')),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.HexColor('#1f4788')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))
        content.append(info_table)
        content.append(Spacer(1, 0.3*inch))
        
        content.append(Paragraph("📋 Your Personalized Itinerary", heading_style))
        content.append(Spacer(1, 0.1*inch))
        
        lines = itinerary_text.split('\n')
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                content.append(Spacer(1, 0.08*inch))
                
            elif stripped.startswith('**Day'):
                day_title = stripped.replace('**', '').strip()
                try:
                    content.append(Paragraph(day_title, day_style))
                    content.append(Spacer(1, 0.05*inch))
                except:
                    pass
                    
            elif stripped.startswith('Morning:') or stripped.startswith('Lunch:') or \
                 stripped.startswith('Afternoon:') or stripped.startswith('Evening:') or \
                 stripped.startswith('Budget Tip:') or stripped.startswith('Local Insight:') or \
                 stripped.startswith('Tips:'):
                cleaned = stripped.replace('**', '').replace('__', '').replace('_', '')
                cleaned = cleaned.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                
                try:
                    para = Paragraph(f"<b>{cleaned.split(':')[0]}:</b> {':'.join(cleaned.split(':')[1:]).strip()}", bullet_style)
                    content.append(para)
                except:
                    try:
                        para = Paragraph(cleaned, bullet_style)
                        content.append(para)
                    except:
                        pass
                        
            else:
                cleaned = stripped.replace('**', '').replace('__', '').replace('_', '')
                cleaned = cleaned.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                
                if cleaned and len(cleaned) > 3:
                    try:
                        para = Paragraph(cleaned, normal_style)
                        content.append(para)
                    except:
                        pass
        
        content.append(Spacer(1, 0.2*inch))
        
        footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} | Language: {language} | AI Cultural Tourism Engine"
        try:
            content.append(Paragraph(footer_text, info_style))
        except:
            pass
        
        try:
            doc.build(content)
            pdf_buffer.seek(0)
            return pdf_buffer
        except Exception as build_error:
            st.error(f"PDF build error: {str(build_error)}")
            return None
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

# =========================
# PAGE FUNCTIONS
# =========================

def home_page():
    st.title("🌍 AI Cultural Tourism Recommendation Engine")
    st.markdown("*Powered by Gemini AI for personalized travel recommendations*")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to Your AI Travel Companion
        
        Discover the world in a personalized way. Our AI-powered platform helps you:
        
        - **Find Perfect Destinations** - Get AI-recommended destinations tailored to your preferences
        - **Generate Itineraries** - Receive day-wise travel plans crafted just for you
        - **Get Smart Recommendations** - Discover hidden gems similar to your interests
        - **Multilingual Support** - Chat with our AI in your preferred language
        - **Create Travel Videos** - Generate beautiful travel recap videos
        - **Save Your Sessions** - All recommendations are securely stored
        
        **Ready to explore?** Go to Personalization to get started.
        """)
    
    with col2:
        st.info("""
        ### 🚀 Quick Start
        
        1. Go to **Personalization** to tell us about yourself
        2. Get **Recommendations** based on your profile
        3. Generate your personalized **Itinerary**
        4. Create a **Travel Video** recap
        5. Provide feedback to improve recommendations
        6. Chat with our **Chatbot** for more help
        """)
    
    st.markdown("---")
    with st.sidebar:
        st.subheader("🔑 Session Info")
        st.code(f"Session ID: {st.session_state.session_id[:8]}...")
        if st.session_state.firebase_doc_id:
            st.success(f"✅ Saved: {st.session_state.firebase_doc_id[:8]}...")
        
        st.markdown("---")
        st.subheader("🤖 AI Status")
        if GEMINI_AVAILABLE:
            st.success("✅ Gemini API Connected")
        else:
            st.error("❌ Gemini API Unavailable")
        
        st.markdown("---")
        st.subheader("🎥 Video Status")
        if PEXELS_AVAILABLE:
            st.success("��� Pexels API Connected")
        else:
            st.error("❌ Pexels API Unavailable")
        
        st.markdown("---")
        st.subheader("🔥 Firebase Status")
        if FIREBASE_AVAILABLE:
            st.success("✅ Firebase Connected")
        else:
            st.warning("⚠️ Firebase Unavailable")
        
        st.markdown("---")
        if st.button("🔄 Start New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.ranked_results = None
            st.session_state.cached_ranked_results = None
            st.session_state.user_input = None
            st.session_state.firebase_doc_id = None
            st.session_state.personalization_complete = False
            st.session_state.last_user_preferences = None
            st.session_state.cached_user_input = {
                'age': 30,
                'interest': 'Culture',
                'duration': 5,
                'weather': 'Pleasant',
                'season': 'Summer',
                'budget': 'Mid-range',
                'continent': 'All Continents',
                'country': 'All Countries'
            }
            st.session_state.itinerary_preferences = {
                'dietary': 'Omnivore',
                'accommodation': 'Hotel',
                'accessibility': 'No specific needs'
            }
            st.success("✅ Session reset! Go to Personalization to start fresh.")
            st.session_state.cached_ranked_results = None
            st.session_state.cached_itineraries = {}
            st.rerun()

def personalization_page():
    """
    FEATURE 1 & 2: Personalization with Cuisine + Proper Session Caching
    """
    st.title("📝 Personalization")
    st.markdown("Tell us about yourself so we can recommend the perfect destinations!")
    st.markdown("---")
    
    # Get cached data - THIS NOW WORKS PROPERLY
    cached_data = st.session_state.cached_user_input
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        age = st.slider(
            "Your Age",
            min_value=18,
            max_value=80,
            value=cached_data['age']
        )
        
        # FEATURE 1: CUISINE OPTION
        interest = st.selectbox(
            "Primary Interest",
            ["Culture", "Adventure", "Nature", "Beach", "Cuisine"],
            index=["Culture", "Adventure", "Nature", "Beach", "Cuisine"].index(cached_data['interest'])
        )
        
        trip_duration = st.slider(
            "Trip Duration (days)",
            1, 14,
            value=cached_data['duration']
        )
        
        season = st.selectbox(
            "Season",
            ["Spring", "Summer", "Autumn", "Winter"],
            index=["Spring", "Summer", "Autumn", "Winter"].index(cached_data['season'])
        )
    
    with col2:
        weather = st.selectbox(
            "Weather Preference",
            ["Cold", "Pleasant", "Warm"],
            index=["Cold", "Pleasant", "Warm"].index(cached_data['weather'])
        )
        
        budget = st.selectbox(
            "Budget Level",
            ["Budget", "Mid-range", "Luxury"],
            index=["Budget", "Mid-range", "Luxury"].index(cached_data['budget'])
        )
        
        continent = st.selectbox(
            "Preferred Continent",
            ["All Continents", "Africa", "Asia", "Europe", "North America", "South America", "Oceania", "Middle East"],
            index=["All Continents", "Africa", "Asia", "Europe", "North America", "South America", "Oceania", "Middle East"].index(cached_data.get('continent', 'All Continents'))
        )
        
        # Get countries based on continent selection
        if master is not None:
            if continent == "All Continents":
                countries_list = ["All Countries"] + sorted(master['country'].unique().tolist())
            else:
                continent_map = {
                    "Africa": "africa",
                    "Asia": "asia",
                    "Europe": "europe",
                    "North America": "north_america",
                    "South America": "south_america",
                    "Oceania": "oceania",
                    "Middle East": "middle_east"
                }
                filtered_countries = master[master['region'] == continent_map[continent]]['country'].unique()
                countries_list = ["All Countries"] + sorted(filtered_countries.tolist())
            
            country = st.selectbox(
                "Specific Country (Optional)",
                countries_list,
                index=0 if continent == "All Continents" else min(0, len(countries_list) - 1)
            )
        else:
            country = "All Countries"
    
    st.markdown("---")
    st.markdown("### Your Profile Summary")
    st.info(f"""
    **Age:** {age} years old | **Interest:** {interest} | **Duration:** {trip_duration} days
    **Season:** {season} | **Weather:** {weather} | **Budget:** {budget}
    **Preferred Region:** {continent} | **Country:** {country}
    """)
    
    st.markdown("---")
    
    # Check if preferences have changed to trigger new recommendations
    current_preferences = {
        "age": age,
        "interest": interest,
        "duration": trip_duration,
        "weather": weather,
        "season": season,
        "budget": budget,
        "continent": continent,
        "country": country
    }
    
    # DEBUG: Show if preferences changed
    preferences_changed = st.session_state.last_user_preferences != current_preferences
    if preferences_changed:
        st.info("📝 Preferences changed - generating new recommendations")
    
    if st.button("🎯 Get Recommendations", use_container_width=True, type="primary", key="get_recommendations_btn"):
        if master is None:
            st.error("❌ Dataset not available.")
        else:
            user_input = current_preferences

            # FEATURE 2: CACHE IN SESSION STATE - THIS PERSISTS NOW
            st.session_state.cached_user_input = user_input
            st.session_state.last_user_preferences = user_input

            with st.spinner("Finding your perfect destinations..."):
                # Filter by continent and country if specified
                filtered_master = master.copy()
                
                if continent != "All Continents":
                    continent_map = {
                        "Africa": "africa",
                        "Asia": "asia",
                        "Europe": "europe",
                        "North America": "north_america",
                        "South America": "south_america",
                        "Oceania": "oceania",
                        "Middle East": "middle_east"
                    }
                    filtered_master = filtered_master[filtered_master['region'] == continent_map[continent]]
                
                if country != "All Countries":
                    filtered_master = filtered_master[filtered_master['country'] == country]
                
                # FEATURE 4: TOP 5 RECOMMENDATIONS
                ranked = compute_similarity(filtered_master, user_input, patterns).head(5)

            # CACHE RECOMMENDATIONS IN SESSION STATE
            st.session_state.cached_ranked_results = ranked
            st.session_state.ranked_results = ranked
            st.session_state.user_input = user_input
            st.session_state.personalization_complete = True

            with st.spinner("Saving recommendations..."):
                doc_id = save_to_firebase(user_input, ranked, st.session_state.session_id)
                if doc_id:
                    st.session_state.firebase_doc_id = doc_id
                    st.success("✅ Recommendations saved!")

            st.rerun()

def recommendations_page():
    """
    FEATURE 4 & 5: Top 5 recommendations with Pexels images
    """
    st.title("⭐ Smart Recommendations")
    st.markdown("---")
    
    # Use cached results if recommendations are None
    if st.session_state.ranked_results is None:
        if st.session_state.cached_ranked_results is not None:
            st.session_state.ranked_results = st.session_state.cached_ranked_results
        else:
            st.info("Complete the **Personalization** step first to get recommendations.")
            return
    
    ranked = st.session_state.ranked_results
    user_input = st.session_state.user_input or st.session_state.cached_user_input
    season = user_input["season"]
    interest = user_input["interest"]
    
    # Apply Gemini verification to recommendations
    with st.spinner("🤖 Verifying recommendations with AI..."):
        ranked = gemini_verify_recommendations(ranked, user_input)
        st.session_state.ranked_results = ranked
    
    st.success(f"✨ Found {len(ranked)} perfect destinations for you!")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"💾 Session ID: `{st.session_state.session_id[:8]}...`")
    with col2:
        if st.button("📋 Generate Itinerary", type="primary", use_container_width=True, key="itinerary_btn"):
            st.session_state.show_itinerary_form = True
            st.rerun()
    
    st.markdown("---")

    # FEATURE 4: TOP RECOMMENDATION (FEATURED)
    if len(ranked) > 0:
        top_row = ranked.iloc[0]
        
        st.markdown("### 🌟 Your Top Pick")
        with st.container(border=True):
            # Full width image on top
            st.image(get_city_image_pexels(top_row['city']), use_container_width=True)
            
            # Info section
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"## {top_row['city']}")
                st.markdown(f"### {top_row['country']}")
            with col2:
                st.caption(f"📍 **Region:** {top_row['region'].title()}")
                st.caption(f"⭐ **Rating:** {top_row['avg_rating']:.1f}/5.0")
            with col3:
                st.caption(f"🎯 **Match:** {top_row['final_score']:.2f}")
            
            st.divider()
            
            # Weather & Activity Suggestions - Full Width
            st.subheader("🌤️ Weather & Activity Suggestions")
            advice = gemini_weather_advice(
                top_row["city"],
                user_input["weather"],
                season,
                interest
            )
            st.write(advice)
            
            st.divider()
            
            # Description section - Full Width
            st.subheader("📝 Description")
            
            # Generate description if not in dataset
            description = top_row.get("description", None)
            if description is None or pd.isna(description):
                with st.spinner("Generating description..."):
                    description = generate_destination_description(top_row['city'], top_row['country'])
            
            lang = st.selectbox(
                "Select language:",
                ["English", "Hindi", "Spanish", "French", "German"],
                key=f"lang_top_{top_row['city']}",
                help="Powered by Gemini AI translation"
            )
            
            if lang != "English":
                with st.spinner(f"Translating to {lang}..."):
                    translated = gemini_translate(description, lang)
                    st.write(translated)
            else:
                st.write(description)
            
            st.divider()
            
            # Feedback section
            st.subheader("How helpful was this recommendation?")
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("👍 Yes", key=f"up_top_{top_row['city']}", use_container_width=True):
                    save_feedback(top_row["city"], "up")
                    st.success("Thanks for the feedback!")
            with col2:
                if st.button("👎 No", key=f"down_top_{top_row['city']}", use_container_width=True):
                    save_feedback(top_row["city"], "down")
                    st.success("Thanks for the feedback!")
    
    st.markdown("---")
    
    # FEATURE 4: OTHER TOP PICKS (VERTICAL WITH ALL DETAILS)
    if len(ranked) > 1:
        st.markdown("### 🎯 Other Top Picks")
        
        for idx, (i, (_, row)) in enumerate(enumerate(ranked.iloc[1:].iterrows(), 2)):
            with st.container(border=True):
                # Header section
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"## {i}. {row['city']}, {row['country']}")
                with col2:
                    st.caption(f"⭐ {row['avg_rating']:.1f}/5.0")
                
                # Two column layout: Image and Info
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.caption(f"🎯 **Match Score:** {row['final_score']:.2f}")
                    
                    # Show AI Confidence if available
                    if 'ai_confidence' in row:
                        st.caption(f"🤖 **AI Confidence:** {row['ai_confidence']}%")
                        if 'ai_explanation' in row:
                            st.caption(f"✨ {row['ai_explanation']}")
                
                with col2:
                    # FEATURE 5: PEXELS IMAGE
                    st.image(get_city_image_pexels(row['city']), use_container_width=True)
                
                st.divider()
                
                # Weather & Activity Suggestions - Full Width, Not Collapsed by Default
                st.subheader("🌤️ Weather & Activity Suggestions")
                advice = gemini_weather_advice(
                    row["city"],
                    user_input["weather"],
                    season,
                    interest
                )
                st.write(advice)
                
                st.divider()
                
                # Description with Language Support - Full Width
                st.subheader("📝 Description")
                
                # Generate description if not in dataset
                description = row.get("description", None)
                if description is None or pd.isna(description):
                    with st.spinner("Generating description..."):
                        description = generate_destination_description(row['city'], row['country'])
                
                lang = st.selectbox(
                    "Select language:",
                    ["English", "Hindi", "Spanish", "French", "German"],
                    key=f"lang_pick_{i}_{row['city']}",
                    help="Powered by Gemini AI translation"
                )
                
                if lang != "English":
                    with st.spinner(f"Translating to {lang}..."):
                        translated = gemini_translate(description, lang)
                        st.write(translated)
                else:
                    st.write(description)
                
                st.divider()
                
                # Feedback buttons
                st.subheader("How helpful was this recommendation?")
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button("👍 Yes", key=f"up_pick_{i}_{row['city']}", use_container_width=True):
                        save_feedback(row["city"], "up")
                        st.success("Thanks for the feedback!")
                with col2:
                    if st.button("👎 No", key=f"down_pick_{i}_{row['city']}", use_container_width=True):
                        save_feedback(row["city"], "down")
                        st.success("Thanks for the feedback!")

def itinerary_page():
    st.title("📅 Itinerary Generator")
    st.markdown("---")
    
    # Use cached results if needed
    if st.session_state.ranked_results is None:
        if st.session_state.cached_ranked_results is not None:
            st.session_state.ranked_results = st.session_state.cached_ranked_results
        else:
            st.info("Complete the **Personalization** step first.")
            return
    
    ranked = st.session_state.ranked_results
    user_input = st.session_state.user_input or st.session_state.cached_user_input
    
    st.subheader("📋 Create Your Itinerary")
    
    cities = [row['city'] for _, row in ranked.iterrows()]
    
    selected_city = st.selectbox(
        "Select a city:",
        cities,
        key="itinerary_city_selector"
    )
    
    city_row = ranked[ranked['city'] == selected_city].iloc[0]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**{city_row['city']}, {city_row['country']}**")
        st.caption(f"Rating: {city_row['avg_rating']}/5.0")
    
    with col2:
        st.markdown(f"**Duration:** {city_row['ideal_duration_days']} days")
    
    duration = st.slider(
        "How many days?",
        min_value=1,
        max_value=14,
        value=min(user_input['duration'], int(city_row['ideal_duration_days'])),
        key="itinerary_duration"
    )
    
    st.markdown("---")
    st.markdown("### 🍽️ Itinerary Preferences")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        dietary = st.selectbox(
            "Dietary Preferences",
            ["Omnivore", "Vegetarian", "Vegan", "Gluten-Free"],
            index=0,
            key="itinerary_dietary"
        )
    
    with col2:
        accommodation = st.selectbox(
            "Accommodation Type",
            ["Hostel", "Hotel", "Resort", "Airbnb", "Luxury"],
            index=1,
            key="itinerary_accommodation"
        )
    
    with col3:
        accessibility = st.selectbox(
            "Accessibility Needs",
            ["No specific needs", "Wheelchair accessible", "Mobility assistance", "Visual assistance", "Hearing assistance"],
            index=0,
            key="itinerary_accessibility"
        )
    
    # Store itinerary preferences in session state
    st.session_state.itinerary_preferences = {
        'dietary': dietary,
        'accommodation': accommodation,
        'accessibility': accessibility
    }
    
    st.markdown("---")
    
    if st.button("Generate Itinerary", type="primary", use_container_width=True, key="generate_itinerary_btn"):
        with st.spinner("Generating itinerary..."):
            # Merge itinerary preferences with user input
            full_user_input = user_input.copy()
            full_user_input.update(st.session_state.itinerary_preferences)
            
            itinerary = generate_itinerary(city_row['city'], city_row['country'], duration, full_user_input, city_row)
            
            # Clean the itinerary for PDF export (remove emojis and special chars)
            cleaned_itinerary = clean_text_for_pdf(itinerary)
            
            st.session_state.current_itinerary = cleaned_itinerary
            st.session_state.current_city = city_row
            st.session_state.current_user_input = full_user_input
            
            # CACHE ITINERARY
            st.session_state.cached_itineraries[selected_city] = cleaned_itinerary
            st.rerun()
    
    if st.session_state.current_itinerary:
        itinerary = st.session_state.current_itinerary
        city_row = st.session_state.current_city
        user_input = st.session_state.current_user_input
        
        st.markdown("### 📅 Your Itinerary")
        st.success("✅ Generated!")
        
        st.text_area(
            "Full Itinerary",
            value=itinerary,
            height=3000,
            disabled=True,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📥 Download as PDF")
        
        pdf_language = st.selectbox(
            "Language:",
            ["English", "Hindi", "Spanish", "French", "German"],
            key="pdf_language_selector"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Generate PDF", type="secondary", use_container_width=True, key="generate_pdf_btn"):
                with st.spinner(f"Creating PDF..."):
                    pdf_buffer = generate_itinerary_pdf(
                        city_row['city'],
                        city_row['country'],
                        user_input['weather'],
                        user_input['season'],
                        itinerary,
                        city_row,
                        user_input,
                        pdf_language
                    )
                    
                    if pdf_buffer:
                        st.session_state.pdf_buffer = pdf_buffer
                        st.success("✅ PDF ready!")
                    else:
                        st.error("Failed to generate PDF.")
        
        with col2:
            if st.session_state.pdf_buffer:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=st.session_state.pdf_buffer,
                    file_name=f"{city_row['city']}_itinerary.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        # =========================
        # ITINERARY FEEDBACK SECTION
        # =========================
        st.markdown("---")
        st.markdown("### Rate This Itinerary")
        
        itinerary_rating = st.slider(
            "How would you rate this itinerary?",
            min_value=1,
            max_value=5,
            value=4,
            key="itinerary_rating_slider"
        )
        
        itinerary_feedback_text = st.text_input(
            "Additional comments (optional):",
            placeholder="What did you like or what could be improved?",
            key="itinerary_feedback_text"
        )
        
        if st.button("Submit Itinerary Feedback", type="secondary", use_container_width=True, key="submit_itinerary_feedback_btn"):
            success = save_feedback_to_firebase(
                module="itinerary",
                feedback_type="rating",
                target=selected_city,
                value=itinerary_rating,
                metadata={"text": itinerary_feedback_text}
            )
            if success:
                st.success("Thank you for your feedback!")
            else:
                st.warning("Could not save feedback. Please ensure you are logged in.")

def video_page():
    st.title("🎬 Travel Video Generator")
    st.markdown("---")

    if st.session_state.current_itinerary is None:
        st.info("Generate an itinerary first in the **Itinerary** tab.")
        return

    if not PEXELS_AVAILABLE:
        st.error("Pexels API not configured.")
        return

    itinerary = st.session_state.current_itinerary
    city_row = st.session_state.current_city
    user_input = st.session_state.current_user_input or st.session_state.cached_user_input

    st.success("Ready to generate your travel video!")

    st.markdown("### Video Preview Details")
    st.write(f"**Destination:** {city_row['city']}, {city_row['country']}")
    st.write(f"**Trip Style:** {user_input.get('interest', 'Travel')}")
    st.write(f"**Duration:** {user_input.get('duration', city_row.get('ideal_duration_days', 'N/A'))} days")

    if st.button("🎥 Generate Video", type="primary", use_container_width=True, key="generate_video_btn"):
        with st.spinner("Creating your travel video..."):
            video_buffer = generate_itinerary_video(
                itinerary_text=itinerary,
                city=city_row['city'],
                country=city_row['country'],
                user_input=user_input
            )

            if video_buffer:
                st.session_state.video_buffer = video_buffer
                st.session_state.video_generated = True
                st.success("✅ Video is ready!")
            else:
                st.session_state.video_generated = False
                st.error("Failed to generate video.")

    if st.session_state.get("video_generated") and st.session_state.get("video_buffer"):
        st.markdown("---")
        st.markdown("### 📥 Download Your Video")

        st.download_button(
            label="⬇️ Download MP4 Video",
            data=st.session_state.video_buffer,
            file_name=f"{city_row['city']}_travel_video.mp4",
            mime="video/mp4",
            use_container_width=True
        )
        
        # =========================
        # VIDEO FEEDBACK SECTION
        # =========================
        st.markdown("---")
        st.markdown("### Rate This Video")
        
        video_rating = st.slider(
            "How would you rate this video?",
            min_value=1,
            max_value=5,
            value=4,
            key="video_rating_slider"
        )
        
        video_feedback_text = st.text_input(
            "Additional comments (optional):",
            placeholder="What did you like or what could be improved?",
            key="video_feedback_text"
        )
        
        if st.button("Submit Video Feedback", type="secondary", use_container_width=True, key="submit_video_feedback_btn"):
            success = save_feedback_to_firebase(
                module="video",
                feedback_type="rating",
                target=city_row['city'],
                value=video_rating,
                metadata={"text": video_feedback_text}
            )
            if success:
                st.success("Thank you for your feedback!")
            else:
                st.warning("Could not save feedback. Please ensure you are logged in.")

def chatbot_page():
    st.title("💬 Multilingual Chatbot")
    st.markdown("---")
    
    st.info("""
    Chat with our AI travel assistant.
    
    Ask about:
    - Destination recommendations
    - Travel tips
    - Cultural information
    - Logistics and planning
    """)
    
    language = st.selectbox(
        "Select Language",
        ["English", "Hindi", "French", "Spanish", "German", "Japanese"],
        index=0
    )
    
    # Clear chat button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Clear Chat", use_container_width=True, key="clear_chat_btn"):
            st.session_state.chat_history = []
            st.rerun()
    
    st.markdown("### Chat")
    
    chat_container = st.container(height=400, border=True)
    
    with chat_container:
        # Show welcome message if no history
        if len(st.session_state.chat_history) == 0:
            st.markdown("**Bot:** Hello! How can I help you plan your perfect trip?")
        else:
            # Display chat history with feedback buttons for assistant messages
            for idx, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Bot:** {message['content']}")
                    
                    # Add thumbs up/down feedback for each assistant message
                    feedback_key = f"feedback_{idx}"
                    col_up, col_down, col_space = st.columns([1, 1, 8])
                    
                    with col_up:
                        if st.button("👍", key=f"thumbs_up_{idx}", help="Helpful response"):
                            save_feedback_to_firebase(
                                module="chatbot",
                                feedback_type="like",
                                target=message['content'][:200],  # First 200 chars as target
                                value="up"
                            )
                            # Find the preceding user query for this response
                            user_query = ""
                            if idx > 0 and st.session_state.chat_history[idx-1]["role"] == "user":
                                user_query = st.session_state.chat_history[idx-1]["content"]
                            # Log rating to CSV
                            log_chatbot_interaction_to_csv(user_query, message['content'], rating="up")
                            st.toast("Thanks for the feedback!")
                    
                    with col_down:
                        if st.button("👎", key=f"thumbs_down_{idx}", help="Not helpful"):
                            save_feedback_to_firebase(
                                module="chatbot",
                                feedback_type="like",
                                target=message['content'][:200],
                                value="down"
                            )
                            # Find the preceding user query for this response
                            user_query = ""
                            if idx > 0 and st.session_state.chat_history[idx-1]["role"] == "user":
                                user_query = st.session_state.chat_history[idx-1]["content"]
                            # Log rating to CSV
                            log_chatbot_interaction_to_csv(user_query, message['content'], rating="down")
                            st.toast("Thanks for the feedback!")
    
    user_input = st.text_input("Your message:", placeholder="Ask me anything!", key="chatbot_user_input")
    
    if st.button("Send", use_container_width=True, key="chatbot_send_btn"):
        if user_input:
            if not GEMINI_AVAILABLE:
                st.error("Gemini API unavailable.")
            else:
                # Add user message to history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                with st.spinner("Thinking..."):
                    try:
                        model = genai.GenerativeModel("gemini-2.5-flash")
                        
                        prompt = f"""You are a helpful travel assistant answering in {language}.
User Question: {user_input}

Provide a helpful, concise answer about travel."""

                        response = model.generate_content(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.7,
                                max_output_tokens=500,
                            )
                        )
                        
                        if response and response.text:
                            response_text = response.text.strip()
                            
                            # Add assistant response to history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response_text
                            })
                            
                            # Log interaction to CSV for learning model improvement
                            log_chatbot_interaction_to_csv(user_input, response_text)
                            
                            st.rerun()
                        else:
                            st.error("No response from bot")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# =========================
# AUTHENTICATION UI FUNCTIONS
# =========================
def login_page():
    """Login page with email authentication"""
    st.markdown("## Welcome Back!")
    st.markdown("Sign in to your account to continue")
    
    # DEBUG: Show Firebase status and secrets
    with st.expander("🔧 Firebase Configuration Status", expanded=not FIREBASE_AUTH_AVAILABLE):
        st.write("FIREBASE_AUTH_AVAILABLE:", FIREBASE_AUTH_AVAILABLE)
        st.write("FIREBASE_API_KEY set:", bool(FIREBASE_API_KEY))
        st.write("FIREBASE_PROJECT_ID set:", bool(FIREBASE_PROJECT_ID))
        
        # Check if secrets are loaded
        try:
            st.write("API KEY LENGTH:", len(st.secrets.get("FIREBASE_API_KEY", "")))
            st.write("API KEY:", st.secrets.get("FIREBASE_API_KEY", "NOT SET"))
            st.write("PROJECT ID:", st.secrets.get("FIREBASE_PROJECT_ID", "NOT SET"))
        except Exception as e:
            st.error(f"Error reading secrets: {str(e)}")
        
        if not FIREBASE_AUTH_AVAILABLE:
            st.error("Firebase authentication is not configured")
            st.markdown("""
            **To fix this:**
            1. Get your Firebase Web API Key from [Firebase Console](https://console.firebase.google.com)
            2. Add to `.streamlit/secrets.toml`:
            ```
            FIREBASE_API_KEY = "your-api-key-here"
            FIREBASE_PROJECT_ID = "tourism-recommendation-engine"
            ```
            3. Restart the app
            
            **Check the Streamlit terminal for [v0] DEBUG messages to see what's missing.**
            """)
            return
    
    st.subheader("Login with Email")
    
    email = st.text_input("Email", key="login_email_field")
    password = st.text_input("Password", type="password", key="login_password_field")
    
    if st.button("Sign In", use_container_width=True, type="primary", key="login_signin_submit"):
        if not email or not password:
            st.error("Please enter both email and password")
        else:
            success, user_id, user_email, message = sign_in(email, password)
            
            if success:
                st.session_state.is_authenticated = True
                st.session_state.user_id = user_id
                st.session_state.user_email = user_email
                st.session_state.user = {"email": user_email, "id": user_id}
                st.session_state.show_auth = False
                st.success(message)
                st.rerun()
            else:
                st.error(message)

def signup_page():
    """Sign up page for new users"""
    st.markdown("## Create Your Account")
    st.markdown("Join our community to get personalized travel recommendations")
    
    # DEBUG: Show Firebase status and secrets
    with st.expander("🔧 Firebase Configuration Status", expanded=not FIREBASE_AUTH_AVAILABLE):
        st.write("FIREBASE_AUTH_AVAILABLE:", FIREBASE_AUTH_AVAILABLE)
        st.write("FIREBASE_API_KEY set:", bool(FIREBASE_API_KEY))
        st.write("FIREBASE_PROJECT_ID set:", bool(FIREBASE_PROJECT_ID))
        
        # Check if secrets are loaded
        try:
            st.write("API KEY LENGTH:", len(st.secrets.get("FIREBASE_API_KEY", "")))
            st.write("API KEY:", st.secrets.get("FIREBASE_API_KEY", "NOT SET"))
            st.write("PROJECT ID:", st.secrets.get("FIREBASE_PROJECT_ID", "NOT SET"))
        except Exception as e:
            st.error(f"Error reading secrets: {str(e)}")
        
        if not FIREBASE_AUTH_AVAILABLE:
            st.error("Firebase authentication is not configured")
            st.markdown("""
            **To fix this:**
            1. Get your Firebase Web API Key from [Firebase Console](https://console.firebase.google.com)
            2. Add to `.streamlit/secrets.toml`:
            ```
            FIREBASE_API_KEY = "your-api-key-here"
            FIREBASE_PROJECT_ID = "tourism-recommendation-engine"
            ```
            3. Restart the app
            
            **Check the Streamlit terminal for [v0] DEBUG messages to see what's missing.**
            """)
            return
    
    name = st.text_input("Full Name", key="signup_name_field")
    email = st.text_input("Email", key="signup_email_field")
    password = st.text_input("Password (min 6 characters)", type="password", key="signup_password_field")
    password_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm_field")
    
    if st.button("Create Account", use_container_width=True, type="primary", key="signup_create_submit"):
        if not name or not email or not password:
            st.error("Please fill in all fields")
        elif password != password_confirm:
            st.error("Passwords do not match")
        elif len(password) < 6:
            st.error("Password must be at least 6 characters")
        else:
            success, message = sign_up(email, password, name)
            
            if success:
                st.success(message)
                st.markdown("### Now signing you in...")
                success_login, user_id, user_email, login_msg = sign_in(email, password)
                
                if success_login:
                    st.session_state.is_authenticated = True
                    st.session_state.user_id = user_id
                    st.session_state.user_email = user_email
                    st.session_state.user = {"email": user_email, "id": user_id}
                    st.session_state.show_auth = False
                    st.success("Account created and logged in!")
                    st.rerun()
            else:
                st.error(message)

def chatbot_page():
    """AI-powered chatbot for tourism recommendations and assistance"""
    st.set_page_config(page_title="Tourism Assistant", layout="wide")
    
    # Custom CSS for better chat UI
    st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0.5rem;
    }
    .user-message {
        background-color: #262730;
        color: white;
    }
    .assistant-message {
        background-color: #1f1f2e;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Tourism Assistant Chatbot")
    st.markdown("Ask me anything about destinations, itineraries, and travel advice!")
    
    # Initialize Gemini if needed
    if not GEMINI_AVAILABLE:
        st.error("Gemini API not configured. Please check your API key.")
        return
    
    # Sidebar settings
    with st.sidebar:
        st.subheader("Settings")
        
        # Language selection
        languages = ['English', 'French', 'Hindi', 'Japanese']
        st.session_state.chat_language = st.selectbox(
            "Response Language",
            languages,
            index=languages.index(st.session_state.chat_language)
        )
        
        # Clear chat history button
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Available Features")
        st.markdown("""
        - Ask about destinations
        - Modify your itinerary
        - Get travel tips & advice
        - Share trip feedback
        """)
    
    # Main chat interface
    col1, col2 = st.columns([0.7, 0.3], gap="medium")
    
    with col1:
        st.markdown("### Chat")
        
        # Chat history container with scrolling
        chat_container = st.container(border=True, height=400)
        
        with chat_container:
            if len(st.session_state.chat_history) == 0:
                st.markdown("*Start a conversation! Ask me about destinations, modify your itinerary, or get travel advice.*")
            else:
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.chat_message("user").markdown(message["content"])
                    else:
                        st.chat_message("assistant").markdown(message["content"])
        
        # Input area
        st.markdown("---")
        
        input_col1, input_col2 = st.columns([0.85, 0.15])
        
        with input_col1:
            user_input = st.text_input(
                "Your message:",
                placeholder="Ask about destinations, modify itinerary, tips...",
                label_visibility="collapsed",
                key="chat_input_" + str(len(st.session_state.chat_history))
            )
        
        with input_col2:
            send_button = st.button("Send", use_container_width=True, type="primary")
        
        # Process user input
        if send_button and user_input.strip():
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Build context from available session state
            context = build_chatbot_context()
            
            # Generate response with Gemini
            try:
                with st.spinner("Generating personalized response..."):
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    
                    # Create comprehensive prompt with full context
                    language_instruction = f"Respond in {st.session_state.chat_language}. Be personalized, specific, and helpful. "
                    full_prompt = f"""{language_instruction}You are an expert tourism assistant who provides personalized travel recommendations and advice based on the user's specific travel profile, interests, and previously recommended destinations.

=== USER'S TRAVEL PROFILE AND HISTORY ===
{context}

=== USER QUESTION ===
{user_input}

=== YOUR RESPONSE ===
Based on the user's profile and travel history above, provide a detailed, personalized response that references their specific interests, budget, duration, and previous recommendations where relevant. Be specific and helpful."""
                    
                    response = model.generate_content(full_prompt)
                    assistant_response = response.text
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                    
                    # Save chat to Firebase if available
                    try:
                        if FIREBASE_AVAILABLE and db and st.session_state.get("user_id"):
                            db.collection("chats").document(st.session_state.user_id).collection("messages").add({
                                "timestamp": datetime.now(),
                                "user_question": user_input,
                                "assistant_response": assistant_response,
                                "language": st.session_state.chat_language,
                                "context_available": bool(st.session_state.get("user_input") or st.session_state.get("cached_ranked_results"))
                            })
                    except Exception as e:
                        pass  # Silently fail Firebase save
                    
                    st.rerun()
            
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
        
        # Show available context at the bottom
        if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
            with st.expander("View Context Being Used", expanded=False):
                context = build_chatbot_context()
                st.markdown(context)
    
    with col2:
        st.markdown("### Trip Feedback")
        st.markdown("Share your travel experience with us!")
        
        # Feedback section
        with st.form("feedback_form"):
            destination = st.text_input("Destination visited")
            feedback_text = st.text_area(
                "Your feedback",
                placeholder="Tell us about your experience...",
                height=120
            )
            rating = st.slider("Rating", 1, 5, 4)
            
            # Image upload
            uploaded_files = st.file_uploader(
                "Upload trip images",
                type=["jpg", "jpeg", "png", "gif"],
                accept_multiple_files=True
            )
            
            submitted = st.form_submit_button("Submit Feedback", use_container_width=True)
            
            if submitted and feedback_text.strip() and destination.strip():
                # Save trip feedback using unified function
                metadata = {
                    "language": st.session_state.chat_language,
                    "images_count": len(uploaded_files) if uploaded_files else 0,
                    "trip_duration": st.session_state.get("current_user_input", {}).get("duration", "N/A")
                }
                
                success = save_feedback_to_firebase(
                    module="trip",
                    feedback_type="text",
                    target=destination,
                    value=feedback_text,
                    metadata=metadata
                )
                
                # Also save the rating separately
                if success:
                    save_feedback_to_firebase(
                        module="trip",
                        feedback_type="rating",
                        target=destination,
                        value=rating,
                        metadata={"rating_description": f"{rating}/5 stars"}
                    )
                    st.success("Thank you! Your feedback has been saved.")
                else:
                    st.warning("Could not save feedback. Please check your connection.")
            elif submitted:
                st.warning("Please fill in destination and feedback")

def build_chatbot_context():
    """Build detailed context from all previous modules for the chatbot"""
    context_parts = []
    
    try:
        # 1. USER PREFERENCES FROM PERSONALIZATION
        if st.session_state.get("user_input"):
            prefs = st.session_state.user_input
            if isinstance(prefs, dict):
                pref_text = "USER PROFILE:\n"
                pref_text += f"- Age: {prefs.get('age', 'Not specified')}\n"
                pref_text += f"- Primary Interest: {prefs.get('interest', 'Not specified')}\n"
                pref_text += f"- Trip Duration: {prefs.get('duration', 'Not specified')} days\n"
                pref_text += f"- Weather Preference: {prefs.get('weather', 'Not specified')}\n"
                pref_text += f"- Budget Level: {prefs.get('budget', 'Not specified')}\n"
                
                # Optional preferences
                if prefs.get('season'):
                    pref_text += f"- Preferred Season: {prefs.get('season')}\n"
                if prefs.get('travel_style'):
                    pref_text += f"- Travel Style: {prefs.get('travel_style')}\n"
                if prefs.get('activity_level'):
                    pref_text += f"- Activity Level: {prefs.get('activity_level')}\n"
                if prefs.get('accommodation'):
                    pref_text += f"- Accommodation Preference: {prefs.get('accommodation')}\n"
                
                context_parts.append(pref_text)
        
        # 2. RECOMMENDATIONS DATA
        if st.session_state.get("cached_ranked_results") is not None:
            results = st.session_state.cached_ranked_results
            
            # Handle DataFrame
            if hasattr(results, 'iterrows'):
                rec_text = "RECOMMENDED DESTINATIONS:\n"
                for idx, (_, row) in enumerate(results.head(5).iterrows(), 1):
                    city = row.get('city', 'Unknown')
                    country = row.get('country', 'Unknown')
                    rating = row.get('avg_rating', 'N/A')
                    score = row.get('final_score', 'N/A')
                    description = row.get('description', '')[:100]  # First 100 chars
                    rec_text += f"{idx}. {city}, {country} (Rating: {rating}/5, Match Score: {score:.2f})\n"
                    rec_text += f"   {description}...\n"
                context_parts.append(rec_text)
            
            # Handle list
            elif isinstance(results, list) and len(results) > 0:
                rec_text = "RECOMMENDED DESTINATIONS:\n"
                for idx, dest in enumerate(results[:5], 1):
                    if isinstance(dest, dict):
                        city = dest.get('city', dest.get('destination', 'Unknown'))
                        country = dest.get('country', '')
                        rec_text += f"{idx}. {city}, {country}\n"
                    else:
                        rec_text += f"{idx}. {str(dest)}\n"
                context_parts.append(rec_text)
        
        # 3. CURRENT ITINERARY DATA
        if st.session_state.get("current_itinerary"):
            itinerary = st.session_state.current_itinerary
            city = st.session_state.get("current_city", "Not specified")
            
            itin_text = f"CURRENT ITINERARY FOR {city}:\n"
            
            # Add user input details for this itinerary
            if st.session_state.get("current_user_input") and isinstance(st.session_state.current_user_input, dict):
                user_inp = st.session_state.current_user_input
                itin_text += f"Duration: {user_inp.get('duration', 'N/A')} days\n"
                itin_text += f"Budget: {user_inp.get('budget', 'N/A')}\n"
            
            # Add itinerary details
            if isinstance(itinerary, str):
                # If it's a string, add first 500 chars
                itin_text += f"Plan:\n{itinerary[:500]}...\n"
            elif isinstance(itinerary, dict):
                for key, value in list(itinerary.items())[:5]:
                    itin_text += f"- {key}: {str(value)[:100]}\n"
            
            context_parts.append(itin_text)
        
        # 4. CACHED ITINERARIES
        if st.session_state.get("cached_itineraries") and len(st.session_state.cached_itineraries) > 0:
            cache_text = "PREVIOUS ITINERARIES:\n"
            for city_name in list(st.session_state.cached_itineraries.keys())[:3]:
                cache_text += f"- {city_name}\n"
            context_parts.append(cache_text)
        
        # Combine all context
        if context_parts:
            context = "\n---\n".join(context_parts)
        else:
            context = "No previous planning data available. I'm ready to help you plan your trip!"
    
    except Exception as e:
        context = f"Using general travel knowledge. (Note: Could not load full context)"
    
    return context

def auth_page():
    """Authentication page for login and signup"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            login_page()
        
        with tab2:
            signup_page()

# =========================
# NAVIGATION
# =========================
pages = {
    "🏠 Home": home_page,
    "📝 Personalization": personalization_page,
    "⭐ Recommendations": recommendations_page,
    "📅 Itinerary": itinerary_page,
    "🎬 Video": video_page,
    "💬 Chatbot": chatbot_page,
}

# Check authentication and show appropriate content
initialize_session_state()

if not st.session_state.is_authenticated:
    st.sidebar.title("Tourism Engine")
    st.sidebar.info("🔐 Please log in to access all features")
    
    if st.sidebar.button("Login / Sign Up", use_container_width=True, type="primary", key="auth_button"):
        st.session_state.show_auth = True
    
    if st.session_state.get("show_auth", False):
        auth_page()
    else:
        st.title("Welcome to AI Cultural Tourism Engine")
        st.markdown("""
        This platform uses AI to recommend personalized travel destinations based on your preferences.
        
        **Login to:**
        - Get personalized destination recommendations
        - Save your preferences
        - Generate custom itineraries
        - Give feedback on recommendations
        
        Click the "Login / Sign Up" button to get started!
        """)
else:
    st.sidebar.title("Navigation")
    
    # Show user info in sidebar
    st.sidebar.markdown(f"👤 Logged in as: **{st.session_state.user_email}**")
    
    if st.sidebar.button("Logout", use_container_width=True, key="logout_button"):
        sign_out()
    
    st.sidebar.markdown("---")
    selected_page = st.sidebar.radio("Go to:", list(pages.keys()), key="page_radio")
    
    st.sidebar.markdown("---")

    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
**AI Cultural Tourism Platform**

*Fully Integrated with Gemini AI, Firebase & Pexels*

### ✨ FEATURES:
1. ✅ **Personalized Recommendations** - AI-powered destination matches
2. ✅ **Custom Itineraries** - Day-by-day travel plans
3. ✅ **Multi-language Support** - Get descriptions in any language
4. ✅ **Feedback System** - Help us improve recommendations
5. ✅ **Video Guides** - Watch destination guides

### 🔐 YOUR ACCOUNT:
- Secure Firebase authentication
- Your feedback helps personalize results
- All data is encrypted
""")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    
    # Render selected page
    pages[selected_page]()
