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
from moviepy import *
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
    """
    Unified feedback function for all modules.
    
    Args:
        module: str - "recommendation", "itinerary", "video", "chatbot", "trip"
        feedback_type: str - "like", "rating", "text", "image"
        target: str - city/itinerary/video/chatbot message/destination being reviewed
        value: any - actual feedback (True/False for like, 1-5 for rating, text string, etc)
        metadata: dict - optional additional context (language, duration, etc)
    """
    try:
        if not FIREBASE_AVAILABLE or not db:
            st.warning("Cannot save feedback: Firebase not available")
            return False
        
        user_id = st.session_state.get("user_id")
        if not user_id:
            st.info("Sign in to save feedback")
            return False
        
        # Build feedback document
        feedback_doc = {
            "user_id": user_id,
            "session_id": st.session_state.get("session_id", "unknown"),
            "module": module,
            "type": feedback_type,
            "target": target,
            "value": value,
            "metadata": metadata or {},
            "timestamp": firestore.transforms.SERVER_TIMESTAMP
        }
        
        # Save to unified feedback collection
        db.collection("feedback").add(feedback_doc)
        return True
        
    except Exception as e:
        st.debug(f"Feedback save error: {str(e)}")
        return False

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
                max_output_tokens=500,
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
                max_output_tokens=600,
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
                max_output_tokens=500,
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
                max_output_tokens=8000,
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
                max_output_tokens=8000,
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

def parse_itinerary_into_days(itinerary_text):
    """Parse itinerary text into structured day data"""
    days_data = []
    
    day_pattern = r'\*\*Day\s+(\d+)\s*-\s*([^*]+)\*\*'
    day_matches = list(re.finditer(day_pattern, itinerary_text))
    
    for i, match in enumerate(day_matches):
        day_num = match.group(1)
        day_title = match.group(2).strip()
        
        start = match.end()
        end = day_matches[i + 1].start() if i + 1 < len(day_matches) else len(itinerary_text)
        day_content = itinerary_text[start:end]
        
        locations = []
        time_periods = ['Morning:', 'Lunch:', 'Afternoon:', 'Evening:']
        
        for period in time_periods:
            if period in day_content:
                start_idx = day_content.find(period)
                next_period_idx = float('inf')
                for next_period in time_periods:
                    idx = day_content.find(next_period, start_idx + 1)
                    if idx != -1:
                        next_period_idx = min(next_period_idx, idx)
                
                if next_period_idx == float('inf'):
                    next_period_idx = len(day_content)
                
                content = day_content[start_idx:next_period_idx].replace(period, '').strip()
                
                sentences = content.split('.')
                first_sentence = sentences[0].strip() if sentences else ''
                
                words = first_sentence.split()
                location = ' '.join(words[:2]) if len(words) > 1 else (words[0] if words else period.replace(':', ''))
                
                caption = f"{period.replace(':', '')}: {location}"
                
                locations.append({
                    'time_period': period.replace(':', ''),
                    'location': location,
                    'caption': caption,
                    'description': content[:80] if len(content) > 80 else content
                })
        
        days_data.append({
            'day_num': day_num,
            'day_title': day_title,
            'locations': locations if locations else [{'time_period': 'All Day', 'location': day_title, 'description': 'Explore the day'}]
        })
    
    return days_data

def fetch_pexels_image(query, filename, page=1):
    """Fetch image from Pexels API"""
    try:
        if not PEXELS_AVAILABLE:
            return None
        
        pexels_key = st.secrets.get("PEXELS_API_KEY")
        if not pexels_key:
            return None
        
        headers = {"Authorization": pexels_key}
        params = {"query": query, "per_page": 5, "page": page}
        
        response = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        if not data.get("photos"):
            return None
        
        photo = random.choice(data["photos"][:3])
        image_url = photo["src"]["landscape"]
        img_data = requests.get(image_url).content
        
        with open(filename, "wb") as f:
            f.write(img_data)
        
        return filename
        
    except Exception as e:
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
    """Generate travel video using Pexels images"""
    
    if not PEXELS_AVAILABLE:
        st.error("Pexels API not configured.")
        return None
    
    try:
        days_data = parse_itinerary_into_days(itinerary_text)
        
        if not days_data:
            st.error("Could not parse itinerary.")
            return None
        
        temp_dir = tempfile.mkdtemp()
        st.info(f"📹 Generating video with {len(days_data)} days...")
        
        day_clips = []
        progress_bar = st.progress(0)
        
        for day_idx, day_data in enumerate(days_data):
            st.write(f"Processing **Day {day_data['day_num']}: {day_data['day_title']}**")
            
            locations = day_data['locations']
            duration_per_location = 2
            image_clips = []
            
            for loc_idx, location in enumerate(locations):
                st.write(f"    🖼️ Fetching image for {location['location']}...")
                
                search_query = f"{location['location']} {city}"
                image_file = os.path.join(temp_dir, f"day_{day_data['day_num']}_loc_{loc_idx}.jpg")
                
                success = False
                for page in range(1, 3):
                    if fetch_pexels_image(search_query, image_file, page=page):
                        success = True
                        break
                
                if not success:
                    st.warning(f"Could not fetch image, using placeholder")
                    placeholder = PILImage.new('RGB', (1280, 720), color=(70, 130, 180))
                    placeholder.save(image_file)
                
                try:
                    img = PILImage.open(image_file)
                    img = img.resize((1280, 720))
                    frame = np.array(img)
                    
                    clip = ImageClip(frame).with_duration(duration_per_location)
                    
                    caption_text = location.get('caption', f"{location['time_period']}: {location['location']}")
                    
                    caption = TextClip(
                        text=caption_text,
                        font_size=24,
                        color="white",
                        size=(1000, None),
                        method="caption",
                        font="Arial"
                    ).with_duration(duration_per_location).with_position(("center", 600))
                    
                    clip = CompositeVideoClip([clip, caption])
                    image_clips.append(clip)
                    
                except Exception as e:
                    continue
            
            if not image_clips:
                st.warning(f"No images for Day {day_data['day_num']}")
                continue
            
            video = concatenate_videoclips(image_clips)
            
            day_header = TextClip(
                text=f"Day {day_data['day_num']} - {day_data['day_title']}",
                font_size=44,
                color="yellow",
                size=(1200, None),
                method="caption",
                font="Arial"
            ).with_duration(2).with_position(("center", "center"))
            
            day_header_video = CompositeVideoClip([
                ColorClip(size=(1280, 720), color=(0, 0, 0)).with_duration(2),
                day_header
            ])
            
            video = concatenate_videoclips([day_header_video, video])
            day_clips.append(video)
            
            progress_bar.progress((day_idx + 1) / len(days_data))
        
        if not day_clips:
            st.error("No valid day clips created.")
            return None
        
        st.write("📀 Merging all days...")
        final_video = concatenate_videoclips(day_clips)
        
        title_slide = TextClip(
            text=f"Your {city}, {country} Adventure",
            font_size=56,
            color="white",
            size=(1200, None),
            method="caption",
            font="Arial"
        ).with_duration(3).with_position(("center", "center"))
        
        title_slide_video = CompositeVideoClip([
            ColorClip(size=(1280, 720), color=(25, 25, 112)).with_duration(3),
            title_slide
        ])
        
        final_video = concatenate_videoclips([title_slide_video, final_video])
        
        st.write("💾 Rendering video...")
        output_buffer = BytesIO()
        output_path = os.path.join(temp_dir, "output.mp4")
        
        final_video.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio=False
        )
        
        with open(output_path, 'rb') as f:
            output_buffer.write(f.read())
        
        output_buffer.seek(0)
        
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
        
        st.success("✅ Video generated!")
        return output_buffer
        
    except Exception as e:
        st.error(f"Video generation error: {str(e)}")
        return None

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
            height=750,
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

def video_page():
    st.title("🎬 Travel Video Generator")
    st.markdown("---")
    
    if st.session_state.current_itinerary is None:
        st.info("Generate an itinerary first in the **Itinerary** tab.")
        return
    
    if not PEXELS_AVAILABLE:
        st.error("Pexels API not configured.")
        return
    
    st.success("✅ Ready to generate video!")
    
    itinerary = st.session_state.current_itinerary
    city_row = st.session_state.current_city
    user_input = st.session_state.current_user_input
    
    st.markdown(f"### 📍 {city_row['city']}, {city_row['country']}")
    
    st.markdown("""
    **Video includes:**
    - 🎬 Title slide
    - 📸 Pexels images for each location
    - 📝 Subtitles
    - 🔤 Day-by-day breakdown
    """)
    
    if st.button("🎬 Generate Video", type="primary", use_container_width=True, key="gen_video_btn"):
        with st.spinner("Creating video... (may take several minutes)"):
            video_buffer = generate_itinerary_video(
                itinerary,
                city_row['city'],
                city_row['country'],
                user_input
            )
            
            if video_buffer:
                st.session_state.video_buffer = video_buffer
                st.session_state.video_generated = True
                st.rerun()
    
    if st.session_state.video_generated and st.session_state.video_buffer:
        st.markdown("---")
        st.markdown("### ✅ Video Ready!")
        
        st.video(st.session_state.video_buffer)
        
        st.download_button(
            label="⬇️ Download Video",
            data=st.session_state.video_buffer,
            file_name=f"{city_row['city']}_travel_video.mp4",
            mime="video/mp4",
            use_container_width=True
        )
        
        st.success("🎉 Ready to share!")

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
    
    st.markdown("### Chat")
    
    chat_container = st.container(height=400, border=True)
    
    with chat_container:
        st.markdown("**Bot:** Hello! How can I help you plan your perfect trip?")
    
    user_input = st.text_input("Your message:", placeholder="Ask me anything!")
    
    if st.button("Send", use_container_width=True):
        if user_input:
            if not GEMINI_AVAILABLE:
                st.error("Gemini API unavailable.")
            else:
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
                            st.success("✅ Response:")
                            st.info(f"**Bot:** {response.text.strip()}")
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
                for idx, message in enumerate(st.session_state.chat_history):
                    if message["role"] == "user":
                        st.chat_message("user").markdown(message["content"])
                    else:
                        st.chat_message("assistant").markdown(message["content"])
                        
                        # Add feedback buttons for assistant messages
                        fb_col1, fb_col2, fb_col3 = st.columns([0.15, 0.15, 0.7])
                        with fb_col1:
                            if st.button("👍", key=f"like_{idx}", help="Helpful"):
                                save_feedback_to_firebase(
                                    module="chatbot",
                                    feedback_type="like",
                                    target=f"message_{idx}",
                                    value=True,
                                    metadata={"message_preview": message["content"][:100]}
                                )
                                st.success("Thanks for the feedback!", icon="✓")
                        with fb_col2:
                            if st.button("👎", key=f"dislike_{idx}", help="Not helpful"):
                                save_feedback_to_firebase(
                                    module="chatbot",
                                    feedback_type="like",
                                    target=f"message_{idx}",
                                    value=False,
                                    metadata={"message_preview": message["content"][:100]}
                                )
                                st.info("We'll improve!", icon="ℹ")
        
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

# =========================
# PAGE FUNCTIONS
# =========================
def home_page():
    """Home page"""
    st.title("Welcome to Tourism Engine")
    st.markdown("Navigate using the sidebar to explore features")

def personalization_page():
    """Personalization page - user preferences"""
    st.title("Personalization")
    st.markdown("Set your travel preferences here")

def recommendations_page():
    """Recommendations page with feedback"""
    st.title("Recommendations")
    st.markdown("Here are your personalized recommendations")
    
    # Example: Add feedback for recommendations if data exists
    if st.session_state.get("cached_ranked_results") is not None:
        st.success("Recommendations loaded")
        # Feedback UI will be added here based on displayed recommendations

def itinerary_page():
    """Itinerary page with feedback"""
    st.title("Itinerary")
    
    if not st.session_state.get("current_itinerary"):
        st.info("Generate an itinerary first in the Personalization section")
        return
    
    # Display itinerary
    city = st.session_state.get("current_city", "Unknown")
    st.subheader(f"Your Itinerary: {city}")
    
    itinerary_content = st.session_state.current_itinerary
    if isinstance(itinerary_content, str):
        st.markdown(itinerary_content)
    else:
        st.write(itinerary_content)
    
    # Itinerary Feedback Section
    st.markdown("---")
    st.subheader("Rate This Itinerary")
    
    feedback_col1, feedback_col2 = st.columns([0.5, 0.5])
    
    with feedback_col1:
        rating = st.slider("How helpful is this itinerary?", 1, 5, 4, key="itinerary_rating")
    
    with feedback_col2:
        feedback_text = st.text_area(
            "Optional feedback",
            placeholder="Tell us what you think about this itinerary...",
            height=100,
            key="itinerary_feedback_text"
        )
    
    # Submit button
    col1, col2, col3 = st.columns([0.3, 0.3, 0.4])
    with col1:
        if st.button("Submit Rating", use_container_width=True, type="primary"):
            # Save rating
            save_feedback_to_firebase(
                module="itinerary",
                feedback_type="rating",
                target=city,
                value=rating,
                metadata={
                    "itinerary_preview": itinerary_content[:200] if isinstance(itinerary_content, str) else str(itinerary_content)[:200],
                    "has_text_feedback": bool(feedback_text.strip())
                }
            )
            
            # Save text feedback if provided
            if feedback_text.strip():
                save_feedback_to_firebase(
                    module="itinerary",
                    feedback_type="text",
                    target=city,
                    value=feedback_text,
                    metadata={"rating": rating}
                )
            
            st.success("Thank you! Your feedback has been saved.")

def video_page():
    """Video page with feedback"""
    st.title("Video Guides")
    st.markdown("Watch destination guides here")

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
