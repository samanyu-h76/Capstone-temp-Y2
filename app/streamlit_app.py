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

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Cultural Tourism Engine",
    layout="wide"
)

# Initialize session state
if 'ranked_results' not in st.session_state:
    st.session_state.ranked_results = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = None
if 'session_id' not in st.session_state:
    # Generate unique session ID when app starts
    st.session_state.session_id = str(uuid.uuid4())
if 'firebase_doc_id' not in st.session_state:
    st.session_state.firebase_doc_id = None
if 'show_itinerary_form' not in st.session_state:
    st.session_state.show_itinerary_form = False
if 'selected_itinerary_city' not in st.session_state:
    st.session_state.selected_itinerary_city = None

# -------------------------
# Firebase setup
# -------------------------
FIREBASE_AVAILABLE = False

def initialize_firebase():
    """Initialize Firebase with proper error handling"""
    global FIREBASE_AVAILABLE
    
    try:
        # Check if already initialized
        if firebase_admin._apps:
            FIREBASE_AVAILABLE = True
            return firestore.client()
        
        # Get Firebase credentials from secrets
        if "FIREBASE_CREDENTIALS" not in st.secrets:
            st.warning("Firebase credentials not found. Recommendations won't be saved.")
            return None
        
        # Parse the credentials from TOML
        firebase_creds = dict(st.secrets["FIREBASE_CREDENTIALS"])
        
        # Handle the private_key: TOML multiline strings preserve newlines
        if "private_key" in firebase_creds:
            firebase_creds["private_key"] = str(firebase_creds["private_key"])
        
        # Initialize Firebase
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
        
        FIREBASE_AVAILABLE = True
        return firestore.client()
        
    except Exception as e:
        st.warning(f"Firebase initialization failed: {str(e)}")
        return None

# Initialize Firebase
db = initialize_firebase()

# -------------------------
# Gemini setup
# -------------------------
GEMINI_AVAILABLE = False
gemini_error_message = ""

def initialize_gemini():
    """Initialize Gemini with proper error handling and diagnostics"""
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

# Show status in sidebar
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
        with st.expander("Error Details"):
            st.write(gemini_error_message)
    
    st.markdown("---")
    st.subheader("🔥 Firebase Status")
    if FIREBASE_AVAILABLE:
        st.success("✅ Firebase Connected")
    else:
        st.warning("⚠️ Firebase Unavailable")
    
    # Reset session button
    st.markdown("---")
    if st.button("🔄 Start New Session", help="Clear current recommendations and start fresh"):
        st.session_state.ranked_results = None
        st.session_state.user_input = None
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.firebase_doc_id = None
        st.rerun()

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    """Load datasets with error handling"""
    try:
        master = pd.read_csv("datasets/master_destinations.csv")
        patterns = pd.read_csv("datasets/user_preference_patterns.csv")
        return master, patterns
    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

master, patterns = load_data()

# =========================
# UTILITIES
# =========================
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
# FILTER + RANK
# =========================
def filter_cities(df, user):
    return df[
        (df["budget_level"] == user["budget"]) &
        (df[f"climate_{user['season'].lower()}_label"] == user["weather"])
    ]

def rank_cities(df, user, patterns):
    age_group = get_age_group(user["age"])
    pattern_row = get_user_pattern(patterns, user["interest"], age_group)
    weights = get_dynamic_weights(pattern_row)

    df = df.copy()
    df["rating_norm"] = df["avg_rating"] / 5
    df["experience_match"] = df[f"{user['interest'].lower()}_score"]

    df["duration_match"] = 1 - (
        abs(df["ideal_duration_days"] - user["duration"]) /
        df["ideal_duration_days"]
    ).clip(0, 1)

    df["final_score"] = (
        weights["experience"] * df["experience_match"] +
        weights["rating"] * df["rating_norm"] +
        weights["duration"] * df["duration_match"]
    )

    return df.sort_values("final_score", ascending=False)

# =========================
# FIREBASE FUNCTIONS
# =========================
def save_to_firebase(user_input, ranked_results, session_id):
    """Save recommendations to Firebase with session tracking"""
    if not FIREBASE_AVAILABLE or db is None:
        return None
    
    try:
        # Prepare data for Firebase
        recommendations = []
        for _, row in ranked_results.iterrows():
            recommendations.append({
                "city": row["city"],
                "country": row["country"],
                "continent": row["continent"],
                "rating": float(row["avg_rating"]),
                "match_score": float(row["final_score"]),
                "budget_level": row["budget_level"],
                "ideal_duration": int(row["ideal_duration_days"]),
                "description": row["description"],
                "culture_score": float(row.get("culture_score", 0)),
                "adventure_score": float(row.get("adventure_score", 0)),
                "nature_score": float(row.get("nature_score", 0)),
                "beach_score": float(row.get("beach_score", 0))
            })
        
        # Create document data with session_id
        doc_data = {
            "session_id": session_id,  # Track session
            "timestamp": firestore.SERVER_TIMESTAMP,
            "user_preferences": {
                "age": user_input["age"],
                "interest": user_input["interest"],
                "duration": user_input["duration"],
                "weather": user_input["weather"],
                "season": user_input["season"],
                "budget": user_input["budget"]
            },
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "itinerary_generated": False  # Track if itinerary was generated
        }
        
        # Add to Firestore
        doc_ref = db.collection("tourism_recommendations").add(doc_data)
        
        return doc_ref[1].id  # Return document ID
        
    except Exception as e:
        st.error(f"Failed to save to Firebase: {e}")
        return None

def get_session_recommendations(session_id):
    """Get recommendations for current session"""
    if not FIREBASE_AVAILABLE or db is None:
        return None
    
    try:
        docs = db.collection("tourism_recommendations") \
            .where("session_id", "==", session_id) \
            .order_by("timestamp", direction=firestore.Query.DESCENDING) \
            .limit(1) \
            .stream()
        
        for doc in docs:
            return doc.to_dict()
        
        return None
        
    except Exception as e:
        st.error(f"Failed to retrieve session data: {e}")
        return None

# =========================
# GEMINI FUNCTIONS
# =========================
def gemini_weather_advice(city, climate, season, interest):
    """Generate weather-based travel advice using Gemini"""
    fallback = f"{city} offers a {climate.lower()} climate during {season}, suitable for {interest.lower()} activities and cultural exploration."
    
    if not GEMINI_AVAILABLE:
        return fallback

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""You are a helpful travel assistant. 

City: {city}
Climate: {climate}
Season: {season}
Traveler Interest: {interest}

Provide 2-3 sentences with:
1. What the weather is typically like
2. 2-3 specific activities or attractions suitable for this weather
3. One practical travel tip

Keep it concise, friendly, and actionable."""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=200,
            )
        )
        
        if response and response.text:
            return response.text.strip()
        else:
            return fallback
            
    except Exception as e:
        st.warning(f"AI advice generation failed: {str(e)}")
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
        st.warning(f"Translation failed: {str(e)}")
        return text

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
    
    # Also save to Firebase if available
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
    city_hash = int(hashlib.md5(city.encode()).hexdigest(), 16)
    image_id = city_hash % 1000
    return f"https://picsum.photos/seed/{image_id}/800/500"

# =========================
# UI
# =========================
st.title("🌍 AI Cultural Tourism Recommendation Engine")
st.markdown("*Powered by Gemini AI for personalized travel recommendations*")

with st.form("user_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 80, 25)
        interest = st.selectbox("Primary Interest", ["Culture", "Adventure", "Nature", "Beach"])
        duration = st.slider("Trip Duration (days)", 1, 14, 5)
    
    with col2:
        weather = st.selectbox("Weather Preference", ["Warm", "Pleasant", "Cold"])
        season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
        budget = st.selectbox("Budget Level", ["Budget", "Mid-range", "Luxury"])

    submitted = st.form_submit_button("🔍 Get Recommendations", use_container_width=True)

# =========================
# EXECUTION
# =========================
if submitted:
    user_input = {
        "age": age,
        "interest": interest,
        "duration": duration,
        "weather": weather,
        "season": season,
        "budget": budget
    }

    with st.spinner("Finding your perfect destinations..."):
        filtered = filter_cities(master, user_input)

    if filtered.empty:
        st.warning("⚠️ No matching cities found. Try adjusting your preferences.")
        st.session_state.ranked_results = None
        st.session_state.user_input = None
    else:
        ranked = rank_cities(filtered, user_input, patterns).head(3)
        
        # Store results in session state
        st.session_state.ranked_results = ranked
        st.session_state.user_input = user_input
        
        # Save to Firebase with session ID
        with st.spinner("Saving recommendations..."):
            doc_id = save_to_firebase(user_input, ranked, st.session_state.session_id)
            if doc_id:
                st.session_state.firebase_doc_id = doc_id
                st.success(f"✅ Recommendations saved for this session!")

# Display results from session state
if st.session_state.ranked_results is not None:
    ranked = st.session_state.ranked_results
    user_input = st.session_state.user_input
    season = user_input["season"]
    interest = user_input["interest"]
    
    st.success(f"✨ Found {len(ranked)} perfect destinations for you!")
    
    # Show session info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"💾 Session ID: `{st.session_state.session_id[:8]}...` - Use this for itinerary generation!")
    with col2:
        if st.button("📋 Generate Itinerary", type="primary", use_container_width=True, key="itinerary_btn"):
            st.session_state.show_itinerary_form = True
    
    st.markdown("---")

    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        with st.container():
            st.subheader(f"{i}. {row['city']}")
            st.caption(f"📍 {row['country']} ({row['continent']})")

            # Image
            st.image(get_city_image(row['city']), use_container_width=True)
            
            # Rating
            st.write(f"⭐ **Rating:** {row['avg_rating']}/5.0")
            st.write(f"🎯 **Match Score:** {row['final_score']:.2f}")

            # AI-generated weather advice
            with st.expander("🌤️ Weather & Activity Suggestions", expanded=True):
                advice = gemini_weather_advice(
                    row["city"],
                    row[f"climate_{season.lower()}_label"],
                    season,
                    interest
                )
                st.info(advice)

            # Description with translation
            st.write("📝 **Description:**")
            
            lang = st.selectbox(
                "Select language:",
                ["English", "Hindi", "Spanish", "French", "German"],
                key=f"lang_{row['city']}_{i}",
                help="Powered by Gemini AI translation"
            )

            if lang != "English":
                with st.spinner(f"Translating to {lang}..."):
                    translated = gemini_translate(row["description"], lang)
                    st.write(translated)
            else:
                st.write(row["description"])

            # Feedback
            st.write("**Was this recommendation helpful?**")
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("👍 Yes", key=f"up_{row['city']}_{i}"):
                    save_feedback(row["city"], "up")
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("👎 No", key=f"down_{row['city']}_{i}"):
                    save_feedback(row["city"], "down")
                    st.success("Thanks for your feedback!")

            st.markdown("---")

# =========================
# ITINERARY GENERATOR
# =========================
if st.session_state.show_itinerary_form and st.session_state.ranked_results is not None:
    st.markdown("---")
    st.subheader("📋 Itinerary Generator")
    
    ranked = st.session_state.ranked_results
    user_input = st.session_state.user_input
    
    # City selector for itinerary
    cities = [row['city'] for _, row in ranked.iterrows()]
    
    selected_city = st.selectbox(
        "Select a city to generate itinerary:",
        cities,
        key="itinerary_city_selector",
        help="Choose one of your recommended cities"
    )
    
    # Find selected city data
    city_row = ranked[ranked['city'] == selected_city].iloc[0]
    
    st.markdown("---")
    
    # Display city details
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📍 {city_row['city']}, {city_row['country']}")
        st.write(f"**Rating:** ⭐ {city_row['avg_rating']}/5.0")
        st.write(f"**Match Score:** 🎯 {city_row['final_score']:.2f}")
        st.write(f"**Ideal Duration:** {city_row['ideal_duration_days']} days")
        
        st.markdown("**Description:**")
        st.write(city_row["description"])
    
    with col2:
        st.metric("Culture Score", f"{city_row.get('culture_score', 0):.1f}/10")
        st.metric("Adventure Score", f"{city_row.get('adventure_score', 0):.1f}/10")
        st.metric("Nature Score", f"{city_row.get('nature_score', 0):.1f}/10")
        st.metric("Beach Score", f"{city_row.get('beach_score', 0):.1f}/10")
    
    st.markdown("---")
    
    # Itinerary generation
    st.subheader("📅 Day-by-Day Itinerary")
    
    duration = st.slider(
        "How many days?",
        min_value=1,
        max_value=14,
        value=min(user_input['duration'], int(city_row['ideal_duration_days'])),
        key="itinerary_duration"
    )
    
    if st.button("🚀 Generate Itinerary", type="primary", use_container_width=True, key="generate_itinerary_btn"):
        st.info("🔨 Generating personalized itinerary using Gemini AI...")
        
        try:
            if GEMINI_AVAILABLE:
                model = genai.GenerativeModel("gemini-2.5-flash")
                
                prompt = f"""Create a detailed {duration}-day itinerary for {selected_city}, {city_row['country']}.

Traveler Profile:
- Interest: {user_input['interest']}
- Budget Level: {user_input['budget']}
- Season: {user_input['season']}
- Weather Preference: {user_input['weather']}
- Age: {user_input['age']}

City Information:
- Culture Score: {city_row.get('culture_score', 0)}/10
- Adventure Score: {city_row.get('adventure_score', 0)}/10
- Nature Score: {city_row.get('nature_score', 0)}/10
- Beach Score: {city_row.get('beach_score', 0)}/10

Create a day-by-day itinerary with:
1. Morning activities
2. Afternoon activities
3. Evening activities
4. Budget-appropriate dining suggestions
5. Practical travel tips

Format each day clearly with times and activity types that match the traveler's interests and budget."""

                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.8,
                        max_output_tokens=2000,
                    )
                )
                
                if response and response.text:
                    st.markdown("### ✨ Your Personalized Itinerary")
                    st.markdown(response.text)
                    
                    # Save itinerary to Firebase
                    if FIREBASE_AVAILABLE and db is not None and st.session_state.firebase_doc_id:
                        try:
                            db.collection("tourism_recommendations").document(st.session_state.firebase_doc_id).update({
                                "itinerary_generated": True,
                                "itinerary_city": selected_city,
                                "itinerary_duration": duration,
                                "itinerary_content": response.text,
                                "itinerary_timestamp": firestore.SERVER_TIMESTAMP
                            })
                            st.success("✅ Itinerary saved to your session!")
                        except Exception as e:
                            st.warning(f"Could not save itinerary: {e}")
                else:
                    st.error("Failed to generate itinerary. Please try again.")
            else:
                st.error("Gemini AI is not available. Please check your API key.")
                
        except Exception as e:
            st.error(f"Error generating itinerary: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>AI Cultural Tourism Engine • Week 3 & 4 Features</small>
</div>
""", unsafe_allow_html=True)
