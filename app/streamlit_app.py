import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime
import os
import hashlib

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Cultural Tourism Engine",
    layout="wide"
)

# Initialize session state for storing results
if 'ranked_results' not in st.session_state:
    st.session_state.ranked_results = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = None

# -------------------------
# Gemini setup with better error handling
# -------------------------
GEMINI_AVAILABLE = False
gemini_error_message = ""

def initialize_gemini():
    """Initialize Gemini with 2.5-flashper error handling and diagnostics"""
    global GEMINI_AVAILABLE, gemini_error_message
    
    try:
        # Check if API key exists
        if "GEMINI_API_KEY" not in st.secrets:
            gemini_error_message = "GEMINI_API_KEY not found in secrets"
            return False
        
        api_key = st.secrets["GEMINI_API_KEY"]
        
        # Validate API key format
        if not api_key or len(api_key) < 10:
            gemini_error_message = "Invalid API key format"
            return False
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Test the connection with a simple 2.5-flashmpt
        model = genai.GenerativeModel("gemini-2.5-flash")  # Using stable gemini-2.5-flash
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

# Initialize Gemini
initialize_gemini()

# Show status in sidebar
with st.sidebar:
    st.subheader("🤖 AI Status")
    if GEMINI_AVAILABLE:
        st.success("✅ Gemini API Connected")
    else:
        st.error("❌ Gemini API Unavailable")
        with st.expander("Error Details"):
            st.write(gemini_error_message)
            st.info("""
            **To fix this:**
            1. Go to your Streamlit Cloud dashboard
            2. Click on your app settings
            3. Navigate to 'Secrets'
            4. Add:
            ```
            GEMINI_API_KEY = "your-api-key-here"
            ```
            5. Get your API key from: https://aistudio.google.com/app/apikey
            """)

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
# GEMINI FUNCTIONS (IM2.5-flashVED)
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

# =========================
# IMAGE (STABLE)
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

# Display results from session state (persists across reruns)
if st.session_state.ranked_results is not None:
    ranked = st.session_state.ranked_results
    user_input = st.session_state.user_input
    season = user_input["season"]
    interest = user_input["interest"]
    
    st.success(f"✨ Found {len(ranked)} perfect destinations for you!")
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>AI Cultural Tourism Engine • Week 3 Capstone 2.5-flashject</small>
</div>
""", unsafe_allow_html=True)
