import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Cultural Tourism Engine",
    layout="wide"
)

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-pro")

GEMINI_AVAILABLE = True
try:
    model.generate_content("Ping")
except Exception:
    GEMINI_AVAILABLE = False

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    master = pd.read_csv("datasets/master_destinations.csv")
    patterns = pd.read_csv("datasets/user_preference_patterns.csv")
    return master, patterns

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
        return {
            "experience": 0.6,
            "rating": 0.25,
            "duration": 0.1,
            "accessibility": 0.05
        }

    return {
        "experience": 0.6,
        "rating": 0.25,
        "duration": 0.1,
        "accessibility": pattern_row["accessibility_rate"] * 0.05
    }

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
# GEMINI FUNCTIONS
# =========================
def gemini_weather_advice(city, climate):
    """
    Gemini-powered weather-based advice.
    Falls back safely if API fails.
    """
    try:
        prompt = f"""
        You are a travel assistant.
        The city is {city} and the climate is {climate}.
        Suggest suitable activities and travel tips in 2-3 sentences.
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception:
        # SAFE FALLBACK (NO CRASH)
        if climate == "Warm":
            return f"{city} is ideal for outdoor exploration, local sightseeing, and relaxed cultural walks."
        elif climate == "Cold":
            return f"{city} is better suited for indoor attractions, museums, cafés, and cultural experiences."
        else:
            return f"{city} offers a pleasant balance of outdoor sightseeing and cultural activities."

def gemini_translate(text, language):
    prompt = f"Translate this into {language}: {text}"
    return model.generate_content(prompt).text

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
# Images
# =========================
import hashlib

def get_city_image(city):
    """
    Deterministic image per city using Picsum.
    No redirects. Works reliably on Streamlit Cloud.
    """
    city_hash = int(hashlib.md5(city.encode()).hexdigest(), 16)
    image_id = city_hash % 1000  # Picsum has many images
    return f"https://picsum.photos/seed/{image_id}/800/500"

# =========================
# UI
# =========================
st.title("🌍 AI Cultural Tourism Recommendation Engine")

with st.form("user_form"):
    age = st.slider("Age", 18, 80, 25)
    interest = st.selectbox(
        "Primary Interest",
        ["Culture", "Adventure", "Nature", "Beach"]
    )
    duration = st.slider("Trip Duration (days)", 1, 14, 5)
    accessibility = st.checkbox("Accessibility required")
    weather = st.selectbox("Weather Preference", ["Warm", "Pleasant", "Cold"])
    season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
    budget = st.selectbox("Budget Level", ["Budget", "Mid-range", "Luxury"])

    submitted = st.form_submit_button("Get Recommendations")

# =========================
# EXECUTION
# =========================
if submitted:
    user_input = {
        "age": age,
        "interest": interest,
        "duration": duration,
        "accessibility": accessibility,
        "weather": weather,
        "season": season,
        "budget": budget
    }

    filtered = filter_cities(master, user_input)

    if filtered.empty:
        st.warning("No matching cities found. Try adjusting preferences.")
    else:
        ranked = rank_cities(filtered, user_input, patterns).head(3)

        for _, row in ranked.iterrows():
            st.subheader(row["city"])
            st.caption(f"{row['country']} ({row['continent']})")

            image_url = get_city_image(row["city"])
            st.image(image_url, use_column_width=True)

            st.write(f"⭐ Rating: {row['avg_rating']}")

            advice = gemini_weather_advice(
                row["city"],
                row[f"climate_{season.lower()}_label"]
            )
            st.info(advice)

            lang = st.selectbox(
                "Translate description to:",
                ["English", "Hindi", "Spanish"],
                key=row["city"]
            )

            translated = gemini_translate(row["description"], lang)
            st.write(translated)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍", key=f"up_{row['city']}"):
                    save_feedback(row["city"], "up")
            with col2:
                if st.button("👎", key=f"down_{row['city']}"):
                    save_feedback(row["city"], "down")
