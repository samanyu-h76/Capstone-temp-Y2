import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime

# Configure Gemini (API key via Streamlit secrets or env variable)
genai.configure(api_key=st.secrets["AQ.Ab8RN6Kzp5PcOXxJCnUXnGY29Jcey7TtMkbiJqAt2r0gKIrtlQ"])
model = genai.GenerativeModel("gemini-pro")

## Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("datasets/places_master_dataset.csv")

places = load_data()

## User Input Form
st.title("🌍 AI Cultural Tourism Recommendation Engine")

with st.form("user_input_form"):
    age = st.slider("Age", 10, 80, 25)
    experience = st.selectbox(
        "Primary Interest",
        ["culture", "nature", "adventure", "beaches"]
    )
    duration = st.slider("Trip Duration (days)", 1, 14, 5)
    accessibility = st.checkbox("Accessibility required")
    weather_pref = st.selectbox(
        "Weather Preference",
        ["Warm", "Pleasant", "Cold"]
    )
    season = st.selectbox(
        "Preferred Season",
        ["Spring", "Summer", "Autumn", "Winter"]
    )
    budget = st.selectbox(
        "Budget Level",
        ["Budget", "Mid-range", "Premium"]
    )

    submitted = st.form_submit_button("Get Recommendations")

## Filtering Logic
def filter_places(df, budget, season, climate):
    return df[
        (df["budget_level"] == budget) &
        (df["best_season"] == season) &
        (df["climate_label"] == climate)
    ]
## Ranking Logic
def rank_places(df, interest):
    df = df.copy()
    df["rating_norm"] = df["avg_rating"] / 5

    df["final_score"] = (
        0.7 * df[interest] +
        0.3 * df["rating_norm"]
    )

    return df.sort_values("final_score", ascending=False)

## Gemini Weather Reasoning
def gemini_weather_advice(place, climate):
    prompt = f"""
    Given a destination with climate {climate},
    suggest suitable activities and travel tips.
    Destination: {place}
    """
    response = model.generate_content(prompt)
    return response.text

## Gemini Multilingual Description
def gemini_translate(text, language):
    prompt = f"Translate this into {language}: {text}"
    response = model.generate_content(prompt)
    return response.text

## Feedback Storage
def save_feedback(destination, feedback):
    row = {
        "destination": destination,
        "feedback": feedback,
        "timestamp": datetime.now()
    }
    try:
        df = pd.read_csv("feedback/feedback.csv")
    except:
        df = pd.DataFrame(columns=row.keys())

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv("feedback/feedback.csv", index=False)

## Main Execution
if submitted:
    filtered = filter_places(places, budget, season, weather_pref)

    if filtered.empty:
        st.warning("No destinations match your preferences.")
    else:
        ranked = rank_places(filtered, experience).head(3)

        for _, row in ranked.iterrows():
            st.subheader(row["destination_name"])
            st.write(f"📍 {row['country']} ({row['continent']})")
            st.write(f"⭐ Rating: {row['avg_rating']}")

            advice = gemini_weather_advice(
                row["destination_name"],
                row["climate_label"]
            )
            st.info(advice)

            language = st.selectbox(
                "Translate description to:",
                ["English", "Hindi", "Spanish"],
                key=row["destination_name"]
            )

            translated = gemini_translate(
                row["destination_name"], language
            )
            st.write(translated)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍", key=f"up_{row['destination_name']}"):
                    save_feedback(row["destination_name"], "up")
            with col2:
                if st.button("👎", key=f"down_{row['destination_name']}"):
                    save_feedback(row["destination_name"], "down")

