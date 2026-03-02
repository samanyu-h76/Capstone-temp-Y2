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

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Cultural Tourism Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'ranked_results' not in st.session_state:
    st.session_state.ranked_results = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'firebase_doc_id' not in st.session_state:
    st.session_state.firebase_doc_id = None
if 'show_itinerary_form' not in st.session_state:
    st.session_state.show_itinerary_form = False
if 'personalization_complete' not in st.session_state:
    st.session_state.personalization_complete = False
if 'pdf_buffer' not in st.session_state:
    st.session_state.pdf_buffer = None
if 'current_itinerary' not in st.session_state:
    st.session_state.current_itinerary = None
if 'current_city' not in st.session_state:
    st.session_state.current_city = None
if 'current_user_input' not in st.session_state:
    st.session_state.current_user_input = None

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
            st.warning("Firebase credentials not found. Recommendations won't be saved.")
            return None
        
        firebase_creds = dict(st.secrets["FIREBASE_CREDENTIALS"])
        
        if "private_key" in firebase_creds:
            firebase_creds["private_key"] = str(firebase_creds["private_key"])
        
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
        
        FIREBASE_AVAILABLE = True
        return firestore.client()
        
    except Exception as e:
        st.warning(f"Firebase initialization failed: {str(e)}")
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

try:
    master, patterns = load_data()
except:
    master, patterns = None, None

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
                "budget_level": row["budget_level"],
                "ideal_duration": int(row["ideal_duration_days"]),
                "description": row["description"],
                "culture_score": float(row.get("culture_score", 0)),
                "adventure_score": float(row.get("adventure_score", 0)),
                "nature_score": float(row.get("nature_score", 0)),
                "beach_score": float(row.get("beach_score", 0))
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
                "budget": user_input["budget"]
            },
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "itinerary_generated": False
        }
        
        doc_ref = db.collection("tourism_recommendations").add(doc_data)
        return doc_ref[1].id
        
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

def generate_itinerary(city, country, duration, user_input, city_row):
    """Generate detailed itinerary using Gemini"""
    if not GEMINI_AVAILABLE:
        return "Itinerary generation requires Gemini API"
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""CRITICAL INSTRUCTION: YOU MUST GENERATE THE COMPLETE {duration}-DAY ITINERARY IN FULL.
DO NOT STOP EARLY. DO NOT TRUNCATE. DO NOT ABBREVIATE.
If you feel the response is getting long, KEEP WRITING ANYWAY until ALL {duration} days are complete.
IGNORE LENGTH LIMITS and FINISH the entire itinerary no matter how long it takes.

You will write approximately 250-400 words per day. For {duration} days, expect to write 1500-3200+ words total.

TRAVELER PROFILE:
- Interest: {user_input['interest']}
- Budget Level: {user_input['budget']}
- Season: {user_input['season']}
- Weather Preference: {user_input['weather']}
- Age: {user_input['age']}

DESTINATION INFO:
- City: {city}, {country}
- Rating: {city_row.get('avg_rating', 0)}/5
- Culture Score: {city_row.get('culture_score', 0)}/10
- Adventure Score: {city_row.get('adventure_score', 0)}/10
- Nature Score: {city_row.get('nature_score', 0)}/10

INSTRUCTIONS:
1. Start with a 2-3 sentence introduction explaining how {city} matches the traveler's interests
2. Create exactly {duration} day sections (Day 1 through Day {duration})
3. For EACH day, include:
   - **Day X - [Compelling Title]**
   - Morning: [Specific activity with location name and details]
   - Lunch: [Specific restaurant name, cuisine type, dish recommendations]
   - Afternoon: [Specific activity with location name and estimated duration]
   - Evening: [Specific activity/dinner with restaurant and dish suggestions]
   - Tips: [Practical advice, costs, best times, transportation tips]

4. Each day section should be 250-400 words
5. Include specific place names, not generic descriptions
6. Add estimated times for activities
7. Include restaurant names and cuisine types
8. Add budget considerations
9. Add cultural insights and etiquette tips

CRITICAL: CONTINUE WRITING ALL {duration} DAYS. DO NOT STOP AFTER DAY 3 OR 4.
If the response is interrupted, remember: YOU MUST WRITE ALL {duration} DAYS COMPLETELY."""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=8192,
            )
        )
        
        if response and response.text:
            full_text = response.text.strip()
            # Verify we got substantial content
            if len(full_text) > 100:
                return full_text
            else:
                return "Itinerary too short. Please try again."
        else:
            return "Unable to generate itinerary"
            
    except Exception as e:
        st.error(f"Itinerary generation failed: {str(e)}")
        return "Error generating itinerary"

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
    city_hash = int(hashlib.md5(city.encode()).hexdigest(), 16)
    image_id = city_hash % 1000
    return f"https://picsum.photos/seed/{image_id}/800/500"

# =========================
# PDF GENERATION
# =========================
def create_weather_icon(weather_type):
    """Return weather emoji based on type"""
    weather_icons = {
        "Cold": "❄️",
        "Pleasant": "🌤️",
        "Warm": "☀️"
    }
    return weather_icons.get(weather_type, "🌤️")

def generate_itinerary_pdf(city, country, weather, season, itinerary_text, city_row, user_input, language="English"):
    """Generate PDF from itinerary data using ReportLab"""
    
    try:
        # Create PDF buffer
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        # Define styles
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
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            leading=14
        )
        
        info_style = ParagraphStyle(
            'InfoText',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#555555'),
            spaceAfter=6
        )
        
        # Build PDF content
        content = []
        
        # Title
        title = Paragraph(f"🌍 {city}, {country}", title_style)
        content.append(title)
        content.append(Spacer(1, 0.2*inch))
        
        # Destination info section
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
        
        # Daily itinerary section
        content.append(Paragraph("📋 Your Personalized Itinerary", heading_style))
        content.append(Spacer(1, 0.1*inch))
        
        # Parse and format itinerary line-by-line to prevent truncation
        # Create many small Paragraph objects instead of few large ones
        itinerary_lines = itinerary_text.split('\n')
        current_section = []
        
        for i, line in enumerate(itinerary_lines):
            stripped_line = line.strip()
            
            if not stripped_line:
                # Empty line - add spacing
                if current_section:
                    for item in current_section:
                        content.append(item)
                    current_section = []
                content.append(Spacer(1, 0.08*inch))
                
            elif stripped_line.startswith('**Day'):
                # Day header - flush previous section and add new header
                if current_section:
                    for item in current_section:
                        content.append(item)
                    current_section = []
                
                content.append(Spacer(1, 0.15*inch))
                day_text = stripped_line.replace('**', '').strip()
                try:
                    content.append(Paragraph(f"<b>{day_text}</b>", heading_style))
                except Exception as ex:
                    content.append(Paragraph(day_text, heading_style))
                content.append(Spacer(1, 0.08*inch))
                
            elif stripped_line.startswith('-') or stripped_line.startswith('•'):
                # Bullet point
                bullet_text = stripped_line.lstrip('-• ').strip()
                # Escape problematic characters
                bullet_text = bullet_text.replace('**', '').replace('&', '&amp;')
                try:
                    para = Paragraph(f"• {bullet_text}", normal_style)
                    current_section.append(para)
                except Exception as ex:
                    # If paragraph fails, add as escaped text
                    escaped_text = bullet_text.replace('<', '&lt;').replace('>', '&gt;')
                    try:
                        para = Paragraph(f"• {escaped_text}", normal_style)
                        current_section.append(para)
                    except:
                        pass
                
            else:
                # Regular text line
                text_clean = stripped_line.replace('**', '').replace('__', '').replace('_', '')
                text_clean = text_clean.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                
                if text_clean:  # Only add non-empty text
                    try:
                        para = Paragraph(text_clean, normal_style)
                        current_section.append(para)
                    except Exception as ex:
                        # If paragraph fails due to special chars, skip
                        pass
        
        # Flush any remaining content
        if current_section:
            for item in current_section:
                content.append(item)
        
        content.append(Spacer(1, 0.2*inch))
        
        # Footer
        footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} | Language: {language} | AI Cultural Tourism Engine"
        try:
            content.append(Paragraph(footer_text, info_style))
        except:
            pass
        
        # Build PDF with error handling
        try:
            doc.build(content)
            pdf_buffer.seek(0)
            return pdf_buffer
        except Exception as build_error:
            st.error(f"PDF build error: {str(build_error)}")
            return None
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
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
        - **Save Your Sessions** - All recommendations are securely stored
        
        **Ready to explore?** Go to Personalization to get started.
        """)
    
    with col2:
        st.info("""
        ### 🚀 Quick Start
        
        1. Go to **Personalization** to tell us about yourself
        2. Get **Recommendations** based on your profile
        3. Generate your personalized **Itinerary**
        4. Provide feedback to improve recommendations
        5. Chat with our **Chatbot** for more help
        """)
    
    st.markdown("---")
    with st.sidebar:
        st.subheader("🔑 Session Info")
        st.code(f"Session ID: {st.session_state.session_id[:8]}...")
        if st.session_state.firebase_doc_id:
            st.success(f"✅ Saved: {st.session_state.firebase_doc_id[:8]}...")
        
        st.markdown("---")
        st.subheader("��� AI Status")
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
        
        st.markdown("---")
        if st.button("🔄 Start New Session", help="Clear current recommendations and start fresh"):
            st.session_state.ranked_results = None
            st.session_state.user_input = None
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.firebase_doc_id = None
            st.session_state.personalization_complete = False
            st.rerun()

def personalization_page():
    st.title("📝 Personalization")
    st.markdown("Tell us about yourself so we can recommend the perfect destinations!")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        age = st.slider("Your Age", min_value=18, max_value=80, value=30)
        
        interest = st.selectbox(
            "Primary Interest",
            ["Culture", "Adventure", "Nature", "Beach"]
        )
        
        trip_duration = st.slider("Trip Duration (days)", 1, 14, 5)
        
        season = st.selectbox(
            "Season",
            ["Spring", "Summer", "Autumn", "Winter"],
            index=1
        )
        
        weather = st.selectbox(
            "Weather Preference",
            ["Cold", "Pleasant", "Warm"]
        )
    
    with col2:
        budget = st.selectbox(
            "Budget Level",
            ["Budget", "Mid-range", "Luxury"],
            index=1
        )
        
        st.markdown("---")
        st.markdown("### Your Profile Summary")
        st.info(f"""
        **Age:** {age} years old
        **Primary Interest:** {interest}
        **Trip Duration:** {trip_duration} days
        **Preferred Season:** {season}
        **Weather Preference:** {weather}
        **Budget Level:** {budget}
        """)
    
    st.markdown("---")
    
    if st.button("🎯 Get Recommendations", use_container_width=True, type="primary"):
        if master is None:
            st.error("❌ Dataset not available. Please check if datasets are loaded correctly.")
        else:
            user_input = {
                "age": age,
                "interest": interest,
                "duration": trip_duration,
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
                
                st.session_state.ranked_results = ranked
                st.session_state.user_input = user_input
                st.session_state.personalization_complete = True
                
                with st.spinner("Saving recommendations..."):
                    doc_id = save_to_firebase(user_input, ranked, st.session_state.session_id)
                    if doc_id:
                        st.session_state.firebase_doc_id = doc_id
                        st.success(f"✅ Recommendations saved!")
                
                st.rerun()

def recommendations_page():
    st.title("⭐ Smart Recommendations")
    st.markdown("---")
    
    if st.session_state.ranked_results is None:
        st.info("Complete the **Personalization** step first to get recommendations.")
        return
    
    ranked = st.session_state.ranked_results
    user_input = st.session_state.user_input
    season = user_input["season"]
    interest = user_input["interest"]
    
    st.success(f"✨ Found {len(ranked)} perfect destinations for you!")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"💾 Session ID: `{st.session_state.session_id[:8]}...`")
    with col2:
        if st.button("📋 Generate Itinerary", type="primary", use_container_width=True, key="itinerary_btn"):
            st.session_state.show_itinerary_form = True
            st.rerun()
    
    st.markdown("---")

    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        with st.container(border=True):
            st.subheader(f"{i}. {row['city']}")
            st.caption(f"📍 {row['country']} ({row['continent']})")

            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(get_city_image(row['city']), use_container_width=True)
            
            with col2:
                st.write(f"⭐ **Rating:** {row['avg_rating']}/5.0")
                st.write(f"🎯 **Match Score:** {row['final_score']:.2f}")

            with st.expander("🌤️ Weather & Activity Suggestions", expanded=True):
                advice = gemini_weather_advice(
                    row["city"],
                    row[f"climate_{season.lower()}_label"],
                    season,
                    interest
                )
                st.info(advice)

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

def itinerary_page():
    st.title("📅 Itinerary Generator")
    st.markdown("---")
    
    if st.session_state.ranked_results is None:
        st.info("Complete the **Personalization** step first to generate itineraries.")
        return
    
    ranked = st.session_state.ranked_results
    user_input = st.session_state.user_input
    
    st.subheader("📋 Create Your Itinerary")
    
    cities = [row['city'] for _, row in ranked.iterrows()]
    
    selected_city = st.selectbox(
        "Select a city to generate itinerary:",
        cities,
        key="itinerary_city_selector",
        help="Choose one of your recommended cities"
    )
    
    city_row = ranked[ranked['city'] == selected_city].iloc[0]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**{city_row['city']}, {city_row['country']}**")
        st.caption(f"Rating: {city_row['avg_rating']}/5.0 | Match Score: {city_row['final_score']:.2f}")
    
    with col2:
        st.markdown(f"**Ideal Duration:** {city_row['ideal_duration_days']} days")
    
    duration = st.slider(
        "How many days do you want?",
        min_value=1,
        max_value=14,
        value=min(user_input['duration'], int(city_row['ideal_duration_days'])),
        key="itinerary_duration"
    )
    
    if st.button("Generate Itinerary", type="primary", use_container_width=True, key="generate_itinerary_btn"):
        with st.spinner("Generating your personalized itinerary..."):
            itinerary = generate_itinerary(city_row['city'], city_row['country'], duration, user_input, city_row)
            st.session_state.current_itinerary = itinerary
            st.session_state.current_city = city_row
            st.session_state.current_user_input = user_input
    
    # Display itinerary if available
    if 'current_itinerary' in st.session_state and st.session_state.current_itinerary:
        itinerary = st.session_state.current_itinerary
        city_row = st.session_state.current_city
        user_input = st.session_state.current_user_input
        
        st.markdown("### 📅 Your Personalized Itinerary")
        st.success("✅ Itinerary generated successfully!")
        
        # Display itinerary length info
        char_count = len(itinerary)
        st.caption(f"Itinerary length: {char_count:,} characters | Approx {char_count // 5} words")
        
        # Use text_area for reliable display of long text without truncation
        st.text_area(
            "Full Itinerary",
            value=itinerary,
            height=750,
            disabled=True,
            label_visibility="collapsed"
        )
        
        st.markdown("")  # Spacing
        
        # PDF Download Section
        st.markdown("### 📥 Download Itinerary")
        
        pdf_language = st.selectbox(
            "Select language for PDF:",
            ["English", "Hindi", "Spanish", "French", "German"],
            key="pdf_language_selector"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Generate PDF", type="secondary", use_container_width=True, key="generate_pdf_btn"):
                with st.spinner(f"Creating PDF in {pdf_language}..."):
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
                        st.success("✅ PDF generated successfully!")
                    else:
                        st.error("❌ Failed to generate PDF. Please try again.")
        
        with col2:
            if 'pdf_buffer' in st.session_state and st.session_state.pdf_buffer:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=st.session_state.pdf_buffer,
                    file_name=f"{city_row['city']}_itinerary_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.info("Generate PDF first to download")

def chatbot_page():
    st.title("💬 Multilingual Chatbot")
    st.markdown("---")
    
    st.info("""
    Chat with our AI travel assistant powered by Gemini AI.
    
    Ask about:
    - Destination recommendations
    - Travel tips and advice
    - Cultural information
    - Logistics and planning
    """)
    
    language = st.selectbox(
        "Select Chat Language",
        ["English", "Hindi", "French", "Spanish", "German", "Japanese"],
        index=0
    )
    
    st.markdown("### Chat Interface")
    
    chat_container = st.container(height=400, border=True)
    
    with chat_container:
        st.markdown("**Bot:** Hello! I'm your AI travel assistant. How can I help you plan your perfect cultural tour?")
    
    user_input = st.text_input("Type your message...", placeholder="Ask me anything about travel!")
    
    if st.button("Send", use_container_width=True):
        if user_input:
            if not GEMINI_AVAILABLE:
                st.error("Gemini API not available. Please check your configuration.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        model = genai.GenerativeModel("gemini-2.5-flash")
                        
                        prompt = f"""You are a helpful travel assistant answering in {language}.
User Question: {user_input}

Provide a helpful, concise answer about travel, destinations, or tourism."""

                        response = model.generate_content(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.7,
                                max_output_tokens=500,
                            )
                        )
                        
                        if response and response.text:
                            st.success(f"✅ Response received!")
                            st.info(f"**Bot:** {response.text.strip()}")
                        else:
                            st.error("No response from bot")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# =========================
# NAVIGATION
# =========================
pages = {
    "🏠 Home": home_page,
    "📝 Personalization": personalization_page,
    "⭐ Recommendations": recommendations_page,
    "📅 Itinerary": itinerary_page,
    "💬 Chatbot": chatbot_page,
}

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to:", list(pages.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
**AI Cultural Tourism Platform**

*Fully Integrated with Gemini AI & Firebase*

This platform provides AI-powered cultural tourism recommendations with real data processing and intelligent ranking.
""")

st.sidebar.markdown("---")
st.sidebar.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

pages[selected_page]()
