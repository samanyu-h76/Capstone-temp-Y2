"""
Gemini API utilities for generating itineraries and AI responses
"""
import streamlit as st
import google.generativeai as genai

GEMINI_AVAILABLE = False
gemini_error_message = ""

def initialize_gemini(api_key):
    """Initialize Gemini with proper error handling"""
    global GEMINI_AVAILABLE, gemini_error_message
    
    try:
        if not api_key or len(api_key) < 10:
            gemini_error_message = "Invalid API key format"
            return False
        
        genai.configure(api_key=api_key)
        
        # Test connection
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

def generate_itinerary(city, country, duration, user_input, city_row):
    """Generate detailed itinerary using Gemini - using multi-call approach for full content"""
    if not GEMINI_AVAILABLE:
        return "Itinerary generation requires Gemini API"
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # First call: Generate introduction and context
        intro_prompt = f"""You are a travel itinerary expert. Write a 3-4 sentence introduction for a {duration}-day trip to {city}, {country}.

Explain how this destination matches the traveler's profile:
- Interest: {user_input['interest']}
- Budget Level: {user_input['budget']}
- Season: {user_input['season']}
- Weather Preference: {user_input['weather']}
- Age: {user_input['age']}

Make it engaging and personalized. Only write the introduction, nothing else."""

        intro_response = model.generate_content(
            intro_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=300,
            )
        )
        
        introduction = intro_response.text.strip() if intro_response and intro_response.text else ""
        print(f"[v0] DEBUG: Introduction length: {len(introduction)} chars")
        
        # Second call: Generate each day separately to avoid truncation
        full_itinerary = introduction + "\n\n"
        
        for day_num in range(1, duration + 1):
            day_prompt = f"""Generate a DETAILED itinerary for Day {day_num} of a {duration}-day trip to {city}, {country}.

Traveler Profile:
- Interest: {user_input['interest']}
- Budget: {user_input['budget']}
- Season: {user_input['season']}

Format your response EXACTLY as follows (include all sections):

**Day {day_num} - [Compelling Title]**
- Morning: [Specific activity with location name, 50-100 words]
- Lunch: [Specific restaurant name, cuisine type, dish recommendations, 30-50 words]
- Afternoon: [Specific activity with location name and estimated duration, 50-100 words]
- Evening: [Specific activity/dinner with restaurant and suggestions, 50-100 words]
- Tips: [Practical advice, costs, best times, transportation, 40-60 words]

Write ONLY the day section above, nothing else. Be detailed and specific with place names, restaurant names, and activity details."""

            day_response = model.generate_content(
                day_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1000,
                )
            )
            
            if day_response and day_response.text:
                day_content = day_response.text.strip()
                full_itinerary += day_content + "\n\n"
                print(f"[v0] DEBUG: Day {day_num} length: {len(day_content)} chars")
            else:
                print(f"[v0] DEBUG: Failed to generate Day {day_num}")
        
        print(f"[v0] DEBUG: FINAL itinerary total length: {len(full_itinerary)} characters")
        
        if len(full_itinerary) > 500:
            return full_itinerary
        else:
            return "Failed to generate complete itinerary. Please try again."
            
    except Exception as e:
        print(f"[v0] DEBUG: Itinerary generation error: {str(e)}")
        st.error(f"Itinerary generation failed: {str(e)}")
        return "Error generating itinerary"
