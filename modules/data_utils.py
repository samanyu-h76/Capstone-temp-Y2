"""
Data processing and utility functions
"""
import streamlit as st
import pandas as pd
import hashlib
import numpy as np

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

def get_city_image(city):
    """Get a consistent image URL for a city based on its name hash"""
    city_hash = int(hashlib.md5(city.encode()).hexdigest(), 16)
    image_id = city_hash % 1000
    return f"https://picsum.photos/seed/{image_id}/800/500"

def filter_destinations(master_data, user_input, patterns_data):
    """Filter destinations based on user preferences"""
    try:
        filtered = master_data.copy()
        
        # Filter by interest
        interest_pattern = patterns_data[patterns_data['interest'] == user_input['interest']]
        if not interest_pattern.empty:
            preferred_scores = interest_pattern.iloc[0]
            
            # Score destinations based on relevance
            filtered['interest_match'] = (
                filtered['culture_score'] * preferred_scores.get('culture_weight', 0) +
                filtered['adventure_score'] * preferred_scores.get('adventure_weight', 0) +
                filtered['nature_score'] * preferred_scores.get('nature_weight', 0)
            )
        else:
            filtered['interest_match'] = 0
        
        # Filter by budget
        budget_mapping = {'Budget': 1, 'Mid-Range': 2, 'Luxury': 3}
        user_budget_level = budget_mapping.get(user_input['budget'], 2)
        filtered = filtered[filtered['price_level'] <= user_budget_level]
        
        # Filter by climate/weather
        weather_map = {'Cold': 'Cold', 'Pleasant': 'Temperate', 'Warm': 'Hot'}
        preferred_climate = weather_map.get(user_input['weather'], 'Temperate')
        filtered = filtered[filtered['climate_type'] == preferred_climate]
        
        return filtered
        
    except Exception as e:
        st.error(f"Error filtering destinations: {str(e)}")
        return master_data

def rank_destinations(filtered_data, user_input):
    """Rank filtered destinations based on user preferences"""
    try:
        ranked = filtered_data.copy()
        
        # Calculate final score
        ranked['final_score'] = (
            ranked['avg_rating'] * 0.3 +
            ranked['interest_match'] * 0.4 +
            ranked['accessibility_score'] * 0.2 +
            (100 - ranked['price_level'] * 20) * 0.1
        )
        
        # Sort by final score
        ranked = ranked.sort_values('final_score', ascending=False)
        
        return ranked.head(10)  # Return top 10
        
    except Exception as e:
        st.error(f"Error ranking destinations: {str(e)}")
        return filtered_data.head(10)
