import streamlit as st

# Improved session caching system
if "personalization_data" not in st.session_state:
    st.session_state.personalization_data = None  # Initialize personalization data
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []  # Initialize recommendations list
if "itineraries" not in st.session_state:
    st.session_state.itineraries = []  # Initialize itineraries list

# Function to set personalization data
def set_personalization_data(data):
    st.session_state.personalization_data = data

# Function to add recommendations
def add_recommendation(item):
    st.session_state.recommendations.append(item)

# Function to add an itinerary
def add_itinerary(itinerary):
    st.session_state.itineraries.append(itinerary)

# Streamlit app content
st.title("Your Travel App")

# Example usage to set personalization data
if st.button("Set Personalization Data"):
    set_personalization_data("Example User Preference")
    st.success("Personalization data set!")

# Display existing personalization data
if st.session_state.personalization_data:
    st.write("Current Personalization Data:", st.session_state.personalization_data)

# UI to add recommendations 
new_recommendation = st.text_input("Add a recommendation:")
if st.button("Add Recommendation") and new_recommendation:
    add_recommendation(new_recommendation)
    st.success("Added recommendation!")

# Display recommendations
st.write("Recommendations:", st.session_state.recommendations)

# UI to add itinerary
new_itinerary = st.text_input("Add an itinerary:")
if st.button("Add Itinerary") and new_itinerary:
    add_itinerary(new_itinerary)
    st.success("Added itinerary!")

# Display itineraries
st.write("Itineraries:", st.session_state.itineraries)