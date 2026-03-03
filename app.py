import streamlit as st
import requests

# Streamlit session state initialization
if 'cuisine' not in st.session_state:
    st.session_state.cuisine = ""
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'top_pick' not in st.session_state:
    st.session_state.top_pick = {}

# Page structure
PAGES = {
    "Home": "home",
    "Personalization": "personalization",
    "Recommendations": "recommendations",
    "Itinerary Generator": "itinerary",
    "Video Generator": "video",
    "Chatbot": "chatbot"
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    if selection == "Home":
        home_page()
    elif selection == "Personalization":
        personalization_page()
    elif selection == "Recommendations":
        recommendations_page()
    elif selection == "Itinerary Generator":
        itinerary_page()
    elif selection == "Video Generator":
        video_page()
    elif selection == "Chatbot":
        chatbot_page()

def home_page():
    st.title("Welcome to the Travel App")
    st.write("Explore personalized travel recommendations based on your cuisine preferences!")

def personalization_page():
    st.title("Personalization")
    st.write("Select your preferred cuisine:")
    cuisine_options = ["Italian", "Chinese", "Mexican", "Indian", "Thai"]
    st.session_state.cuisine = st.selectbox("Cuisine", cuisine_options)

def recommendations_page():
    st.title("Top Recommendations")
    if st.session_state.cuisine:
        st.session_state.recommendations = get_recommendations(st.session_state.cuisine)
        st.session_state.top_pick = st.session_state.recommendations[0] if st.session_state.recommendations else {}

        # Display top pick
        if st.session_state.top_pick:
            st.subheader(f"Top Pick: {st.session_state.top_pick['name']}")
            st.image(st.session_state.top_pick['image_url'])
            st.write(st.session_state.top_pick['description'])

        # Display other recommendations
        st.subheader("Other Recommendations:")
        for rec in st.session_state.recommendations[1:]:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.image(rec['image_url'], use_column_width=True)
            with col2:
                st.write(rec['name'])
            with col3:
                st.write(rec['description'])

def get_recommendations(cuisine):
    # Placeholder: Replace with API call or logic
    return [{
        'name': 'Pasta Primavera',
        'description': 'A colorful pasta dish with seasonal vegetables.',
        'image_url': 'https://www.example.com/image1.jpg'
    }, {
        'name': 'Fried Rice',
        'description': 'A delicious stir-fried rice dish with vegetables.',
        'image_url': 'https://www.example.com/image2.jpg'
    }]

def itinerary_page():
    st.title("Itinerary Generator")
    st.write("This feature will generate your travel itinerary...")

def video_page():
    st.title("Video Generator")
    st.write("This feature will generate your travel video...")

def chatbot_page():
    st.title("Chatbot")
    st.write("This feature will allow you to interact with a chatbot...")

if __name__ == "__main__":
    main()