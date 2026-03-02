"""
Firebase utilities for database operations and data persistence
"""
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import json

FIREBASE_AVAILABLE = False

def initialize_firebase(firebase_creds_dict):
    """Initialize Firebase with proper error handling"""
    global FIREBASE_AVAILABLE
    
    try:
        # Check if already initialized
        if firebase_admin._apps:
            FIREBASE_AVAILABLE = True
            return firestore.client()
        
        if not firebase_creds_dict:
            st.warning("Firebase credentials not found. Recommendations won't be saved.")
            return None
        
        firebase_creds = firebase_creds_dict.copy()
        
        # Handle private key formatting
        if "private_key" in firebase_creds:
            firebase_creds["private_key"] = str(firebase_creds["private_key"])
        
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
        
        FIREBASE_AVAILABLE = True
        return firestore.client()
        
    except Exception as e:
        st.warning(f"Firebase initialization failed: {str(e)}")
        return None

def save_recommendation_to_firebase(db, doc_id, user_input, city_name, rank, score):
    """Save a recommendation to Firebase"""
    if not FIREBASE_AVAILABLE or not db:
        return False
    
    try:
        doc_ref = db.collection("recommendations").document(doc_id)
        doc_ref.update({
            "cities": firestore.ArrayUnion([{
                "city": city_name,
                "rank": rank,
                "score": score,
                "timestamp": firestore.SERVER_TIMESTAMP
            }])
        })
        return True
    except Exception as e:
        print(f"[v0] Error saving to Firebase: {str(e)}")
        return False

def get_user_preferences_from_firebase(db, session_id):
    """Retrieve user preferences from Firebase"""
    if not FIREBASE_AVAILABLE or not db:
        return None
    
    try:
        doc = db.collection("user_sessions").document(session_id).get()
        if doc.exists:
            return doc.to_dict()
        return None
    except Exception as e:
        print(f"[v0] Error retrieving from Firebase: {str(e)}")
        return None
