import streamlit as st
import json

def login_page():
    """Login page with email and Google authentication"""
    st.markdown("## Welcome Back!")
    st.markdown("Sign in to your account to continue")
    
    # Import auth functions from main app
    import sys
    sys.path.insert(0, '/vercel/share/v0-project')
    from app import sign_in, FIREBASE_AUTH_AVAILABLE
    
    tab1, tab2 = st.tabs(["Email Login", "Google Sign In"])
    
    with tab1:
        st.subheader("Login with Email")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Sign In", use_container_width=True, type="primary"):
                if not email or not password:
                    st.error("Please enter both email and password")
                elif not FIREBASE_AUTH_AVAILABLE:
                    st.error("Authentication service unavailable")
                else:
                    success, user_id, user_email, message = sign_in(email, password)
                    
                    if success:
                        st.session_state.is_authenticated = True
                        st.session_state.user_id = user_id
                        st.session_state.user_email = user_email
                        st.session_state.user = {"email": user_email, "id": user_id}
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        
        with col2:
            st.info("📝 **New user?** Go to the Sign Up tab to create an account")
    
    with tab2:
        st.subheader("Login with Google")
        st.info("Google Sign-In feature coming soon. Please use email login for now.")

def signup_page():
    """Sign up page for new users"""
    st.markdown("## Create Your Account")
    st.markdown("Join our community to get personalized travel recommendations")
    
    # Import auth functions from main app
    import sys
    sys.path.insert(0, '/vercel/share/v0-project')
    from app import sign_up, sign_in, FIREBASE_AUTH_AVAILABLE
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        name = st.text_input("Full Name", key="signup_name")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password (min 6 characters)", type="password", key="signup_password")
        password_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm")
        
        if st.button("Create Account", use_container_width=True, type="primary"):
            if not name or not email or not password:
                st.error("Please fill in all fields")
            elif password != password_confirm:
                st.error("Passwords do not match")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters")
            elif not FIREBASE_AUTH_AVAILABLE:
                st.error("Authentication service unavailable")
            else:
                success, message = sign_up(email, password, name)
                
                if success:
                    st.success(message)
                    st.markdown("### Now signing you in...")
                    success_login, user_id, user_email, login_msg = sign_in(email, password)
                    
                    if success_login:
                        st.session_state.is_authenticated = True
                        st.session_state.user_id = user_id
                        st.session_state.user_email = user_email
                        st.session_state.user = {"email": user_email, "id": user_id}
                        st.success("Account created and logged in!")
                        st.rerun()
                else:
                    st.error(message)
    
    with col2:
        st.info("""
        ### Security Commitment
        - Your password is securely encrypted
        - We never share your data with third parties
        - You control your account preferences
        """)
