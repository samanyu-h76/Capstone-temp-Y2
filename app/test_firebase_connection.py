import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

st.title("🔥 Firebase Connection Test")

# Check if credentials exist
if "FIREBASE_CREDENTIALS" not in st.secrets:
    st.error("❌ FIREBASE_CREDENTIALS not found in secrets!")
    st.info("""
    Make sure you added this to your secrets:
    
    [FIREBASE_CREDENTIALS]
    type = "service_account"
    project_id = "y2-capstone--culture-tourism"
    private_key_id = "..."
    private_key = '''
    -----BEGIN PRIVATE KEY-----
    ...
    -----END PRIVATE KEY-----
    '''
    client_email = "..."
    # ... rest of the fields
    """)
    st.stop()

st.success("✅ Firebase credentials found in secrets")

# Try to initialize
try:
    # Check if already initialized
    if not firebase_admin._apps:
        firebase_creds = dict(st.secrets["FIREBASE_CREDENTIALS"])
        
        st.write("**Credentials loaded:**")
        st.write(f"- Project ID: {firebase_creds.get('project_id', 'Missing')}")
        st.write(f"- Client Email: {firebase_creds.get('client_email', 'Missing')}")
        st.write(f"- Private Key: {'Present ✓' if firebase_creds.get('private_key') else 'Missing ✗'}")
        
        # Initialize
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
        
        st.success("✅ Firebase initialized successfully!")
    else:
        st.info("Firebase already initialized")
    
    # Get Firestore client
    db = firestore.client()
    st.success("✅ Firestore client created!")
    
    # Test write
    if st.button("Test Write to Firestore"):
        with st.spinner("Writing test document..."):
            doc_ref = db.collection("connection_test").add({
                "message": "Hello from Streamlit!",
                "timestamp": firestore.SERVER_TIMESTAMP,
                "test": True
            })
            st.success(f"✅ Test document created! ID: {doc_ref[1].id}")
    
    # Test read
    if st.button("Test Read from Firestore"):
        with st.spinner("Reading test documents..."):
            docs = db.collection("connection_test").limit(5).stream()
            
            found = False
            for doc in docs:
                found = True
                st.write(f"**Document ID:** {doc.id}")
                st.json(doc.to_dict())
            
            if not found:
                st.info("No test documents found. Try 'Test Write' first.")
            else:
                st.success("✅ Successfully read from Firestore!")
    
    st.markdown("---")
    st.success("🎉 All tests passed! Firebase is ready to use!")
    
except Exception as e:
    st.error(f"❌ Firebase initialization failed!")
    st.error(f"**Error:** {str(e)}")
    
    st.info("""
    **Common fixes:**
    
    1. **Check your private_key format:**
       - Should use triple quotes: '''...'''
       - Should include -----BEGIN PRIVATE KEY-----
       - Should include -----END PRIVATE KEY-----
    
    2. **Verify all fields are present:**
       - type, project_id, private_key_id
       - private_key, client_email, client_id
       - auth_uri, token_uri, etc.
    
    3. **Check Firestore is enabled:**
       - Go to Firebase Console
       - Enable Cloud Firestore
       - Set up basic rules
    """)
