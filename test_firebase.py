#!/usr/bin/env python3
"""
Simple Firebase Authentication Test Script
Run this to verify Firebase is working before trying the full app
"""

import requests
import json
import sys

def test_firebase_auth():
    """Test Firebase REST API"""
    
    # Check if secrets file exists
    try:
        import streamlit as st
        config = st.secrets
        
        firebase_api_key = config.get("FIREBASE_API_KEY")
        firebase_project_id = config.get("FIREBASE_PROJECT_ID")
        
        if not firebase_api_key:
            print("❌ FIREBASE_API_KEY not found in .streamlit/secrets.toml")
            print("\nTo fix:")
            print("1. Go to https://console.firebase.google.com")
            print("2. Select your project (tourism-recommendation-engine)")
            print("3. Go to Settings > Project Settings > Web API Key")
            print("4. Copy the API Key")
            print("5. Add to .streamlit/secrets.toml:")
            print('   FIREBASE_API_KEY = "paste-your-key-here"')
            return False
        
        if not firebase_project_id:
            print("❌ FIREBASE_PROJECT_ID not found in .streamlit/secrets.toml")
            print("Add: FIREBASE_PROJECT_ID = \"tourism-recommendation-engine\"")
            return False
        
        print(f"✅ Found FIREBASE_API_KEY: {firebase_api_key[:30]}...")
        print(f"✅ Found FIREBASE_PROJECT_ID: {firebase_project_id}")
        
        # Test signup
        print("\n🧪 Testing Firebase signup...")
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={firebase_api_key}"
        
        payload = {
            "email": "test@example.com",
            "password": "TestPassword123",
            "returnSecureToken": True
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            print("✅ Firebase signup API is working!")
            data = response.json()
            print(f"   User ID: {data.get('localId')}")
            return True
        else:
            print(f"❌ Firebase API returned status {response.status_code}")
            error_data = response.json()
            print(f"   Error: {error_data}")
            
            if response.status_code == 400:
                error_msg = error_data.get('error', {}).get('message', '')
                if 'EMAIL_EXISTS' in error_msg:
                    print("   (Email already exists - Firebase is working!)")
                    return True
                elif 'OPERATION_NOT_ALLOWED' in error_msg:
                    print("   Email/Password auth is disabled in Firebase Console")
                    return False
            
            return False
    
    except ImportError:
        print("❌ Streamlit not installed")
        print("Run: pip install streamlit")
        return False
    except FileNotFoundError:
        print("❌ .streamlit/secrets.toml file not found")
        print("Create .streamlit/secrets.toml with:")
        print('FIREBASE_API_KEY = "your-api-key-here"')
        print('FIREBASE_PROJECT_ID = "tourism-recommendation-engine"')
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Firebase Authentication Test\n" + "="*50)
    success = test_firebase_auth()
    
    if success:
        print("\n✅ Firebase is properly configured!")
        print("You can now run: streamlit run app.py")
        sys.exit(0)
    else:
        print("\n❌ Firebase is not properly configured")
        print("Please follow the instructions above and try again")
        sys.exit(1)
