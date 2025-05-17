import streamlit as st
from db.database import SessionLocal
from db.models import User
from auth.auth_utils import verify_password

def login():
    db = SessionLocal()
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = db.query(User).filter_by(username=username).first()
        if user and verify_password(password, user.hashed_password):
            st.session_state.user = user.username
            st.success(f"Welcome back, {user.username}")
            return True
        else:
            st.error("Invalid credentials")
            return False
