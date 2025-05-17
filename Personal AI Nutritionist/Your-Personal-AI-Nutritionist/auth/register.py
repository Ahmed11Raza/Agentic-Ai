import streamlit as st
from db.database import SessionLocal
from db.models import User
from auth.auth_utils import hash_password

def register():
    db = SessionLocal()
    st.subheader("Register")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if db.query(User).filter_by(username=username).first():
            st.error("Username already exists.")
            return
        user = User(username=username, email=email, hashed_password=hash_password(password))
        db.add(user)
        db.commit()
        st.success("Registered! Please log in.")
