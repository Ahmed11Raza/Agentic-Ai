from db.database import Base, engine
Base.metadata.create_all(bind=engine)

import streamlit as st
from auth.login import login
from auth.register import register
from core.payment import Payment
from core.meal import MealLogger
from core.report import HealthReport
from core.diet_plan import DietPlan

st.set_page_config(page_title="NutriTrackAI", page_icon="🥗")

if "user" not in st.session_state:
    option = st.sidebar.radio("Choose:", ["Login", "Register"])
    if option == "Login":
        login()
    else:
        register()
else:
    st.sidebar.write(f"Logged in as: {st.session_state.user}")
    st.sidebar.button("Logout", on_click=lambda: st.session_state.clear())

    st.title("NutriTrackAI Dashboard")
    st.write("Welcome to your personal AI Nutrition Tracker.")

    if st.button("Upgrade to Premium"):
        try:
            payment = Payment(amount=9.99, email="test@example.com")
            st.markdown(f"[Click here to Pay]({payment.create_checkout()})", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Payment error: {e}")

    st.subheader("Log a Meal")
    name = st.text_input("Meal Name")
    cal = st.number_input("Calories", 0.0)
    pro = st.number_input("Protein (g)", 0.0)
    carbs = st.number_input("Carbs (g)", 0.0)
    fats = st.number_input("Fats (g)", 0.0)

    if st.button("Save Meal"):
        try:
            MealLogger(st.session_state.user).log_meal(name, cal, pro, carbs, fats)
            st.success("Meal logged!")
        except Exception as e:
            st.error(f"Error logging meal: {e}")

    st.subheader("Today's Summary")
    try:
        report = HealthReport(st.session_state.user).generate()
        st.json(report)
    except Exception as e:
        st.error(f"Error generating report: {e}")

    st.subheader("Suggested Diet Plan")
    goal = st.selectbox("Goal", ["maintain", "lose", "gain"])
    if st.button("Generate Plan"):
        try:
            plan = DietPlan(goal).generate_plan()
            for item in plan:
                st.write(f"{item['meal']} - {item['calories']} kcal")
        except Exception as e:
            st.error(f"Error generating plan: {e}")