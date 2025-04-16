import streamlit as st

# Set page config
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
)

# Title and description
st.title("Banking Customer Churn Analytics")
st.write("Welcome to the Customer Churn Analytics Dashboard")

# Maintenance notice
st.info("🔄 Dashboard is currently being updated with new features. Please check back soon for the full version.")

# Basic navigation
option = st.selectbox(
    "Choose a section",
    ["Overview", "Customer Segments", "Risk Analysis", "Predictive Tools"]
)

st.header(option)
st.write("This section is being updated with enhanced features and visualizations.") 