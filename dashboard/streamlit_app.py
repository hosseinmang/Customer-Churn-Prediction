import streamlit as st
import pandas as pd
import os
import sys

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add scripts to path
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))

# Set page config
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Banking Customer Churn Analytics")
st.markdown("""
    This dashboard provides insights into customer churn patterns and risk factors, 
    helping identify and retain at-risk customers through data-driven decisions.
""")

# Maintenance notice
st.info("🔄 Dashboard is currently being updated with new features. Please check back soon for the full version.")

# Basic structure
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a section",
    ["Overview", "Customer Segments", "Risk Analysis", "Predictive Tools"]
)

# Display maintenance message for each section
st.header(page)
st.write("This section is being updated with enhanced features and visualizations.")

# Footer
st.markdown("---")
st.markdown("Dashboard Version: 1.0 (Maintenance Mode)") 