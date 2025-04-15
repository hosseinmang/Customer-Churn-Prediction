import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set page config
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Customer Churn Analytics Dashboard")

# Load data
@st.cache_data
def load_data():
    try:
        # Get the absolute path to the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_path = os.path.join(project_root, 'data', 'Telco_customer_churn.xlsx')
        
        if not os.path.exists(data_path):
            st.error(f"Data file not found at: {data_path}")
            return None
            
        # Load data
        df = pd.read_excel(data_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load the data
df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.header("Dashboard Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Customer Analysis", "Churn Prediction"]
    )

    # Overview Page
    if page == "Overview":
        st.header("Overview")
        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Customers", len(df))
        
        with col2:
            churn_rate = (df['Churn Label'].value_counts(normalize=True).get('Yes', 0) * 100)
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        with col3:
            avg_tenure = df['Tenure Months'].mean()
            st.metric("Avg. Tenure (months)", f"{avg_tenure:.1f}")
        
        # Add visualization section
        st.subheader("Churn Analysis")
        
        # Churn Distribution
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            # Churn by Contract Type
            contract_churn = pd.crosstab(df['Contract'], df['Churn Label'], normalize='index') * 100
            st.bar_chart(contract_churn['Yes'])
            st.write("Churn Rate by Contract Type (%)")
        
        with fig_col2:
            # Monthly Charges Distribution
            fig, ax = plt.subplots()
            sns.boxplot(x='Churn Label', y='Monthly Charges', data=df)
            st.pyplot(fig)
            st.write("Monthly Charges Distribution by Churn Status")

    # Customer Analysis Page
    elif page == "Customer Analysis":
        st.header("Customer Analysis")
        
        # Customer Segments
        st.subheader("Customer Segments")
        
        # Tenure Groups
        df['Tenure Group'] = pd.qcut(df['Tenure Months'], 
                                   q=4, 
                                   labels=['0-25%', '25-50%', '50-75%', '75-100%'])
        
        # Monthly Charges Groups
        df['Monthly Charges Group'] = pd.qcut(df['Monthly Charges'], 
                                            q=4, 
                                            labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Display segment analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Tenure analysis
            tenure_churn = pd.crosstab(df['Tenure Group'], 
                                     df['Churn Label'], 
                                     normalize='index') * 100
            st.bar_chart(tenure_churn['Yes'])
            st.write("Churn Rate by Tenure Group (%)")
        
        with col2:
            # Monthly charges analysis
            charges_churn = pd.crosstab(df['Monthly Charges Group'], 
                                      df['Churn Label'], 
                                      normalize='index') * 100
            st.bar_chart(charges_churn['Yes'])
            st.write("Churn Rate by Monthly Charges Group (%)")

    # Churn Prediction Page
    else:
        st.header("Churn Prediction")
        
        # Add customer input form
        st.subheader("Customer Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=1000.0, value=50.0)
            contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
        
        with col2:
            internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
            online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
            payment_method = st.selectbox("Payment Method", 
                                        ['Electronic check', 'Mailed check', 
                                         'Bank transfer (automatic)', 
                                         'Credit card (automatic)'])
        
        if st.button("Predict Churn Risk"):
            st.info("Churn prediction functionality coming soon!")
else:
    st.error("Failed to load data. Please check the data file and try again.") 