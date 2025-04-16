import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set page config
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Banking Customer Churn Analytics")
st.write("Interactive dashboard for customer churn analysis and prediction")

# Data loading function
@st.cache_data
def load_data():
    try:
        data_path = os.path.join('data', 'Telco_customer_churn.xlsx')
        df = pd.read_excel(data_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a section",
    ["Overview", "Customer Segments", "Risk Analysis", "Predictive Tools"]
)

if df is not None:
    if page == "Overview":
        st.header("Overview")
        
        # Key metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_customers = len(df)
            st.metric("Total Customers", f"{total_customers:,}")
            
        with col2:
            churn_rate = (df['Churn Value'].mean() * 100).round(2)
            st.metric("Churn Rate", f"{churn_rate}%")
            
        with col3:
            avg_tenure = df['Tenure Months'].mean().round(2)
            st.metric("Avg. Tenure (Months)", f"{avg_tenure}")
        
        # Churn Distribution
        st.subheader("Churn Distribution")
        fig = px.pie(df, names='Churn Label', title='Customer Churn Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
    elif page == "Customer Segments":
        st.header("Customer Segments")
        
        # Tenure vs Monthly Charges scatter plot
        st.subheader("Customer Distribution")
        fig = px.scatter(df, 
                        x='Tenure Months',
                        y='Monthly Charges',
                        color='Churn Label',
                        title='Customer Distribution by Tenure and Monthly Charges')
        st.plotly_chart(fig, use_container_width=True)
        
    elif page == "Risk Analysis":
        st.header("Risk Analysis")
        
        # Contract Type Analysis
        st.subheader("Churn by Contract Type")
        contract_churn = df.groupby('Contract')['Churn Value'].mean().reset_index()
        fig = px.bar(contract_churn,
                    x='Contract',
                    y='Churn Value',
                    title='Churn Rate by Contract Type')
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Predictive Tools
        st.header("Predictive Tools")
        st.info("🔄 Predictive features are being updated. Please check back soon.")

else:
    st.error("Unable to load data. Please check the data file and try again.")

# Footer
st.markdown("---")
st.markdown("Dashboard Version: 2.0") 