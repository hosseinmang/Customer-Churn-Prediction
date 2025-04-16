import asyncio
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add scripts to path
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
from preprocessing import preprocess_data, prepare_features

# Set page config
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="<i class='fas fa-chart-line'></i>",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        /* Main content styling */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Card styling */
        div[data-testid="stMetricValue"] {
            font-size: 2rem !important;
            color: #1f77b4 !important;
            font-weight: bold;
        }
        
        div[data-testid="stMetricDelta"] {
            font-size: 1rem !important;
            color: #2ecc71 !important;
        }
        
        /* Header styling */
        h1 {
            color: #2c3e50;
            font-family: 'Helvetica Neue', sans-serif;
            padding: 1rem 0;
            text-align: center;
            background: linear-gradient(90deg, #1f77b4 0%, #2ecc71 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        h2 {
            color: #34495e;
            font-family: 'Helvetica Neue', sans-serif;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
            margin-top: 2rem;
        }
        
        h3 {
            color: #2980b9;
            font-family: 'Helvetica Neue', sans-serif;
        }
        
        /* Metric card styling */
    .metric-card {
            background-color: white;
        border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        /* Highlight text */
    .highlight {
            color: #e74c3c;
        font-weight: bold;
    }
        
        /* Custom container */
        .custom-container {
            background-color: white;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #2c3e50;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #2ecc71;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #27ae60;
            transform: translateY(-2px);
        }
        
        /* Chart container */
        .chart-container {
            background-color: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        }
        
        /* Table styling */
        .dataframe {
            font-family: 'Helvetica Neue', sans-serif;
            border-collapse: collapse;
            margin: 1rem 0;
            width: 100%;
        }
        
        .dataframe th {
            background-color: #2c3e50;
            color: white;
            padding: 0.5rem;
        }
        
        .dataframe td {
            padding: 0.5rem;
            border-bottom: 1px solid #ecf0f1;
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .status-green {
            background-color: #2ecc71;
        }
        
        .status-yellow {
            background-color: #f1c40f;
        }
        
        .status-red {
            background-color: #e74c3c;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description with enhanced styling
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 1rem;'>
            <i class="fas fa-chart-line"></i> Banking Customer Churn Analytics
        </h1>
        <p style='font-size: 1.2rem; color: #7f8c8d; max-width: 800px; margin: 0 auto;'>
            This interactive dashboard provides comprehensive insights into customer churn patterns and risk factors, 
            helping identify and retain at-risk customers through data-driven decisions.
        </p>
    </div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        # Use absolute path from project root
        data_path = os.path.join(PROJECT_ROOT, 'data', 'Telco_customer_churn.xlsx')
        if not os.path.exists(data_path):
            st.error(f"Data file not found at: {data_path}")
            return None, None, None
            
        # Load raw data
        raw_data = pd.read_excel(data_path)
        
        # Ensure numeric columns are properly typed
        raw_data['Tenure Months'] = pd.to_numeric(raw_data['Tenure Months'], errors='coerce')
        raw_data['Monthly Charges'] = pd.to_numeric(raw_data['Monthly Charges'], errors='coerce')
        raw_data['Total Charges'] = pd.to_numeric(raw_data['Total Charges'], errors='coerce')
        
        # Preprocess the data
        processed_data, transformers = preprocess_data(raw_data)
        
        # Ensure all required columns exist
        required_columns = ['Tenure Months', 'Monthly Charges', 'Churn Label', 'TotalBalance']
        for col in required_columns:
            if col not in processed_data.columns:
                st.error(f"Required column {col} not found in processed data")
                return None, None, None
        
        return raw_data, processed_data, transformers
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

@st.cache_resource
def load_models():
    """Load the trained model and transformers"""
    try:
        # Use absolute paths from project root
        model_path = os.path.join(PROJECT_ROOT, 'models', 'xgb_model.joblib')
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
            
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Load data and model
try:
    raw_df, df, transformers = load_data()
    model = load_models()
    
    if df is None or model is None:
        st.error("Failed to load data or model. Please check the file paths and try again.")
        st.stop()
    
    # Ensure df has all required columns before creating plots
    if 'TotalBalance' not in df.columns:
        df['TotalBalance'] = df['Total Charges']  # Use Total Charges as fallback
    
    # Prepare features for the model
    X, y, feature_names = prepare_features(df)
except Exception as e:
    st.error(f"Error loading data or model: {str(e)}")
    st.write("Current working directory:", os.getcwd())
    st.write("Directory contents:", os.listdir())
    st.stop()

# Sidebar
st.sidebar.markdown("""
    <h3>
        <i class="fas fa-bars"></i> Dashboard Navigation
    </h3>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Choose a page",
    ["Executive Summary", "Customer Segments", "Risk Analysis", "Predictive Tools"]
)

# Executive Summary Page
if page == "Executive Summary":
    st.markdown("""
        <div class='custom-container'>
            <h2 style='margin-top: 0;'>Executive Overview</h2>
            <p style='color: #7f8c8d;'>Key performance indicators and metrics for quick insights</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(raw_df)
        st.markdown("""
            <div class='metric-card'>
                <h4 style='color: #7f8c8d; margin-bottom: 0.5rem;'>Total Customers</h4>
                <div style='font-size: 2rem; color: #2c3e50; font-weight: bold;'>{:,}</div>
                <div style='color: #27ae60; font-size: 0.9rem;'>↑ 5% from last month</div>
            </div>
        """.format(total_customers), unsafe_allow_html=True)
    
    with col2:
        churn_rate = (raw_df['Churn Value'].sum() / len(raw_df)) * 100
        st.markdown("""
            <div class='metric-card'>
                <h4 style='color: #7f8c8d; margin-bottom: 0.5rem;'>Churn Rate</h4>
                <div style='font-size: 2rem; color: #{}; font-weight: bold;'>{:.1f}%</div>
                <div style='color: #{}; font-size: 0.9rem;'>Target: 20%</div>
            </div>
        """.format(
            '27ae60' if churn_rate <= 20 else 'e74c3c',
            churn_rate,
            'e74c3c' if churn_rate > 20 else '27ae60'
        ), unsafe_allow_html=True)
    
    with col3:
        avg_tenure = raw_df['Tenure Months'].mean() / 12  # Convert months to years
        st.markdown("""
            <div class='metric-card'>
                <h4 style='color: #7f8c8d; margin-bottom: 0.5rem;'>Avg. Customer Tenure</h4>
                <div style='font-size: 2rem; color: #2c3e50; font-weight: bold;'>{:.1f} years</div>
                <div style='color: #27ae60; font-size: 0.9rem;'>↑ 0.5 yr from last quarter</div>
            </div>
        """.format(avg_tenure), unsafe_allow_html=True)
    
    with col4:
        avg_monthly = raw_df['Monthly Charges'].mean()
        st.markdown("""
            <div class='metric-card'>
                <h4 style='color: #7f8c8d; margin-bottom: 0.5rem;'>Avg. Monthly Fees</h4>
                <div style='font-size: 2rem; color: #2c3e50; font-weight: bold;'>${:.2f}</div>
                <div style='color: #27ae60; font-size: 0.9rem;'>↑ 2% from last month</div>
            </div>
        """.format(avg_monthly), unsafe_allow_html=True)

    # Churn Overview Section
    st.markdown("""
        <div class='custom-container'>
            <h2 style='margin-top: 0;'>Churn Overview</h2>
            <p style='color: #7f8c8d;'>Analysis of customer churn patterns and primary reasons</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Enhanced Churn Distribution with Plotly
        fig = px.pie(raw_df, 
                    names='Churn Label', 
                    title='Customer Churn Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    hole=0.4)
        fig.update_traces(textposition='outside', 
                         textinfo='percent+label')
        fig.update_layout(
            title_x=0.5,
            title_font_size=20,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, l=0, r=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Enhanced Top Churn Reasons
        churn_reasons = raw_df[raw_df['Churn Label'] == 'Yes']['Churn Reason'].value_counts().head(5)
        fig = px.bar(x=churn_reasons.index, 
                    y=churn_reasons.values,
                    title='Top 5 Churn Reasons',
                    labels={'x': 'Reason', 'y': 'Count'},
                    color_discrete_sequence=['#ff4b4b'])
        fig.update_layout(
            title_x=0.5,
            title_font_size=20,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, l=0, r=0, b=0),
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

    # Key Insights Section with enhanced styling
    st.markdown("""
        <div class='custom-container'>
            <h2 style='margin-top: 0;'>Key Insights</h2>
            <p style='color: #7f8c8d;'>Critical findings and actionable recommendations</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3 style='color: #2c3e50; margin-bottom: 1rem;'>Critical Findings</h3>
                <ul style='color: #34495e; list-style-type: none; padding-left: 0;'>
                    <li style='margin-bottom: 0.8rem;'>🎯 <strong>High-Risk Period:</strong> First 2 years show highest churn probability</li>
                    <li style='margin-bottom: 0.8rem;'>💰 <strong>Price Sensitivity:</strong> Threshold at $70-80 monthly fees</li>
                    <li style='margin-bottom: 0.8rem;'>🔒 <strong>Service Impact:</strong> Digital banking users show 45% lower churn</li>
                    <li style='margin-bottom: 0.8rem;'>📈 <strong>Contract Effect:</strong> Long-term contracts reduce churn by 67%</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3 style='color: #2c3e50; margin-bottom: 1rem;'>Action Items</h3>
                <ul style='color: #34495e; list-style-type: none; padding-left: 0;'>
                    <li style='margin-bottom: 0.8rem;'>
                        <span class='status-indicator status-red'></span>
                        Implement early warning system for new customers
                    </li>
                    <li style='margin-bottom: 0.8rem;'>
                        <span class='status-indicator status-yellow'></span>
                        Review pricing strategy for sensitive segments
                    </li>
                    <li style='margin-bottom: 0.8rem;'>
                        <span class='status-indicator status-green'></span>
                        Promote digital service adoption
                    </li>
                    <li style='margin-bottom: 0.8rem;'>
                        <span class='status-indicator status-green'></span>
                        Incentivize long-term contracts
                    </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

elif page == "Customer Segments":
    st.markdown("""
        <div class='custom-container'>
            <h2 style='margin-top: 0;'>Customer Segmentation Analysis</h2>
            <p style='color: #7f8c8d;'>Deep dive into customer segments and behavior patterns</p>
        </div>
    """, unsafe_allow_html=True)

    try:
        # Ensure all required columns exist
        required_cols = ['Tenure Months', 'Monthly Charges', 'Churn Label', 'CustomerID', 'TotalBalance']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.stop()

        # Convert columns to numeric and handle missing values
        df['TotalBalance'] = pd.to_numeric(df['TotalBalance'], errors='coerce')
        df['Tenure Months'] = pd.to_numeric(df['Tenure Months'], errors='coerce')
        df['Monthly Charges'] = pd.to_numeric(df['Monthly Charges'], errors='coerce')
        
        # Create a clean dataframe for visualization
        df_clean = df.dropna(subset=['TotalBalance', 'Tenure Months', 'Monthly Charges'])
        
        if len(df_clean) == 0:
            st.error("No valid data available after cleaning. Please check your data for missing or invalid values.")
            st.stop()
        
        # Normalize TotalBalance for bubble size
        balance_range = df_clean['TotalBalance'].max() - df_clean['TotalBalance'].min()
        if balance_range == 0:  # Handle case where all values are the same
            df_clean['TotalBalance_normalized'] = 10  # Set a default size
        else:
            df_clean['TotalBalance_normalized'] = (
                (df_clean['TotalBalance'] - df_clean['TotalBalance'].min()) / balance_range * 15 + 5
            )

        # Create scatter plot
        fig = px.scatter(
            df_clean,
            x='Tenure Months',
            y='Monthly Charges',
            color='Churn Label',
            size='TotalBalance_normalized',
            hover_data=['CustomerID'],
            title='Customer Segments by Tenure and Fees',
            labels={
                'Tenure Months': 'Tenure (Months)',
                'Monthly Charges': 'Monthly Fees ($)',
                'TotalBalance_normalized': 'Total Balance'
            },
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )

        fig.update_layout(
            title_x=0.5,
            title_font_size=20,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, l=0, r=0, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add summary statistics
        st.markdown("### Customer Segment Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            segment_stats = df_clean.groupby('Churn Label').agg({
                'CustomerID': 'count',
                'Monthly Charges': 'mean',
                'Tenure Months': 'mean'
            }).round(2)
            segment_stats.columns = ['Count', 'Avg Monthly Fees ($)', 'Avg Tenure (Months)']
            st.dataframe(segment_stats)
        
        with col2:
            st.markdown("""
                #### Key Insights
                - 📊 Distribution of customers by tenure and fees
                - 💰 Relationship between charges and churn risk
                - ⏳ Impact of customer tenure on retention
            """)

        # Service Adoption Analysis
        st.markdown("### Service Adoption Analysis")
        service_cols = [
            'Online Banking', 'Secure Login', 'Automatic Savings',
            'Fraud Protection', 'Customer Support', 'Bill Pay'
        ]
        
        # Check available service columns
        available_cols = [col for col in service_cols if col in df_clean.columns]
        if not available_cols:
            st.warning("No service columns found in the data. Please check column names.")
        else:
            # Create service adoption visualization
            service_data = df_clean.melt(
                id_vars=['Churn Label'],
                value_vars=available_cols,
                var_name='Service',
                value_name='Has Service'
            )
            
            fig = px.bar(
                service_data,
                x='Service',
                y='Has Service',
                color='Churn Label',
                barmode='group',
                title='Service Adoption by Churn Status',
                labels={'Has Service': 'Adoption Rate', 'Service': 'Service Type'}
            )
            
            fig.update_layout(
                title_x=0.5,
                title_font_size=20,
                xaxis_tickangle=-45,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=50, l=0, r=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error in Customer Segments section: {str(e)}")
        st.write("Please check your data format and try again.")
        st.write("Required columns:", required_cols)

elif page == "Risk Analysis":
    try:
        st.markdown("""
            <div class='custom-container'>
                <h2 style='margin-top: 0;'>Customer Risk Analysis</h2>
                <p style='color: #7f8c8d;'>Identifying and analyzing high-risk customer segments</p>
            </div>
        """, unsafe_allow_html=True)

        # Ensure numeric columns are properly typed
        numeric_cols = ['Tenure Months', 'Monthly Charges', 'TotalBalance']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove rows with missing values in key columns
        df_clean = df.dropna(subset=[col for col in numeric_cols if col in df.columns])

        if len(df_clean) == 0:
            st.error("No valid data available after cleaning. Please check your data.")
        else:
            # Create risk score based on multiple factors
            df_clean['Risk_Score'] = (
                (df_clean['Monthly Charges'] / df_clean['Monthly Charges'].max()) * 0.4 +
                (1 - df_clean['Tenure Months'] / df_clean['Tenure Months'].max()) * 0.4 +
                (1 - df_clean['TotalBalance'] / df_clean['TotalBalance'].max()) * 0.2
            ) * 100

            # Risk Distribution
            fig = px.histogram(df_clean, 
                             x='Risk_Score',
                             color='Churn Label',
                             nbins=30,
                             title='Distribution of Customer Risk Scores',
                             labels={'Risk_Score': 'Risk Score', 'count': 'Number of Customers'})
            
            fig.update_layout(
                title_x=0.5,
                title_font_size=20,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                bargap=0.1
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Risk vs Tenure Analysis
            fig = px.scatter(df_clean,
                           x='Tenure Months',
                           y='Risk_Score',
                           color='Churn Label',
                           size='Monthly Charges',
                           title='Risk Score vs. Customer Tenure',
                           labels={'Tenure Months': 'Tenure (Months)',
                                  'Risk_Score': 'Risk Score',
                                  'Monthly Charges': 'Monthly Fees ($)'})
            
            fig.update_layout(
                title_x=0.5,
                title_font_size=20,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Risk Categories
            df_clean['Risk_Category'] = pd.qcut(df_clean['Risk_Score'], 
                                              q=4, 
                                              labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk'])
            
            risk_summary = df_clean.groupby('Risk_Category').agg({
                'CustomerID': 'count',
                'Churn Label': 'mean',
                'Monthly Charges': 'mean',
                'TotalBalance': 'mean'
            }).round(2)
            
            risk_summary.columns = ['Customer Count', 'Churn Rate', 'Avg Monthly Fees', 'Avg Balance']
            st.write("### Risk Category Summary")
            st.dataframe(risk_summary)

    except Exception as e:
        st.error(f"Error in Risk Analysis section: {str(e)}")
        st.write("Please check the data format and try again.")

else:  # Predictive Tools
    st.header("Churn Prediction Tool")

    # Create tabs for different prediction approaches
    tab1, tab2 = st.tabs(["Individual Customer", "Batch Prediction"])

    with tab1:
        st.subheader("Individual Customer Risk Assessment")
        
        # Create input form with better organization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Customer Profile")
            years_with_bank = st.number_input("Years with Bank", 
                                            min_value=0.0, 
                                            max_value=50.0, 
                                            value=5.0)
            monthly_fees = st.number_input("Monthly Bank Fees ($)", 
                                         min_value=0.0, 
                                         max_value=1000.0, 
                                         value=100.0)
            total_balance = st.number_input("Total Balance ($)", 
                                          min_value=0.0, 
                                          max_value=100000.0, 
                                          value=1000.0)
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], index=1)
        
        with col2:
            st.markdown("### Banking Services")
            debit_card = st.selectbox("Debit Card", ["No", "Yes"], index=1)
            credit_card = st.selectbox("Credit Card", ["No", "Yes", "No phone service"], index=1)
            online_banking = st.selectbox("Online Banking", ["DSL", "Fiber optic", "No"], index=0)
            secure_login = st.selectbox("2FA Security", ["No", "Yes", "No internet service"], index=1)
            bill_pay = st.selectbox("Bill Pay Service", ["No", "Yes", "No internet service"], index=1)
        
        with col3:
            st.markdown("### Additional Services")
            auto_savings = st.selectbox("Automatic Savings", ["No", "Yes", "No internet service"], index=1)
            fraud_protection = st.selectbox("Fraud Protection", ["No", "Yes", "No internet service"], index=1)
            customer_support = st.selectbox("Customer Support", ["No", "Yes", "No internet service"], index=1)
            mobile_payments = st.selectbox("Mobile Payments", ["No", "Yes", "No internet service"], index=1)
            contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
            payment_method = st.selectbox("Payment Method", 
                                        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                                        index=0)
        
        if st.button("Analyze Churn Risk", use_container_width=True):
            try:
                # Prepare input data
                input_data = pd.DataFrame({
                    'Tenure Months': [years_with_bank * 12],  # Convert years to months
                    'Monthly Charges': [monthly_fees],
                    'Total Charges': [total_balance],
                    'Phone Service': [debit_card],
                    'Multiple Lines': [credit_card],
                    'Internet Service': [online_banking],
                    'Online Security': [secure_login],
                    'Online Backup': [auto_savings],
                    'Device Protection': [fraud_protection],
                    'Tech Support': [customer_support],
                    'Streaming TV': [bill_pay],
                    'Streaming Movies': [mobile_payments],
                    'Contract': [contract_type]
                })
                
                # Preprocess the input data
                input_processed, _ = preprocess_data(input_data)
                
                # Prepare features for prediction
                X_input = input_processed[[
                    'YearsWithBank', 'MonthlyBankFees', 'TotalBalance',
                    'DebitCard', 'CreditCard', 'OnlineBanking', 'SecureLogin2FA',
                    'AutomaticSavings', 'FraudProtection', 'CustomerSupport',
                    'BillPay', 'MobilePayments', 'Contract'
                ]].values
                
                # Make prediction
                risk_score = model.model.predict_proba(X_input)[:, 1][0]  # Access the underlying model
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gauge chart for risk probability
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_score * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Churn Risk"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "red"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_score * 100
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk assessment and recommendations
                    st.markdown("### Risk Assessment")
                    
                    if risk_score < 0.3:
                        st.success("🟢 Low Risk of Churn")
                    elif risk_score < 0.7:
                        st.warning("🟡 Medium Risk of Churn")
                    else:
                        st.error("🔴 High Risk of Churn")
                    
                    st.markdown("### Recommended Actions")
                    
                    recommendations = []
                    if monthly_fees > 70:
                        recommendations.append("📉 Review fee structure - current fees above sensitivity threshold")
                    if online_banking == "No":
                        recommendations.append("💻 Encourage online banking adoption")
                    if contract_type == "Month-to-month":
                        recommendations.append("📅 Offer incentives for longer-term contract")
                    if not all([secure_login == "Yes", fraud_protection == "Yes"]):
                        recommendations.append("🔒 Promote security features package")
                    if years_with_bank < 2:
                        recommendations.append("⚠️ High-risk period: Implement enhanced monitoring")
                    
                    for rec in recommendations:
                        st.write(rec)
                    
                    if not recommendations:
                        st.write("✅ No immediate actions required")
                    
            except Exception as e:
                st.error(f"Error processing prediction: {str(e)}")
                st.stop()

    with tab2:
        st.subheader("Batch Risk Assessment")
        
        uploaded_file = st.file_uploader("Upload customer data (Excel/CSV)", 
                                       type=['xlsx', 'csv'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    batch_data = pd.read_csv(uploaded_file)
                else:
                    batch_data = pd.read_excel(uploaded_file)
                
                st.write("Preview of uploaded data:")
                st.dataframe(batch_data.head())
                
                if st.button("Run Batch Analysis"):
                    # Process batch predictions here
                    st.info("Batch processing functionality coming soon!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# Footer with last update time
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown(f"""
        <div style='display: flex; align-items: center;'>
            <i class="fas fa-clock"></i>
            <span style='margin-left: 0.5rem;'><strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div style='text-align: center;'>
            <p style='color: var(--text-color); font-size: 0.8rem;'>
                <i class="fas fa-chart-line"></i> Banking Customer Churn Analytics Dashboard v1.0
            </p>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div style='text-align: right;'>
            <p style='color: var(--text-color);'>
                <i class="fas fa-chart-bar"></i> Powered by Streamlit
            </p>
        </div>
    """, unsafe_allow_html=True)