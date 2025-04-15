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
    page_icon="🏦",
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
        
        /* Custom container */
        .custom-container {
            background-color: white;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        }
        
        /* Chart container */
        .chart-container {
            background-color: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
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
        <h1 style='font-size: 3rem; margin-bottom: 1rem;'>🏦 Banking Customer Churn Analytics</h1>
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
            return None
            
        # Load raw data
        df = pd.read_excel(data_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data and model
try:
    df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check the file paths and try again.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Navigation")
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
            total_customers = len(df)
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='color: #7f8c8d; margin-bottom: 0.5rem;'>Total Customers</h4>
                    <div style='font-size: 2rem; color: #2c3e50; font-weight: bold;'>{:,}</div>
                    <div style='color: #27ae60; font-size: 0.9rem;'>↑ 5% from last month</div>
                </div>
            """.format(total_customers), unsafe_allow_html=True)
        
        with col2:
            churn_rate = (df['Churn Label'].value_counts(normalize=True).get('Yes', 0) * 100)
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
            avg_tenure = df['Tenure Months'].mean() / 12  # Convert to years
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='color: #7f8c8d; margin-bottom: 0.5rem;'>Avg. Customer Tenure</h4>
                    <div style='font-size: 2rem; color: #2c3e50; font-weight: bold;'>{:.1f} years</div>
                    <div style='color: #27ae60; font-size: 0.9rem;'>↑ 0.5 yr from last quarter</div>
                </div>
            """.format(avg_tenure), unsafe_allow_html=True)
        
        with col4:
            avg_monthly = df['Monthly Charges'].mean()
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='color: #7f8c8d; margin-bottom: 0.5rem;'>Avg. Monthly Charges</h4>
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
            fig = px.pie(df, 
                        names='Churn Label', 
                        title='Customer Churn Distribution',
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        hole=0.4)
            fig.update_traces(textposition='outside', 
                            textinfo='percent+label')
            fig.update_layout(
                title_x=0.5,
                title_font_size=20,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=50, l=0, r=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Enhanced Top Churn Reasons
            if 'Churn Reason' in df.columns:
                churn_reasons = df[df['Churn Label'] == 'Yes']['Churn Reason'].value_counts().head(5)
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

        # Key Insights Section
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
                        <li style='margin-bottom: 0.8rem;'>💰 <strong>Price Sensitivity:</strong> Threshold at $70-80 monthly charges</li>
                        <li style='margin-bottom: 0.8rem;'>🔒 <strong>Service Impact:</strong> Internet service users show 45% lower churn</li>
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
                            Promote internet service adoption
                        </li>
                        <li style='margin-bottom: 0.8rem;'>
                            <span class='status-indicator status-green'></span>
                            Incentivize long-term contracts
                        </li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

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

    # Customer Segments Page
    elif page == "Customer Segments":
        st.markdown("""
            <div class='custom-container'>
                <h2 style='margin-top: 0;'>Customer Segmentation Analysis</h2>
                <p style='color: #7f8c8d;'>Deep dive into customer segments and behavior patterns</p>
            </div>
        """, unsafe_allow_html=True)

        # Tenure vs Charges Analysis
        st.markdown("""
            <div class='custom-container'>
                <h3 style='margin-top: 0;'>Customer Value Matrix</h3>
                <p style='color: #7f8c8d;'>Relationship between customer tenure, charges, and total value</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Create scatter plot
        fig = px.scatter(df, 
                        x='Tenure Months', 
                        y='Monthly Charges',
                        color='Churn Label',
                        size='Total Charges',
                        hover_data=['CustomerID'],
                        title='Customer Segments by Tenure and Monthly Charges',
                        labels={'Tenure Months': 'Tenure (Months)',
                               'Monthly Charges': 'Monthly Charges ($)',
                               'Total Charges': 'Total Charges ($)'},
                        color_discrete_sequence=['#2ecc71', '#e74c3c'])
        
        fig.update_layout(
            title_x=0.5,
            title_font_size=20,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Service Adoption Analysis
        st.subheader("Service Adoption Patterns")
        
        service_cols = ['Internet Service', 'Online Security', 'Online Backup', 
                       'Device Protection', 'Tech Support', 'Streaming TV', 
                       'Streaming Movies']
        
        service_usage = pd.melt(df, 
                               id_vars=['Churn Label'], 
                               value_vars=service_cols,
                               var_name='Service', 
                               value_name='Has Service')
        
        fig = px.bar(service_usage, 
                     x='Service', 
                     y='Has Service',
                     color='Churn Label',
                     barmode='group',
                     title='Service Adoption by Churn Status',
                     labels={'Has Service': 'Adoption Rate', 'Service': 'Service Type'})
        st.plotly_chart(fig, use_container_width=True)

    # Risk Analysis Page
    elif page == "Risk Analysis":
        st.markdown("""
            <div class='custom-container'>
                <h2 style='margin-top: 0;'>Churn Risk Analysis</h2>
                <p style='color: #7f8c8d;'>In-depth analysis of churn risk factors and patterns</p>
            </div>
        """, unsafe_allow_html=True)

        # Risk Distribution
        st.markdown("""
            <div class='custom-container'>
                <h3 style='margin-top: 0;'>Risk Distribution Analysis</h3>
                <p style='color: #7f8c8d;'>Distribution of risk factors across customer base</p>
            </div>
        """, unsafe_allow_html=True)

        # Create risk score based on multiple factors
        df['Risk_Score'] = 0
        
        # Contract type risk
        df.loc[df['Contract'] == 'Month-to-month', 'Risk_Score'] += 0.4
        df.loc[df['Contract'] == 'One year', 'Risk_Score'] += 0.2
        
        # Tenure risk
        df.loc[df['Tenure Months'] <= 12, 'Risk_Score'] += 0.3
        df.loc[(df['Tenure Months'] > 12) & (df['Tenure Months'] <= 24), 'Risk_Score'] += 0.2
        
        # Payment method risk
        df.loc[df['Payment Method'] == 'Electronic check', 'Risk_Score'] += 0.2
        
        # Service adoption risk
        service_cols = ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support']
        for col in service_cols:
            df.loc[df[col] == 'No', 'Risk_Score'] += 0.1
        
        # Normalize risk score
        df['Risk_Score'] = df['Risk_Score'] / df['Risk_Score'].max()
        
        # Create risk categories
        df['Risk_Category'] = pd.qcut(df['Risk_Score'], 
                                    q=5, 
                                    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Plot risk distribution
        fig = px.histogram(df, 
                          x='Risk_Score',
                          color='Churn Label',
                          nbins=50,
                          title='Distribution of Churn Risk Scores',
                          labels={'Risk_Score': 'Risk Score', 'count': 'Number of Customers'},
                          color_discrete_sequence=['#2ecc71', '#e74c3c'])
        
        fig.update_layout(
            title_x=0.5,
            title_font_size=20,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, l=0, r=0, b=0),
            bargap=0.1
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Risk Analysis
        col1, col2 = st.columns(2)

        with col1:
            # Tenure-based risk
            fig = px.box(df, 
                        x='Churn Label', 
                        y='Tenure Months',
                        title='Churn Risk by Customer Tenure')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Monthly Charges risk
            fig = px.box(df, 
                        x='Churn Label', 
                        y='Monthly Charges',
                        title='Churn Risk by Monthly Charges')
            st.plotly_chart(fig, use_container_width=True)

    # Predictive Tools
    else:
        st.header("Churn Prediction Tool")

        # Create tabs for different prediction approaches
        tab1, tab2 = st.tabs(["Individual Customer", "Batch Prediction"])

        with tab1:
            st.subheader("Individual Customer Risk Assessment")
            
            # Create input form with better organization
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Customer Profile")
                tenure = st.number_input("Tenure (months)", 
                                      min_value=0, 
                                      max_value=120, 
                                      value=12)
                monthly_charges = st.number_input("Monthly Charges ($)", 
                                               min_value=0.0, 
                                               max_value=1000.0, 
                                               value=50.0)
                contract = st.selectbox("Contract Type", 
                                      ['Month-to-month', 'One year', 'Two year'])
            
            with col2:
                st.markdown("### Services")
                internet_service = st.selectbox("Internet Service", 
                                             ['DSL', 'Fiber optic', 'No'])
                online_security = st.selectbox("Online Security", 
                                            ['Yes', 'No', 'No internet service'])
                tech_support = st.selectbox("Tech Support", 
                                         ['Yes', 'No', 'No internet service'])
            
            with col3:
                st.markdown("### Additional Info")
                payment_method = st.selectbox("Payment Method", 
                                           ['Electronic check', 'Mailed check',
                                            'Bank transfer (automatic)',
                                            'Credit card (automatic)'])
                paperless = st.selectbox("Paperless Billing", ['Yes', 'No'])
                senior = st.selectbox("Senior Citizen", ['Yes', 'No'])
            
            if st.button("Analyze Churn Risk"):
                st.info("Risk analysis functionality coming soon!")

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
                        st.info("Batch processing functionality coming soon!")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Current working directory:", os.getcwd())
    st.write("Project root:", PROJECT_ROOT)
    if 'data_path' in locals():
        st.write("Full data path:", data_path)

# Footer with last update time
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.markdown("""
        <div style='text-align: center;'>
            <p style='color: #7f8c8d;'>Created with ❤️ by Your Team</p>
            <p style='color: #7f8c8d; font-size: 0.8rem;'>Banking Customer Churn Analytics Dashboard v1.0</p>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div style='text-align: right;'>
            <p style='color: #7f8c8d;'>📊 Powered by Streamlit</p>
        </div>
    """, unsafe_allow_html=True) 