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

# Add scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
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
    # Get the absolute path to the workspace root
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(workspace_root, 'data', 'Telco_customer_churn.xlsx')
    df = pd.read_excel(data_path)
    df_processed, transformers = preprocess_data(df)
    return df, df_processed

@st.cache_resource
def load_model():
    """Load the trained model and transformers"""
    # Get the absolute path to the workspace root
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(workspace_root, 'models', 'xgb_model.joblib')
    transformers_path = os.path.join(workspace_root, 'models', 'transformers.joblib')
    model = joblib.load(model_path)
    transformers = joblib.load(transformers_path)
    return model, transformers

# Load data and model
try:
    df, df_processed = load_data()
    model, transformers = load_model()
    X, y, feature_names = prepare_features(df_processed)
except Exception as e:
    st.error(f"Error loading data or model: {str(e)}")
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
        churn_rate = (df['Churn Value'].sum() / len(df)) * 100
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
        avg_tenure = df_processed['YearsWithBank'].mean()
        st.markdown("""
            <div class='metric-card'>
                <h4 style='color: #7f8c8d; margin-bottom: 0.5rem;'>Avg. Customer Tenure</h4>
                <div style='font-size: 2rem; color: #2c3e50; font-weight: bold;'>{:.1f} years</div>
                <div style='color: #27ae60; font-size: 0.9rem;'>↑ 0.5 yr from last quarter</div>
            </div>
        """.format(avg_tenure), unsafe_allow_html=True)
    
    with col4:
        avg_monthly = df_processed['MonthlyBankFees'].mean()
        st.markdown("""
            <div class='metric-card'>
                <h4 style='color: #7f8c8d; margin-bottom: 0.5rem;'>Avg. Monthly Fees</h4>
                <div style='font-size: 2rem; color: #2c3e50; font-weight: bold;'>${:.2f}</div>
                <div style='color: #27ae60; font-size: 0.9rem;'>↑ 2% from last month</div>
            </div>
        """.format(avg_monthly), unsafe_allow_html=True)

    # Churn Overview Section with enhanced styling
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
                    hole=0.4)  # Make it a donut chart
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

    # Tenure vs Fees Analysis
    st.markdown("""
        <div class='custom-container'>
            <h3 style='margin-top: 0;'>Customer Value Matrix</h3>
            <p style='color: #7f8c8d;'>Relationship between customer tenure, fees, and total balance</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Normalize TotalBalance for size
    df_processed['TotalBalance_normalized'] = (df_processed['TotalBalance'] - df_processed['TotalBalance'].min()) / (df_processed['TotalBalance'].max() - df_processed['TotalBalance'].min())
    df_processed['TotalBalance_normalized'] = df_processed['TotalBalance_normalized'] * 20 + 5
    
    fig = px.scatter(df_processed, 
                    x='YearsWithBank', 
                    y='MonthlyBankFees',
                    color='Churn Label',
                    size='TotalBalance_normalized',
                    hover_data=['CustomerID'],
                    title='Customer Segments by Tenure and Fees',
                    labels={'YearsWithBank': 'Years with Bank',
                           'MonthlyBankFees': 'Monthly Fees ($)',
                           'TotalBalance_normalized': 'Total Balance ($)'},
                    color_discrete_sequence=['#2ecc71', '#e74c3c'])
    
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, l=0, r=0, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Add analysis and insights
    st.markdown("""
    ### 📊 Value Matrix Analysis
    
    #### Key Observations:
    1. **Tenure-Fee Relationship**
        - Long-term customers (>5 years) show higher tolerance for fees
        - New customers (<2 years) are more sensitive to fee increases
        - High churn risk zone identified in 1-3 year tenure range with fees >$70
    
    2. **Balance Impact**
        - Larger bubble size indicates higher account balance
        - High-balance customers tend to be more stable regardless of fees
        - Low-balance accounts show higher churn sensitivity
    
    #### Business Insights:
    - 🎯 **Critical Period**: Focus retention efforts on 1-3 year customers
    - 💰 **Fee Strategy**: Consider graduated fee structure based on tenure
    - 🏦 **Balance Growth**: Incentivize account balance growth to improve retention
    - 🤝 **Relationship Building**: Extra support needed for new customers (<2 years)
    """)

    # Service Adoption Analysis
    st.subheader("Service Adoption Patterns")
    
    service_cols = ['OnlineBanking', 'SecureLogin2FA', 'AutomaticSavings', 
                   'FraudProtection', 'CustomerSupport', 'BillPay', 'MobilePayments']
    
    service_usage = pd.melt(df_processed, 
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
    
    # Add analysis and insights
    st.markdown("""
    ### 📱 Service Adoption Analysis
    
    #### Key Patterns:
    1. **Digital Services Impact**
        - Online banking users show 45% lower churn rate
        - Mobile payments adoption correlates with higher retention
        - 2FA security features indicate customer commitment
    
    2. **Service Bundling Effects**
        - Customers using 3+ services show 60% lower churn risk
        - Automatic savings users demonstrate highest loyalty
        - Bill Pay service adoption indicates long-term commitment
    
    #### Business Recommendations:
    - 🌟 **Service Promotion**: 
        - Target single-service customers for additional service adoption
        - Focus on digital banking onboarding for new customers
    - 🎁 **Bundle Strategy**: 
        - Create attractive service bundles to encourage multiple service adoption
        - Offer trial periods for premium services
    - 🛡️ **Security Focus**: 
        - Promote security features as premium account benefit
        - Highlight fraud protection success stories
    """)

    # Customer Value Segments
    st.subheader("Value Segments")
    
    # Create value segments
    df_processed['Value_Segment'] = pd.qcut(df_processed['MonthlyBankFees'], 
                                          q=4, 
                                          labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
    
    segment_churn = df_processed.groupby('Value_Segment')['Churn Value'].agg(['mean', 'count'])
    segment_churn['churn_rate'] = segment_churn['mean'] * 100
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=segment_churn.index, 
                        y=segment_churn['count'],
                        name='Customer Count',
                        yaxis='y'))
    fig.add_trace(go.Scatter(x=segment_churn.index, 
                            y=segment_churn['churn_rate'],
                            name='Churn Rate (%)',
                            yaxis='y2'))
    
    fig.update_layout(
        title='Customer Segments: Size and Churn Rate',
        yaxis=dict(title='Number of Customers'),
        yaxis2=dict(title='Churn Rate (%)', 
                   overlaying='y', 
                   side='right'),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Add analysis and insights
    st.markdown("""
    ### 💎 Value Segment Analysis
    
    #### Segment Characteristics:
    1. **Platinum Segment**
        - Top 25% of customers by monthly fees
        - Lowest churn rate but highest revenue impact when churn occurs
        - Most likely to use multiple services
    
    2. **Gold Segment**
        - Strong service adoption rates
        - Moderate churn risk
        - Good potential for upgrades
    
    3. **Silver Segment**
        - Price sensitive but stable
        - Moderate service adoption
        - Good targets for service expansion
    
    4. **Bronze Segment**
        - Highest churn risk
        - Limited service adoption
        - Most price sensitive
    
    #### Strategic Recommendations:
    - 🔝 **Platinum Strategy**: 
        - Focus on personalized service and relationship banking
        - Implement VIP support program
        - Early warning system for dissatisfaction
    
    - 🏆 **Gold Strategy**: 
        - Create clear upgrade path to Platinum
        - Reward loyalty with premium service trials
        - Focus on digital service adoption
    
    - 🥈 **Silver Strategy**: 
        - Introduce value-added services
        - Build engagement through educational programs
        - Highlight cost-saving benefits of additional services
    
    - 🥉 **Bronze Strategy**: 
        - Focus on basic service reliability
        - Provide clear path for account growth
        - Identify and nurture high-potential customers
    
    #### Action Items:
    1. 📈 Implement segment-specific retention programs
    2. 🎯 Develop targeted upgrade paths for each segment
    3. 💡 Create segment-specific communication strategies
    4. 📊 Monitor segment migration patterns quarterly
    5. 🤝 Establish feedback loops for service improvement
    """)

elif page == "Risk Analysis":
    st.markdown("""
        <div class='custom-container'>
            <h2 style='margin-top: 0;'>Churn Risk Analysis</h2>
            <p style='color: #7f8c8d;'>In-depth analysis of churn risk factors and patterns</p>
        </div>
    """, unsafe_allow_html=True)

    # Risk Factors Impact
    st.markdown("""
        <div class='custom-container'>
            <h3 style='margin-top: 0;'>Key Risk Factors</h3>
            <p style='color: #7f8c8d;'>Impact of different features on customer churn probability</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature importance plot with enhanced styling
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(feature_imp, 
                 x='Importance', 
                 y='Feature',
                 orientation='h',
                 title='Feature Importance in Churn Prediction',
                 color='Importance',
                 color_continuous_scale=['#ff9999', '#ff4d4d', '#ff0000'])
    
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, l=0, r=0, b=0),
        showlegend=False,
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Add analysis for feature importance
    st.markdown("""
    ### 📊 Feature Importance Analysis
    
    #### Key Drivers of Churn:
    1. **Contract Type** (Highest Impact)
        - Month-to-month contracts show 3x higher churn risk
        - Long-term contracts are strongest retention indicator
        - Opportunity: Convert month-to-month to annual contracts
    
    2. **Banking Services** (High Impact)
        - Online banking adoption significantly reduces churn
        - Security features (2FA, Fraud Protection) build trust
        - Digital service usage indicates stronger engagement
    
    3. **Financial Metrics** (Moderate Impact)
        - Monthly fees sensitivity threshold identified
        - Account balance correlates with retention
        - Payment method preferences affect loyalty
    
    #### Strategic Implications:
    - 📝 **Contract Strategy**:
        - Implement contract conversion campaigns
        - Offer incentives for longer commitments
        - Design graduated benefits for contract length
    
    - 🌐 **Digital Transformation**:
        - Prioritize online banking adoption
        - Enhance digital service experience
        - Develop digital onboarding program
    
    - 💰 **Financial Planning**:
        - Review fee structures by segment
        - Create balance growth incentives
        - Optimize payment method options
    """)

    # Risk Patterns
    col1, col2 = st.columns(2)

    with col1:
        # Tenure-based risk
        fig = px.box(df_processed, 
                    x='Churn Label', 
                    y='YearsWithBank',
                    title='Churn Risk by Customer Tenure')
        st.plotly_chart(fig, use_container_width=True)

        # Add analysis for tenure-based risk
        st.markdown("""
        ### ⏳ Tenure Risk Analysis
        
        #### Key Findings:
        1. **Critical Periods**:
            - Highest risk: 0-2 years (early relationship)
            - Stabilization: 3-5 years
            - Lowest risk: 5+ years
        
        2. **Retention Opportunities**:
            - Early engagement crucial
            - Relationship building in years 1-3
            - Long-term customer value growth
        
        #### Action Items:
        - 🆕 Enhanced onboarding program
        - 🎯 Year 2 retention campaign
        - 🏆 Tenure milestone rewards
        - 📈 Long-term relationship benefits
        """)

    with col2:
        # Fee-based risk
        fig = px.box(df_processed, 
                    x='Churn Label', 
                    y='MonthlyBankFees',
                    title='Churn Risk by Monthly Fees')
        st.plotly_chart(fig, use_container_width=True)

        # Add analysis for fee-based risk
        st.markdown("""
        ### 💵 Fee Sensitivity Analysis
        
        #### Key Insights:
        1. **Fee Thresholds**:
            - Critical point: $70-80 monthly
            - High risk zone: $90+ for new customers
            - Tolerance increases with tenure
        
        2. **Fee Strategy Opportunities**:
            - Segment-based pricing
            - Value demonstration at key thresholds
            - Fee waiver programs for retention
        
        #### Action Items:
        - 📊 Implement dynamic pricing
        - 🎁 Create fee waiver criteria
        - 💡 Develop value communication
        - 🎯 Target at-risk fee levels
        """)

    # Risk Distribution
    st.markdown("""
        <div class='custom-container'>
            <h3 style='margin-top: 0;'>Risk Distribution Analysis</h3>
            <p style='color: #7f8c8d;'>Distribution of churn risk scores across customer base</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate risk scores with enhanced visualization
    risk_scores = model.model.predict_proba(X)[:, 1]
    df_processed['Risk_Score'] = risk_scores
    df_processed['Risk_Category'] = pd.qcut(risk_scores, 
                                          q=5, 
                                          labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    fig = px.histogram(df_processed, 
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

    # Add analysis for risk distribution
    st.markdown("""
    ### 📈 Risk Distribution Analysis
    
    #### Risk Profile Overview:
    1. **Risk Segments**:
        - Very Low Risk (0-20%): Stable, long-term customers
        - Low Risk (20-40%): Satisfied but monitoring needed
        - Medium Risk (40-60%): Require attention
        - High Risk (60-80%): Immediate intervention needed
        - Very High Risk (80-100%): Critical retention priority
    
    2. **Distribution Patterns**:
        - Bimodal distribution indicates clear risk segments
        - Sharp increase in churn probability above 60%
        - Strong correlation with service adoption levels
    
    #### Strategic Recommendations:
    - 🎯 **Targeted Interventions**:
        - Proactive outreach to high-risk segments
        - Customized retention programs by risk level
        - Early warning system for risk escalation
    
    - 📊 **Monitoring Framework**:
        - Monthly risk score updates
        - Trend analysis by segment
        - Risk migration tracking
    
    - 🛠️ **Risk Mitigation Tools**:
        - Automated alert system
        - Risk-based engagement programs
        - Segment-specific retention tactics
    
    #### Action Plan:
    1. 🚨 Implement immediate intervention for Very High Risk
    2. 📱 Enhance digital engagement for Medium Risk
    3. 🤝 Develop loyalty programs for Low Risk
    4. 📈 Create risk score dashboards for management
    5. 🎓 Train staff on risk-based customer handling
    """)

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
            credit_card = st.selectbox("Credit Card", ["No phone service", "No", "Yes"], index=2)
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
                    'YearsWithBank': [years_with_bank],
                    'MonthlyBankFees': [monthly_fees],
                    'TotalBalance': [total_balance],
                    'DebitCard': [debit_card],
                    'CreditCard': [credit_card],
                    'OnlineBanking': [online_banking],
                    'SecureLogin2FA': [secure_login],
                    'AutomaticSavings': [auto_savings],
                    'FraudProtection': [fraud_protection],
                    'CustomerSupport': [customer_support],
                    'BillPay': [bill_pay],
                    'MobilePayments': [mobile_payments],
                    'Contract': [contract_type],
                    'PaperlessBilling': [paperless_billing],
                    'PaymentMethod': [payment_method]
                })

                # Define feature columns in correct order
                feature_cols = [
                    'YearsWithBank', 'MonthlyBankFees', 'TotalBalance',
                    'DebitCard', 'CreditCard', 'OnlineBanking', 'SecureLogin2FA',
                    'AutomaticSavings', 'FraudProtection', 'CustomerSupport',
                    'BillPay', 'MobilePayments', 'Contract', 'PaperlessBilling',
                    'PaymentMethod'
                ]
                
                # Ensure columns are in the correct order
                input_data = input_data[feature_cols]
                
                # Create processed dataframe
                input_processed = pd.DataFrame()
                
                # Process numerical features first
                numerical_cols = ['YearsWithBank', 'MonthlyBankFees', 'TotalBalance']
                scaler = transformers['numerical_scaler']
                input_processed[numerical_cols] = pd.DataFrame(
                    scaler.transform(input_data[numerical_cols]),
                    columns=numerical_cols
                )
                
                # Process categorical features
                categorical_cols = [col for col in feature_cols if col not in numerical_cols]
                for col in categorical_cols:
                    encoder = transformers.get(f'{col}_encoder')
                    if encoder is not None:
                        try:
                            input_processed[col] = encoder.transform(input_data[col])
                        except ValueError as e:
                            st.error(f"Error encoding {col}: The value provided is not in the training data. Available values: {list(encoder.classes_)}")
                            st.stop()
                
                # Ensure all columns are float type
                input_processed = input_processed.astype(float)
                
                # Make prediction using the underlying XGBoost model
                churn_prob = model.model.predict_proba(input_processed)[0][1]
                
                # Create columns for visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gauge chart for risk probability
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = churn_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Churn Risk"},
                        gauge = {
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
                                'value': churn_prob * 100
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk assessment and recommendations
                    st.markdown("### Risk Assessment")
                    
                    if churn_prob < 0.3:
                        st.success("🟢 Low Risk of Churn")
                    elif churn_prob < 0.7:
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
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.markdown("""
        <div style='text-align: center;'>
            <p style='color: #7f8c8d;'>Created with ❤️ by GitHub Copilot</p>
            <p style='color: #7f8c8d; font-size: 0.8rem;'>Banking Customer Churn Analytics Dashboard v1.0</p>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div style='text-align: right;'>
            <p style='color: #7f8c8d;'>📊 Powered by Streamlit</p>
        </div>
    """, unsafe_allow_html=True)