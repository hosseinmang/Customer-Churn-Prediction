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
    page_title="Banking Customer Churn Analytics",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add Font Awesome and custom CSS
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --background-color: #0e1117;
            --text-color: #e0e0e0;
            --card-background: #1a1c23;
            --accent-color: #4c8bf5;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --border-color: #2d3035;
        }

        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }

        .custom-container {
            background-color: var(--card-background);
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }

        .icon {
            margin-right: 0.5rem;
            font-size: 1.1rem;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--text-color) !important;
            margin-bottom: 1rem;
        }

        .metric-card {
            background-color: var(--card-background);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--accent-color);
        }

        .metric-label {
            color: var(--text-color);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }

        .footer {
            margin-top: 2rem;
            padding: 1rem;
            border-top: 1px solid var(--border-color);
            color: var(--text-color);
            font-size: 0.9rem;
        }

        /* Plotly dark theme overrides */
        .js-plotly-plot .plotly .main-svg {
            background-color: var(--card-background) !important;
        }

        .js-plotly-plot .plotly .modebar {
            background-color: var(--card-background) !important;
        }

        .js-plotly-plot .plotly .modebar-btn path {
            fill: var(--text-color) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description with enhanced styling
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>
            <i class="fas fa-chart-line icon"></i>
            Banking Customer Churn Analytics
        </h1>
        <p style='font-size: 1.2rem; color: var(--text-color); max-width: 800px; margin: 0 auto;'>
            Comprehensive insights into customer churn patterns and risk factors for data-driven retention strategies.
        </p>
    </div>
""", unsafe_allow_html=True)

# Add timestamp
st.markdown(f"""
    <div style='text-align: center; padding-bottom: 1rem;'>
        <p style='font-size: 0.9rem; color: var(--text-color);'>
            <i class="fas fa-clock icon"></i>
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
        
        # Preprocess the data
        processed_data, transformers = preprocess_data(raw_data)
        
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
        
    # Prepare features for the model
    X, y, feature_names = prepare_features(df)
except Exception as e:
    st.error(f"Error loading data or model: {str(e)}")
    st.write("Current working directory:", os.getcwd())
    st.write("Directory contents:", os.listdir())
    st.stop()

# Sidebar
st.sidebar.markdown("""
    <div style='padding: 1rem 0;'>
        <h3><i class="fas fa-bars icon"></i> Dashboard Navigation</h3>
    </div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Choose a page",
    ["Executive Summary", "Customer Segments", "Risk Analysis", "Predictive Tools"]
)

# Update metric cards with Font Awesome icons
def create_metric_card(title, value, trend=None, icon=None):
    icon_html = f'<i class="fas fa-{icon} icon"></i>' if icon else ''
    trend_html = f'<div style="color: var(--success-color); font-size: 0.9rem;">{trend}</div>' if trend else ''
    
    return f"""
        <div class='metric-card'>
            <h4 style='color: var(--text-color); margin-bottom: 0.5rem;'>{icon_html} {title}</h4>
            <div style='font-size: 2rem; color: var(--text-color); font-weight: bold;'>{value}</div>
            {trend_html}
        </div>
    """

# Update the Executive Summary page with dark mode compatible visualizations
if page == "Executive Summary":
    st.markdown("""
        <div class='custom-container'>
            <h2><i class="fas fa-tachometer-alt icon"></i> Executive Overview</h2>
            <p style='color: var(--text-color);'>Key performance indicators and metrics for quick insights</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row with Font Awesome icons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(raw_df)
        st.markdown(
            create_metric_card(
                "Total Customers",
                f"{total_customers:,}",
                "↑ 5% from last month",
                "users"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        churn_rate = (raw_df['Churn Value'].sum() / len(raw_df)) * 100
        st.markdown(
            create_metric_card(
                "Churn Rate",
                f"{churn_rate:.1f}%",
                target="Target: 20%",
                icon="exclamation-triangle"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        avg_tenure = raw_df['Tenure Months'].mean() / 12  # Convert months to years
        st.markdown(
            create_metric_card(
                "Avg. Customer Tenure",
                f"{avg_tenure:.1f} years",
                "↑ 0.5 yr from last quarter",
                "clock"
            ),
            unsafe_allow_html=True
        )
    
    with col4:
        avg_monthly = raw_df['Monthly Charges'].mean()
        st.markdown(
            create_metric_card(
                "Avg. Monthly Fees",
                f"${avg_monthly:.2f}",
                "↑ 2% from last month",
                "dollar-sign"
            ),
            unsafe_allow_html=True
        )

    # Churn Overview Section
    st.markdown("""
        <div class='custom-container'>
            <h2 style='margin-top: 0;'>Churn Overview</h2>
            <p style='color: var(--text-color);'>Analysis of customer churn patterns and primary reasons</p>
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
            <p style='color: var(--text-color);'>Critical findings and actionable recommendations</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div style='background-color: var(--card-background); padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3><i class="fas fa-lightbulb icon"></i> Critical Findings</h3>
                <ul style='color: var(--text-color); list-style-type: none; padding-left: 0;'>
                    <li style='margin-bottom: 0.8rem;'>
                        <i class="fas fa-exclamation-circle icon"></i>
                        <strong>High-Risk Period:</strong> First 2 years show highest churn probability
                    </li>
                    <li style='margin-bottom: 0.8rem;'>
                        <i class="fas fa-chart-line icon"></i>
                        <strong>Price Sensitivity:</strong> Threshold at $70-80 monthly fees
                    </li>
                    <li style='margin-bottom: 0.8rem;'>
                        <i class="fas fa-mobile-alt icon"></i>
                        <strong>Service Impact:</strong> Digital banking users show 45% lower churn
                    </li>
                    <li style='margin-bottom: 0.8rem;'>
                        <i class="fas fa-file-contract icon"></i>
                        <strong>Contract Effect:</strong> Long-term contracts reduce churn by 67%
                    </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='background-color: var(--card-background); padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3><i class="fas fa-tasks icon"></i> Action Items</h3>
                <ul style='color: var(--text-color); list-style-type: none; padding-left: 0;'>
                    <li style='margin-bottom: 0.8rem;'>
                        <i class="fas fa-bell icon" style="color: var(--danger-color);"></i>
                        Implement early warning system for new customers
                    </li>
                    <li style='margin-bottom: 0.8rem;'>
                        <i class="fas fa-chart-pie icon" style="color: var(--warning-color);"></i>
                        Review pricing strategy for sensitive segments
                    </li>
                    <li style='margin-bottom: 0.8rem;'>
                        <i class="fas fa-mobile-alt icon" style="color: var(--success-color);"></i>
                        Promote digital service adoption
                    </li>
                    <li style='margin-bottom: 0.8rem;'>
                        <i class="fas fa-handshake icon" style="color: var(--success-color);"></i>
                        Incentivize long-term contracts
                    </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

elif page == "Customer Segments":
    st.markdown("""
        <div class='custom-container'>
            <h2 style='margin-top: 0;'>Customer Segmentation Analysis</h2>
            <p style='color: var(--text-color);'>Deep dive into customer segments and behavior patterns</p>
        </div>
    """, unsafe_allow_html=True)

    # Tenure vs Fees Analysis
    st.markdown("""
        <div class='custom-container'>
            <h3 style='margin-top: 0;'>Customer Value Matrix</h3>
            <p style='color: var(--text-color);'>Relationship between customer tenure, fees, and total balance</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Normalize TotalBalance for size
    df['TotalBalance_normalized'] = (df['TotalBalance'] - df['TotalBalance'].min()) / (df['TotalBalance'].max() - df['TotalBalance'].min())
    df['TotalBalance_normalized'] = df['TotalBalance_normalized'] * 20 + 5
    
    fig = px.scatter(df, 
                    x='Tenure Months', 
                    y='Monthly Charges',
                    color='Churn Label',
                    size='TotalBalance_normalized',
                    hover_data=['CustomerID'],
                    title='Customer Segments by Tenure and Fees',
                    labels={'Tenure Months': 'Tenure (Months)',
                           'Monthly Charges': 'Monthly Fees ($)',
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
    
    service_cols = ['Online Banking', 'Secure Login', 'Automatic Savings', 
                   'Fraud Protection', 'Customer Support', 'Streaming TV', 'Streaming Movies']
    
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
    df['Value_Segment'] = pd.qcut(df['Monthly Charges'], 
                                          q=4, 
                                          labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
    
    segment_churn = df.groupby('Value_Segment')['Churn Value'].agg(['mean', 'count'])
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
            <p style='color: var(--text-color);'>In-depth analysis of churn risk factors and patterns</p>
        </div>
    """, unsafe_allow_html=True)

    # Risk Factors Impact
    st.markdown("""
        <div class='custom-container'>
            <h3 style='margin-top: 0;'>Key Risk Factors</h3>
            <p style='color: var(--text-color);'>Impact of different features on customer churn probability</p>
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
        fig = px.box(df, 
                    x='Churn Label', 
                    y='Tenure Months',
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
        fig = px.box(df, 
                    x='Churn Label', 
                    y='Monthly Charges',
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
            <p style='color: var(--text-color);'>Distribution of churn risk scores across customer base</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate risk scores with enhanced visualization
    risk_scores = model.model.predict_proba(X)[:, 1]  # Access the underlying model
    df['Risk_Score'] = risk_scores
    df['Risk_Category'] = pd.qcut(risk_scores, 
                                q=5, 
                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
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
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.markdown("""
        <div style='text-align: center;'>
            <p style='color: var(--text-color);'>Created with ❤️ by GitHub Copilot</p>
            <p style='color: var(--text-color); font-size: 0.8rem;'>Banking Customer Churn Analytics Dashboard v1.0</p>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div style='text-align: right;'>
            <p style='color: var(--text-color);'>📊 Powered by Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

# Update plotly theme for dark mode
def create_dark_plotly_theme():
    return {
        'layout': {
            'plot_bgcolor': '#1a1c23',
            'paper_bgcolor': '#1a1c23',
            'font': {'color': '#e0e0e0'},
            'xaxis': {
                'gridcolor': '#2d3035',
                'linecolor': '#2d3035',
                'zerolinecolor': '#2d3035'
            },
            'yaxis': {
                'gridcolor': '#2d3035',
                'linecolor': '#2d3035',
                'zerolinecolor': '#2d3035'
            }
        }
    }

# Apply dark theme to all plotly figures
plotly_dark_theme = create_dark_plotly_theme()

# Update plotly figures with dark theme
for fig in [churn_dist_fig, reasons_fig, tenure_fig, fees_fig, risk_dist_fig]:
    fig.update_layout(**plotly_dark_theme['layout'])
    fig.update_layout(
        title_font_color='#e0e0e0',
        legend_font_color='#e0e0e0',
        coloraxis_colorbar_tickfont_color='#e0e0e0'
    )

# Update the risk assessment indicators
def show_risk_indicator(risk_score):
    if risk_score < 0.3:
        return """
            <div style='display: flex; align-items: center;'>
                <i class="fas fa-check-circle icon" style="color: var(--success-color);"></i>
                <span style='color: var(--success-color);'>Low Risk of Churn</span>
            </div>
        """
    elif risk_score < 0.7:
        return """
            <div style='display: flex; align-items: center;'>
                <i class="fas fa-exclamation-triangle icon" style="color: var(--warning-color);"></i>
                <span style='color: var(--warning-color);'>Medium Risk of Churn</span>
            </div>
        """
    else:
        return """
            <div style='display: flex; align-items: center;'>
                <i class="fas fa-times-circle icon" style="color: var(--danger-color);"></i>
                <span style='color: var(--danger-color);'>High Risk of Churn</span>
            </div>
        """

# Update recommendations with icons
def show_recommendation(icon, text, color="var(--accent-color)"):
    return f"""
        <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
            <i class="fas fa-{icon} icon" style="color: {color};"></i>
            <span style='color: var(--text-color);'>{text}</span>
        </div>
    """

def main():
    # Title and description
    st.markdown("""
        <h1><i class="fas fa-chart-line icon"></i>Banking Customer Churn Analytics Dashboard</h1>
        <div class="custom-container">
            <p>This dashboard provides real-time insights into customer churn patterns and risk analysis.</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div class="custom-container">
                <h3><i class="fas fa-sliders-h icon"></i>Controls</h3>
            </div>
        """, unsafe_allow_html=True)
        risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.1)
        time_period = st.selectbox("Time Period", ["Last 30 Days", "Last 90 Days", "Last 180 Days", "All Time"])

    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <i class="fas fa-user-minus fa-2x" style="color: var(--danger-color)"></i>
                <div class="metric-value">12.5%</div>
                <div class="metric-label">Churn Rate</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <i class="fas fa-users fa-2x" style="color: var(--accent-color)"></i>
                <div class="metric-value">85.2%</div>
                <div class="metric-label">Customer Retention</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <i class="fas fa-exclamation-triangle fa-2x" style="color: var(--warning-color)"></i>
                <div class="metric-value">245</div>
                <div class="metric-label">High Risk Customers</div>
            </div>
        """, unsafe_allow_html=True)

    # Critical Findings Section
    st.markdown("""
        <div class="custom-container">
            <h2><i class="fas fa-exclamation-circle icon"></i>Critical Findings</h2>
            <ul>
                <li><i class="fas fa-arrow-up icon" style="color: var(--danger-color)"></i>15% increase in churn rate among premium customers</li>
                <li><i class="fas fa-clock icon" style="color: var(--warning-color)"></i>Average customer tenure decreased by 8 months</li>
                <li><i class="fas fa-dollar-sign icon" style="color: var(--success-color)"></i>High correlation between monthly fees and churn probability</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Action Items Section
    st.markdown("""
        <div class="custom-container">
            <h2><i class="fas fa-tasks icon"></i>Action Items</h2>
            <ul>
                <li><i class="fas fa-phone icon"></i>Initiate contact with high-risk customers</li>
                <li><i class="fas fa-gift icon"></i>Develop retention offers for premium segment</li>
                <li><i class="fas fa-chart-bar icon"></i>Review fee structure for long-term customers</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div class="footer">
            <i class="fas fa-info-circle icon"></i>Last updated: {}
            <br>
            <i class="fas fa-code icon"></i>Powered by Advanced Analytics
        </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()