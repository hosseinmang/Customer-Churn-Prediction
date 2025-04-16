import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="Banking Customer Churn Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f5f7f9;
        }
        .stMetric {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .insight-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        h1, h2, h3 {
            color: #1f4287;
        }
    </style>
""", unsafe_allow_html=True)

# Data loading function
@st.cache_data
def load_data():
    try:
        data_path = os.path.join('data', 'Telco_customer_churn.xlsx')
        df = pd.read_excel(data_path)
        
        # Convert Total Charges to numeric
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        
        # Fill missing values
        df['Total Charges'].fillna(df['Monthly Charges'], inplace=True)
        
        # Ensure numeric columns are properly typed
        df['Monthly Charges'] = pd.to_numeric(df['Monthly Charges'], errors='coerce')
        df['Tenure Months'] = pd.to_numeric(df['Tenure Months'], errors='coerce')
        
        # Calculate additional metrics
        df['Revenue_Risk'] = df['Monthly Charges'] * df['Churn Value']
        df['Customer_Lifetime'] = df['Total Charges'] / df['Monthly Charges']
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
df = load_data()

# Title
st.title("🏦 Banking Customer Churn Analytics")
st.markdown("""
    This dashboard provides comprehensive insights into customer churn patterns, 
    helping identify at-risk customers and optimize retention strategies.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Analysis",
    ["Executive Summary", "Customer Segments", "Risk Factors", "Financial Impact", "Retention Strategies"]
)

if df is not None:
    if page == "Executive Summary":
        st.header("Executive Summary")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(df)
            st.metric("Total Customers", f"{total_customers:,}")
            
        with col2:
            churn_rate = (df['Churn Value'].mean() * 100).round(2)
            st.metric("Churn Rate", f"{churn_rate}%")
            
        with col3:
            avg_tenure = df['Tenure Months'].mean().round(1)
            st.metric("Avg. Tenure (Months)", f"{avg_tenure}")
            
        with col4:
            monthly_revenue = df['Monthly Charges'].sum()
            at_risk_revenue = df[df['Churn Value'] == 1]['Monthly Charges'].sum()
            st.metric("Revenue at Risk", f"${at_risk_revenue:,.2f}", 
                     f"{(at_risk_revenue/monthly_revenue*100):.1f}% of Revenue")
        
        # Churn Overview
        st.subheader("Churn Distribution and Trends")
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn by Contract Type
            contract_churn = df.groupby('Contract')['Churn Value'].mean().reset_index()
            fig = px.bar(contract_churn,
                        x='Contract',
                        y='Churn Value',
                        title='Churn Rate by Contract Type',
                        labels={'Churn Value': 'Churn Rate', 'Contract': 'Contract Type'},
                        color='Churn Value',
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Top Churn Reasons
            churn_reasons = df[df['Churn Value'] == 1]['Churn Reason'].value_counts().head(5)
            fig = px.pie(values=churn_reasons.values,
                        names=churn_reasons.index,
                        title='Top 5 Churn Reasons')
            st.plotly_chart(fig, use_container_width=True)
            
    elif page == "Customer Segments":
        st.header("Customer Segmentation Analysis")
        
        try:
            # Clean and prepare data for visualization
            plot_df = df.copy()
            plot_df = plot_df.dropna(subset=['Tenure Months', 'Monthly Charges', 'Total Charges'])
            
            # Normalize Total Charges for bubble size
            plot_df['Size'] = (plot_df['Total Charges'] - plot_df['Total Charges'].min()) / \
                            (plot_df['Total Charges'].max() - plot_df['Total Charges'].min()) * 30 + 5
            
            # Create scatter plot with cleaned data
            fig = px.scatter(plot_df,
                           x='Tenure Months',
                           y='Monthly Charges',
                           color='Churn Label',
                           size='Size',
                           hover_data={
                               'Size': False,
                               'Total Charges': ':$.2f',
                               'Contract': True,
                               'Payment Method': True
                           },
                           title='Customer Distribution by Tenure and Monthly Charges',
                           labels={
                               'Tenure Months': 'Tenure (Months)',
                               'Monthly Charges': 'Monthly Charges ($)'
                           },
                           color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'})
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Segment Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Contract Distribution
                contract_dist = df['Contract'].value_counts()
                fig = px.pie(values=contract_dist.values,
                           names=contract_dist.index,
                           title='Customer Contract Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Service Adoption
                services = ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support']
                service_adoption = df[services].apply(lambda x: (x == 'Yes').mean() * 100)
                fig = px.bar(x=service_adoption.index,
                           y=service_adoption.values,
                           title='Service Adoption Rates',
                           labels={'x': 'Service', 'y': 'Adoption Rate (%)'},
                           color=service_adoption.values,
                           color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in Customer Segments visualization: {str(e)}")
            st.write("Please check the data format and try again.")
            
    elif page == "Risk Factors":
        st.header("Churn Risk Analysis")
        
        # Risk Factors Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn by Tenure
            tenure_bins = pd.qcut(df['Tenure Months'].dropna(), q=5)
            tenure_churn = df.dropna(subset=['Tenure Months']).groupby(tenure_bins)['Churn Value'].mean().reset_index()
            fig = px.line(tenure_churn,
                         x='Tenure Months',
                         y='Churn Value',
                         title='Churn Rate by Customer Tenure',
                         labels={'Churn Value': 'Churn Rate'})
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Payment Method Analysis
            payment_churn = df.groupby('Payment Method')['Churn Value'].mean().reset_index()
            fig = px.bar(payment_churn,
                        x='Payment Method',
                        y='Churn Value',
                        title='Churn Rate by Payment Method',
                        color='Churn Value',
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
            
    elif page == "Financial Impact":
        st.header("Financial Impact Analysis")
        
        # Revenue Analysis
        monthly_revenue = df['Monthly Charges'].sum()
        at_risk_revenue = df[df['Churn Value'] == 1]['Monthly Charges'].sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue at Risk
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=at_risk_revenue/monthly_revenue*100,
                title={'text': "Revenue at Risk (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#ff6b6b"},
                       'steps': [
                           {'range': [0, 20], 'color': "#4ecdc4"},
                           {'range': [20, 40], 'color': "#ffe66d"},
                           {'range': [40, 100], 'color': "#ff6b6b"}
                       ]}
            ))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # CLTV Analysis
            fig = px.box(df,
                        x='Contract',
                        y='CLTV',
                        color='Churn Label',
                        title='Customer Lifetime Value by Contract Type',
                        labels={'CLTV': 'Customer Lifetime Value ($)'})
            st.plotly_chart(fig, use_container_width=True)
            
    else:  # Retention Strategies
        st.header("Retention Strategy Recommendations")
        
        st.markdown("""
        <div class="insight-card">
            <h3>1. Contract Optimization 📋</h3>
            <ul>
                <li>Promote long-term contracts with incentives</li>
                <li>Develop loyalty programs for month-to-month customers</li>
                <li>Create special offers for contract renewals</li>
            </ul>
        </div>
        
        <div class="insight-card">
            <h3>2. Service Enhancement 🌟</h3>
            <ul>
                <li>Improve online security and support services</li>
                <li>Introduce bundled services packages</li>
                <li>Enhance customer support quality</li>
            </ul>
        </div>
        
        <div class="insight-card">
            <h3>3. Early Intervention Program ⚡</h3>
            <ul>
                <li>Monitor customer satisfaction in first 12 months</li>
                <li>Implement proactive support outreach</li>
                <li>Develop early warning system for churn risk</li>
            </ul>
        </div>
        
        <div class="insight-card">
            <h3>4. Competitive Pricing Strategy 💰</h3>
            <ul>
                <li>Regular market price analysis</li>
                <li>Flexible pricing for high-risk segments</li>
                <li>Value-added services for price-sensitive customers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("Unable to load data. Please check the data file and try again.")

# Footer
st.markdown("---")
st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")