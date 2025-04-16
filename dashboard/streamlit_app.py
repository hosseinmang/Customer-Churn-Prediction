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
        
        # Key Performance Indicators (KPIs)
        st.subheader("📊 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(df)
            st.metric("Total Customers", f"{total_customers:,}")
            
        with col2:
            churn_rate = (df['Churn Value'].mean() * 100).round(2)
            industry_avg = 15.0  # Banking industry average
            delta = industry_avg - churn_rate
            st.metric("Churn Rate", f"{churn_rate}%", 
                     f"{abs(delta):.1f}% {'below' if delta > 0 else 'above'} industry average")
            
        with col3:
            avg_tenure = df['Tenure Months'].mean().round(1)
            st.metric("Avg. Tenure (Months)", f"{avg_tenure:.1f}", 
                     f"{(avg_tenure/12):.1f} years")
            
        with col4:
            monthly_revenue = df['Monthly Charges'].sum()
            at_risk_revenue = df[df['Churn Value'] == 1]['Monthly Charges'].sum()
            st.metric("Monthly Revenue at Risk", 
                     f"${at_risk_revenue:,.2f}", 
                     f"{(at_risk_revenue/monthly_revenue*100):.1f}% of total revenue")

        # Executive Insights
        st.subheader("💡 Executive Insights")
        
        # Calculate key metrics for insights
        mtm_churn = df[df['Contract'] == 'Month-to-month']['Churn Value'].mean() * 100
        long_term_churn = df[df['Contract'] != 'Month-to-month']['Churn Value'].mean() * 100
        high_value_churn = df[df['Monthly Charges'] > df['Monthly Charges'].median()]['Churn Value'].mean() * 100
        service_adoption = df[['Online Security', 'Online Backup', 'Device Protection', 'Tech Support']].apply(lambda x: (x == 'Yes').mean() * 100)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-card">
                <h3>🎯 Key Findings</h3>
                <ul>
                    <li>Month-to-month contracts show {:.1f}x higher churn rate compared to long-term contracts</li>
                    <li>High-value customers (>${:.0f}/month) have {:.1f}% churn rate</li>
                    <li>Average service adoption rate is only {:.1f}%</li>
                    <li>{:.1f}% of churned customers cited competitor offers as the reason</li>
                </ul>
            </div>
            """.format(
                mtm_churn / long_term_churn,
                df['Monthly Charges'].median(),
                high_value_churn,
                service_adoption.mean(),
                (df[df['Churn Value'] == 1]['Churn Reason'].str.contains('competitor', case=False, na=False).mean() * 100)
            ), unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="insight-card">
                <h3>🚀 Recommended Actions</h3>
                <ul>
                    <li>Launch targeted retention campaigns for month-to-month customers</li>
                    <li>Develop competitive service bundles to increase adoption</li>
                    <li>Implement early warning system for high-value customers</li>
                    <li>Review pricing strategy against competitors</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Churn Distribution Analysis
        st.subheader("📈 Churn Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced Contract Analysis
            contract_churn = df.groupby('Contract').agg({
                'Churn Value': ['mean', 'count']
            }).reset_index()
            contract_churn.columns = ['Contract', 'Churn Rate', 'Customer Count']
            contract_churn['Churn Rate'] = contract_churn['Churn Rate'] * 100
            
            fig = px.bar(contract_churn,
                        x='Contract',
                        y='Churn Rate',
                        title='Churn Rate by Contract Type',
                        text=contract_churn['Churn Rate'].round(1).astype(str) + '%',
                        color='Churn Rate',
                        color_continuous_scale='RdYlGn_r',
                        custom_data=['Customer Count'])
            
            fig.update_traces(
                textposition='outside',
                hovertemplate="<br>".join([
                    "Contract: %{x}",
                    "Churn Rate: %{text}",
                    "Customer Count: %{customdata[0]:,.0f}",
                    "<extra></extra>"
                ])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add analysis
            highest_churn = contract_churn.loc[contract_churn['Churn Rate'].idxmax()]
            st.info(f"📊 {highest_churn['Contract']} contracts have the highest churn rate at {highest_churn['Churn Rate']:.1f}%, "
                   f"affecting {highest_churn['Customer Count']:,} customers.")
            
        with col2:
            # Enhanced Churn Reasons Analysis
            churn_reasons = df[df['Churn Value'] == 1]['Churn Reason'].value_counts().head(5)
            total_churned = len(df[df['Churn Value'] == 1])
            
            fig = px.pie(
                values=churn_reasons.values,
                names=churn_reasons.index,
                title='Top 5 Churn Reasons',
                hole=0.4
            )
            
            fig.update_traces(
                textposition='outside',
                textinfo='label+percent',
                hovertemplate="<br>".join([
                    "Reason: %{label}",
                    "Count: %{value:,.0f}",
                    "Percentage: %{percent}",
                    "<extra></extra>"
                ])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add analysis
            top_reason = churn_reasons.index[0]
            top_reason_pct = (churn_reasons.values[0] / total_churned * 100)
            st.info(f"🔍 '{top_reason}' is the leading churn reason, accounting for {top_reason_pct:.1f}% of all churned customers.")
            
    elif page == "Customer Segments":
        st.header("Customer Segmentation Analysis")
        
        # Overview metrics
        st.subheader("📊 Customer Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_monthly = df['Monthly Charges'].mean()
            st.metric("Avg. Monthly Charges", 
                     f"${avg_monthly:.2f}",
                     f"${df[df['Churn Value'] == 1]['Monthly Charges'].mean() - avg_monthly:.2f} for churned")
            
        with col2:
            avg_tenure = df['Tenure Months'].mean()
            st.metric("Avg. Customer Tenure",
                     f"{avg_tenure:.1f} months",
                     f"{df[df['Churn Value'] == 1]['Tenure Months'].mean() - avg_tenure:.1f} for churned")
            
        with col3:
            service_penetration = df[['Online Security', 'Online Backup', 'Device Protection', 'Tech Support']].apply(lambda x: (x == 'Yes').mean() * 100).mean()
            st.metric("Service Adoption Rate",
                     f"{service_penetration:.1f}%",
                     f"{service_penetration - df[df['Churn Value'] == 1][['Online Security', 'Online Backup', 'Device Protection', 'Tech Support']].apply(lambda x: (x == 'Yes').mean() * 100).mean():.1f}% vs churned")
            
        with col4:
            paperless_rate = (df['Paperless Billing'] == 'Yes').mean() * 100
            st.metric("Paperless Billing Rate",
                     f"{paperless_rate:.1f}%",
                     f"{((df[df['Churn Value'] == 1]['Paperless Billing'] == 'Yes').mean() * 100) - paperless_rate:.1f}% for churned")
        
        try:
            # Customer Value Segmentation
            st.subheader("💎 Customer Value Analysis")
            
            # Create value segments
            df['Value_Segment'] = pd.qcut(df['Monthly Charges'], q=3, labels=['Budget', 'Mid-tier', 'Premium'])
            
            # Calculate segment metrics
            segment_metrics = df.groupby('Value_Segment').agg({
                'Churn Value': 'mean',
                'Monthly Charges': 'mean',
                'Tenure Months': 'mean',
                'CustomerID': 'count'
            }).round(2)
            
            segment_metrics.columns = ['Churn Rate', 'Avg Monthly Charges', 'Avg Tenure', 'Customer Count']
            segment_metrics['Churn Rate'] = segment_metrics['Churn Rate'] * 100
            
            # Display segment metrics
            st.dataframe(segment_metrics.style.format({
                'Churn Rate': '{:.1f}%',
                'Avg Monthly Charges': '${:.2f}',
                'Avg Tenure': '{:.1f} months',
                'Customer Count': '{:,.0f}'
            }))
            
            # Customer Distribution Visualization
            st.subheader("📊 Customer Distribution Analysis")
            
            # Clean and prepare data for visualization
            plot_df = df.copy()
            plot_df = plot_df.dropna(subset=['Tenure Months', 'Monthly Charges', 'Total Charges'])
            
            # Normalize Total Charges for bubble size
            plot_df['Size'] = (plot_df['Total Charges'] - plot_df['Total Charges'].min()) / \
                            (plot_df['Total Charges'].max() - plot_df['Total Charges'].min()) * 30 + 5
            
            # Create enhanced scatter plot
            fig = px.scatter(plot_df,
                           x='Tenure Months',
                           y='Monthly Charges',
                           color='Churn Label',
                           size='Size',
                           hover_data={
                               'Size': False,
                               'Total Charges': ':$.2f',
                               'Contract': True,
                               'Payment Method': True,
                               'Value_Segment': True
                           },
                           title='Customer Distribution by Tenure and Monthly Charges',
                           labels={
                               'Tenure Months': 'Tenure (Months)',
                               'Monthly Charges': 'Monthly Charges ($)'
                           },
                           color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'})
            
            fig.update_layout(
                annotations=[{
                    'text': 'Bubble size represents total customer spend',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0,
                    'y': -0.1,
                    'showarrow': False,
                    'font': {'size': 10, 'color': 'gray'}
                }]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add distribution insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="insight-card">
                    <h3>📈 Distribution Insights</h3>
                    <ul>
                        <li>High concentration of churned customers in early tenure months</li>
                        <li>Premium segment shows higher retention in long-term contracts</li>
                        <li>Clear correlation between contract type and churn risk</li>
                        <li>Higher monthly charges correlate with increased churn risk in new customers</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="insight-card">
                    <h3>🎯 Targeting Opportunities</h3>
                    <ul>
                        <li>Focus retention efforts on first 12 months of customer lifecycle</li>
                        <li>Develop specialized offerings for each value segment</li>
                        <li>Create early engagement programs for high-value customers</li>
                        <li>Design targeted upgrade paths for budget segment</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Service Adoption Analysis
            st.subheader("🔧 Service Adoption Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced Contract Distribution
                contract_dist = df.groupby(['Contract', 'Churn Label']).size().unstack(fill_value=0)
                contract_dist_pct = contract_dist.div(contract_dist.sum(axis=1), axis=0) * 100
                
                fig = px.bar(
                    contract_dist_pct.reset_index(),
                    x='Contract',
                    y=['No', 'Yes'],
                    title='Customer Distribution by Contract Type',
                    labels={'value': 'Percentage', 'variable': 'Churned'},
                    color_discrete_map={'No': '#4ecdc4', 'Yes': '#ff6b6b'}
                )
                
                fig.update_layout(barmode='stack')
                fig.update_traces(texttemplate='%{y:.1f}%', textposition='inside')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add contract insights
                best_contract = contract_dist_pct['Yes'].idxmin()
                worst_contract = contract_dist_pct['Yes'].idxmax()
                st.info(f"📋 {best_contract} contracts show lowest churn at {contract_dist_pct.loc[best_contract, 'Yes']:.1f}%, "
                       f"while {worst_contract} contracts have {contract_dist_pct.loc[worst_contract, 'Yes']:.1f}% churn rate")
            
            with col2:
                # Enhanced Service Adoption
                services = ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support']
                service_adoption = df[services].apply(lambda x: (x == 'Yes').mean() * 100)
                service_impact = df.groupby(services).agg({
                    'Churn Value': ['mean', 'count']
                }).reset_index()
                
                fig = px.bar(
                    x=service_adoption.index,
                    y=service_adoption.values,
                    title='Service Adoption Rates and Churn Impact',
                    labels={'x': 'Service', 'y': 'Adoption Rate (%)'},
                    color=service_adoption.values,
                    color_continuous_scale='Viridis'
                )
                
                fig.update_traces(
                    text=service_adoption.values.round(1).astype(str) + '%',
                    textposition='outside'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add service insights
                lowest_adoption = service_adoption.idxmin()
                highest_adoption = service_adoption.idxmax()
                st.info(f"🔧 {highest_adoption} has highest adoption at {service_adoption[highest_adoption]:.1f}%, "
                       f"while {lowest_adoption} needs attention at {service_adoption[lowest_adoption]:.1f}%")
        
        except Exception as e:
            st.error(f"Error in Customer Segments visualization: {str(e)}")
            st.write("Please check the data format and try again.")
            
    elif page == "Risk Factors":
        st.header("Churn Risk Analysis")
        
        # Risk Overview
        st.subheader("🎯 Risk Overview")
        
        # Calculate key risk metrics
        high_risk_count = len(df[df['Churn Score'] >= 80])
        medium_risk_count = len(df[(df['Churn Score'] >= 50) & (df['Churn Score'] < 80)])
        low_risk_count = len(df[df['Churn Score'] < 50])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("High Risk Customers",
                     f"{high_risk_count:,}",
                     f"{high_risk_count/len(df)*100:.1f}% of base")
            
        with col2:
            st.metric("Medium Risk Customers",
                     f"{medium_risk_count:,}",
                     f"{medium_risk_count/len(df)*100:.1f}% of base")
            
        with col3:
            st.metric("Low Risk Customers",
                     f"{low_risk_count:,}",
                     f"{low_risk_count/len(df)*100:.1f}% of base")
        
        # Risk Factor Analysis
        st.subheader("📊 Key Risk Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                # Enhanced Tenure Analysis
                df['Tenure_Range'] = pd.cut(
                    df['Tenure Months'].dropna(),
                    bins=[0, 12, 24, 36, 48, float('inf')],
                    labels=['0-12 months', '13-24 months', '25-36 months', '37-48 months', '48+ months']
                )
                
                tenure_analysis = df.groupby('Tenure_Range').agg({
                    'Churn Value': ['mean', 'count'],
                    'Monthly Charges': 'mean',
                    'CLTV': 'mean'
                }).round(2)
                
                tenure_analysis.columns = ['Churn Rate', 'Customer Count', 'Avg Monthly Charges', 'Avg CLTV']
                tenure_analysis['Churn Rate'] = tenure_analysis['Churn Rate'] * 100
                
                # Create enhanced visualization
                fig = px.bar(
                    tenure_analysis.reset_index(),
                    x='Tenure_Range',
                    y='Churn Rate',
                    text=tenure_analysis['Churn Rate'].round(1).astype(str) + '%',
                    title='Churn Rate by Customer Tenure',
                    labels={
                        'Tenure_Range': 'Tenure Range',
                        'Churn Rate': 'Churn Rate (%)'
                    },
                    color='Churn Rate',
                    color_continuous_scale='RdYlGn_r',
                    custom_data=['Customer Count', 'Avg Monthly Charges', 'Avg CLTV']
                )
                
                fig.update_traces(
                    textposition='outside',
                    hovertemplate="<br>".join([
                        "Tenure: %{x}",
                        "Churn Rate: %{text}",
                        "Customers: %{customdata[0]:,.0f}",
                        "Avg. Monthly: $%{customdata[1]:.2f}",
                        "Avg. CLTV: $%{customdata[2]:,.2f}",
                        "<extra></extra>"
                    ])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add tenure insights
                highest_risk_tenure = tenure_analysis['Churn Rate'].idxmax()
                st.info(f"⚠️ Highest risk period: {highest_risk_tenure} with {tenure_analysis.loc[highest_risk_tenure, 'Churn Rate']:.1f}% churn rate, "
                       f"affecting {tenure_analysis.loc[highest_risk_tenure, 'Customer Count']:,.0f} customers")
                
            except Exception as e:
                st.error(f"Error in tenure analysis: {str(e)}")
        
        with col2:
            try:
                # Enhanced Payment Method Analysis
                payment_analysis = df.groupby('Payment Method').agg({
                    'Churn Value': ['mean', 'count'],
                    'Monthly Charges': 'mean',
                    'CLTV': 'mean'
                }).round(2)
                
                payment_analysis.columns = ['Churn Rate', 'Customer Count', 'Avg Monthly Charges', 'Avg CLTV']
                payment_analysis['Churn Rate'] = payment_analysis['Churn Rate'] * 100
                
                fig = px.bar(
                    payment_analysis.reset_index(),
                    x='Payment Method',
                    y='Churn Rate',
                    text=payment_analysis['Churn Rate'].round(1).astype(str) + '%',
                    title='Churn Rate by Payment Method',
                    color='Churn Rate',
                    color_continuous_scale='RdYlGn_r',
                    custom_data=['Customer Count', 'Avg Monthly Charges', 'Avg CLTV']
                )
                
                fig.update_traces(
                    textposition='outside',
                    hovertemplate="<br>".join([
                        "Method: %{x}",
                        "Churn Rate: %{text}",
                        "Customers: %{customdata[0]:,.0f}",
                        "Avg. Monthly: $%{customdata[1]:.2f}",
                        "Avg. CLTV: $%{customdata[2]:,.2f}",
                        "<extra></extra>"
                    ])
                )
                
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add payment method insights
                riskiest_payment = payment_analysis['Churn Rate'].idxmax()
                st.info(f"💳 {riskiest_payment} shows highest risk with {payment_analysis.loc[riskiest_payment, 'Churn Rate']:.1f}% churn rate, "
                       f"impacting {payment_analysis.loc[riskiest_payment, 'Customer Count']:,.0f} customers")
                
            except Exception as e:
                st.error(f"Error in payment method analysis: {str(e)}")
        
        # Risk Patterns and Insights
        st.subheader("🔍 Risk Patterns & Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-card">
                <h3>⚠️ High-Risk Indicators</h3>
                <ul>
                    <li>Early tenure customers (0-12 months)</li>
                    <li>Month-to-month contracts</li>
                    <li>Electronic check payments</li>
                    <li>High monthly charges without service adoption</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="insight-card">
                <h3>🛡️ Protective Factors</h3>
                <ul>
                    <li>Long-term contracts</li>
                    <li>Multiple service subscriptions</li>
                    <li>Automatic payment methods</li>
                    <li>Consistent billing history</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional Risk Metrics
        st.subheader("📈 Risk Factor Analysis")
        
        try:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Contract Type Risk
                contract_risk = df.groupby('Contract')['Churn Value'].mean() * 100
                highest_contract_risk = contract_risk.idxmax()
                st.metric(
                    "Highest Risk Contract",
                    highest_contract_risk,
                    f"{contract_risk[highest_contract_risk]:.1f}% Churn Rate"
                )
            
            with col2:
                # Service Impact
                service_cols = ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support']
                df['Service_Count'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
                service_impact = df.groupby('Service_Count')['Churn Value'].mean() * 100
                st.metric(
                    "Service Impact",
                    f"{service_impact.min():.1f}% Churn",
                    f"with {service_impact.idxmin()} services"
                )
            
            with col3:
                # Price Sensitivity
                df['Charge_Level'] = pd.qcut(df['Monthly Charges'], q=3, labels=['Low', 'Medium', 'High'])
                charge_impact = df.groupby('Charge_Level')['Churn Value'].mean() * 100
                highest_charge_risk = charge_impact.idxmax()
                st.metric(
                    "Price Sensitivity",
                    f"{highest_charge_risk} Charges",
                    f"{charge_impact[highest_charge_risk]:.1f}% Churn Rate"
                )
        
        except Exception as e:
            st.error(f"Error in risk metrics calculation: {str(e)}")
        
        # Mitigation Strategies
        st.subheader("🎯 Risk Mitigation Strategies")
        
        st.markdown("""
        <div class="insight-card">
            <h3>Recommended Actions</h3>
            <ol>
                <li><strong>Early Tenure Focus:</strong>
                    <ul>
                        <li>Implement enhanced onboarding program</li>
                        <li>Schedule regular check-ins during first 12 months</li>
                        <li>Provide special offers for contract upgrades</li>
                    </ul>
                </li>
                <li><strong>Payment Method Optimization:</strong>
                    <ul>
                        <li>Promote automatic payment enrollment</li>
                        <li>Offer incentives for switching from high-risk payment methods</li>
                        <li>Streamline payment processing</li>
                    </ul>
                </li>
                <li><strong>Service Adoption:</strong>
                    <ul>
                        <li>Create bundled service packages</li>
                        <li>Implement targeted service promotions</li>
                        <li>Develop loyalty rewards program</li>
                    </ul>
                </li>
                <li><strong>Price Sensitivity Management:</strong>
                    <ul>
                        <li>Review pricing strategy for high-risk segments</li>
                        <li>Introduce flexible payment options</li>
                        <li>Develop value-added services</li>
                    </ul>
                </li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
            
    elif page == "Financial Impact":
        st.header("Financial Impact Analysis")
        
        # Financial Overview
        st.subheader("💰 Financial Overview")
        
        # Calculate key financial metrics
        total_monthly_revenue = df['Monthly Charges'].sum()
        at_risk_revenue = df[df['Churn Value'] == 1]['Monthly Charges'].sum()
        avg_cltv = df['CLTV'].mean()
        at_risk_cltv = df[df['Churn Value'] == 1]['CLTV'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Monthly Revenue",
                f"${total_monthly_revenue:,.2f}",
                f"${df['Monthly Charges'].mean():.2f} avg/customer"
            )
            
        with col2:
            st.metric(
                "Revenue at Risk",
                f"${at_risk_revenue:,.2f}",
                f"{(at_risk_revenue/total_monthly_revenue*100):.1f}% of total"
            )
            
        with col3:
            st.metric(
                "Avg. Customer LTV",
                f"${avg_cltv:,.2f}",
                f"${df[df['Churn Value'] == 1]['CLTV'].mean() - avg_cltv:.2f} for churned"
            )
            
        with col4:
            st.metric(
                "Total LTV at Risk",
                f"${at_risk_cltv:,.2f}",
                f"{(at_risk_cltv/df['CLTV'].sum()*100):.1f}% of total"
            )
        
        # Revenue Risk Analysis
        st.subheader("📊 Revenue Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced Revenue at Risk Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=at_risk_revenue/total_monthly_revenue*100,
                title={'text': "Revenue at Risk (%)"},
                delta={'reference': 15,  # Industry benchmark
                       'increasing': {'color': "red"},
                       'decreasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ff6b6b"},
                    'steps': [
                        {'range': [0, 20], 'color': "#4ecdc4"},
                        {'range': [20, 40], 'color': "#ffe66d"},
                        {'range': [40, 100], 'color': "#ff6b6b"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 15  # Industry benchmark
                    }
                }
            ))
            
            fig.update_layout(
                annotations=[{
                    'text': 'Compared to 15% industry benchmark',
                    'x': 0.5,
                    'y': 0.25,
                    'showarrow': False,
                    'font': {'size': 10}
                }]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add risk level insight
            risk_level = "High" if at_risk_revenue/total_monthly_revenue > 0.2 else "Medium" if at_risk_revenue/total_monthly_revenue > 0.1 else "Low"
            st.info(f"⚠️ Current revenue risk level: {risk_level}")
        
        with col2:
            # Enhanced CLTV Analysis
            fig = px.box(
                df,
                x='Contract',
                y='CLTV',
                color='Churn Label',
                title='Customer Lifetime Value Distribution',
                labels={'CLTV': 'Customer Lifetime Value ($)'},
                color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'}
            )
            
            fig.update_layout(
                annotations=[{
                    'text': f'Average CLTV: ${avg_cltv:,.2f}',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0,
                    'y': -0.15,
                    'showarrow': False,
                    'font': {'size': 10}
                }]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add CLTV insight
            best_contract = df.groupby('Contract')['CLTV'].mean().idxmax()
            st.info(f"💎 {best_contract} contracts show highest average CLTV at ${df[df['Contract'] == best_contract]['CLTV'].mean():,.2f}")
        
        # Financial Impact by Segment
        st.subheader("📈 Financial Impact by Segment")
        
        try:
            # Create segment analysis
            df['Revenue_Segment'] = pd.qcut(df['Monthly Charges'], q=3, labels=['Low', 'Medium', 'High'])
            
            segment_analysis = df.groupby('Revenue_Segment').agg({
                'Monthly Charges': ['sum', 'mean'],
                'Churn Value': 'mean',
                'CLTV': 'mean',
                'CustomerID': 'count'
            }).round(2)
            
            segment_analysis.columns = ['Total Revenue', 'Avg Revenue', 'Churn Rate', 'Avg CLTV', 'Customer Count']
            segment_analysis['Churn Rate'] = segment_analysis['Churn Rate'] * 100
            segment_analysis['Revenue at Risk'] = segment_analysis['Total Revenue'] * segment_analysis['Churn Rate'] / 100
            
            # Display segment analysis
            st.dataframe(segment_analysis.style.format({
                'Total Revenue': '${:,.2f}',
                'Avg Revenue': '${:,.2f}',
                'Churn Rate': '{:.1f}%',
                'Avg CLTV': '${:,.2f}',
                'Customer Count': '{:,}',
                'Revenue at Risk': '${:,.2f}'
            }))
            
            # Add segment insights
            highest_risk_segment = segment_analysis['Revenue at Risk'].idxmax()
            st.warning(f"🚨 Highest revenue risk in {highest_risk_segment} segment: ${segment_analysis.loc[highest_risk_segment, 'Revenue at Risk']:,.2f}")
        
        except Exception as e:
            st.error(f"Error in segment analysis: {str(e)}")
        
        # Financial Insights
        st.subheader("💡 Financial Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-card">
                <h3>📊 Key Financial Findings</h3>
                <ul>
                    <li>Monthly revenue at risk: ${:,.2f}</li>
                    <li>Average CLTV gap: ${:,.2f} (loyal vs churned)</li>
                    <li>High-value segment churn rate: {:.1f}%</li>
                    <li>Revenue concentration in top segment: {:.1f}%</li>
                </ul>
            </div>
            """.format(
                at_risk_revenue,
                df[df['Churn Value'] == 0]['CLTV'].mean() - df[df['Churn Value'] == 1]['CLTV'].mean(),
                segment_analysis.loc['High', 'Churn Rate'],
                segment_analysis.loc['High', 'Total Revenue'] / total_monthly_revenue * 100
            ), unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="insight-card">
                <h3>💰 Revenue Protection Strategies</h3>
                <ul>
                    <li>Implement targeted retention programs for high-value customers</li>
                    <li>Develop upgrade paths for medium-segment customers</li>
                    <li>Create value-added services for revenue growth</li>
                    <li>Review pricing strategy for at-risk segments</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Cost of Churn Analysis
        st.subheader("💸 Cost of Churn Analysis")
        
        try:
            # Calculate churn costs
            acquisition_cost = 500  # Example customer acquisition cost
            service_cost = 100  # Example service cost per customer
            
            total_churn_cost = (
                (at_risk_revenue * 12) +  # Annual revenue loss
                (df[df['Churn Value'] == 1]['CustomerID'].count() * acquisition_cost) +  # Replacement cost
                (df[df['Churn Value'] == 1]['CustomerID'].count() * service_cost)  # Service costs
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Annual Revenue Loss",
                    f"${at_risk_revenue * 12:,.2f}",
                    f"{df[df['Churn Value'] == 1]['CustomerID'].count():,} customers"
                )
                
            with col2:
                st.metric(
                    "Customer Replacement Cost",
                    f"${df[df['Churn Value'] == 1]['CustomerID'].count() * acquisition_cost:,.2f}",
                    f"${acquisition_cost:,} per customer"
                )
                
            with col3:
                st.metric(
                    "Total Churn Impact",
                    f"${total_churn_cost:,.2f}",
                    f"${total_churn_cost/df[df['Churn Value'] == 1]['CustomerID'].count():,.2f} per customer"
                )
        
        except Exception as e:
            st.error(f"Error in churn cost analysis: {str(e)}")
            
        # Investment Recommendations
        st.markdown("""
        <div class="insight-card">
            <h3>💼 Investment Recommendations</h3>
            <ol>
                <li><strong>Retention Program Investment</strong>
                    <ul>
                        <li>Allocate ${:,.2f} (30% of annual revenue loss) to retention initiatives</li>
                        <li>Focus on high-value customer segments</li>
                        <li>Implement early warning systems</li>
                    </ul>
                </li>
                <li><strong>Service Enhancement</strong>
                    <ul>
                        <li>Invest in customer service improvements</li>
                        <li>Develop new value-added services</li>
                        <li>Upgrade technological infrastructure</li>
                    </ul>
                </li>
                <li><strong>Customer Experience</strong>
                    <ul>
                        <li>Improve onboarding process</li>
                        <li>Enhance digital platforms</li>
                        <li>Streamline customer support</li>
                    </ul>
                </li>
            </ol>
        </div>
        """.format(at_risk_revenue * 12 * 0.3), unsafe_allow_html=True)
            
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