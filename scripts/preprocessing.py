import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(df):
    """
    Preprocess the customer churn data for modeling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw customer data
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data ready for modeling
    dict
        Dictionary containing fitted encoders and scalers
    """
    # Create a copy
    df_processed = df.copy()
    
    # Drop location-specific columns
    location_cols = ['City', 'State', 'Country', 'Zip Code', 'Lat Long', 'Latitude', 'Longitude']
    df_processed = df_processed.drop(columns=location_cols, errors='ignore')
    
    # Handle Total Charges with improved missing value treatment
    total_charges_null = df_processed['Total Charges'].isnull().sum()
    if total_charges_null > 0:
        logger.info(f"Found {total_charges_null} missing values in Total Charges")
        
        # Convert to numeric, keeping track of conversion errors
        df_processed['Total Charges'] = pd.to_numeric(df_processed['Total Charges'], errors='coerce')
        
        # Calculate imputation value based on tenure and monthly charges
        df_processed['Total Charges'] = df_processed.apply(
            lambda row: row['Monthly Charges'] * row['Tenure Months'] 
            if pd.isnull(row['Total Charges']) 
            else row['Total Charges'],
            axis=1
        )
        
        logger.info("Imputed missing Total Charges using Monthly Charges * Tenure")
    
    # Rename columns to match banking context
    column_mapping = {
        'Monthly Charges': 'MonthlyBankFees',
        'Total Charges': 'TotalBalance',
        'Tenure Months': 'YearsWithBank',
        'Phone Service': 'DebitCard',
        'Multiple Lines': 'CreditCard',
        'Internet Service': 'OnlineBanking',
        'Online Security': 'SecureLogin2FA',
        'Online Backup': 'AutomaticSavings',
        'Device Protection': 'FraudProtection',
        'Tech Support': 'CustomerSupport',
        'Streaming TV': 'BillPay',
        'Streaming Movies': 'MobilePayments'
    }
    df_processed = df_processed.rename(columns=column_mapping)
    
    # Convert YearsWithBank from months to years
    df_processed['YearsWithBank'] = df_processed['YearsWithBank'] / 12
    
    # Initialize dictionary to store transformers
    transformers = {}
    
    # Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in ['CustomerID', 'Churn Label', 'Churn Reason']]
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        transformers[f'{col}_encoder'] = le
    
    # Scale numerical features
    numerical_cols = ['YearsWithBank', 'MonthlyBankFees', 'TotalBalance']
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    transformers['numerical_scaler'] = scaler
    
    # Verify no missing values remain
    missing_values = df_processed[numerical_cols].isnull().sum()
    if missing_values.any():
        logger.warning(f"Remaining missing values after preprocessing: {missing_values}")
    
    return df_processed, transformers

def prepare_features(df):
    """
    Prepare feature matrix X and target variable y
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed data
        
    Returns:
    --------
    numpy.ndarray
        Feature matrix X
    numpy.ndarray
        Target variable y
    list
        Feature names
    """
    # Define features to use
    feature_cols = [
        'YearsWithBank', 'MonthlyBankFees', 'TotalBalance',
        'DebitCard', 'CreditCard', 'OnlineBanking', 'SecureLogin2FA',
        'AutomaticSavings', 'FraudProtection', 'CustomerSupport',
        'BillPay', 'MobilePayments', 'Contract', 'PaperlessBilling',
        'PaymentMethod'
    ]
    
    # Remove any columns that don't exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].values
    y = df['Churn Value'].values
    
    return X, y, feature_cols