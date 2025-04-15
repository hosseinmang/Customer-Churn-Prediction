import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
    
    # Drop location-specific columns if they exist
    location_cols = ['City', 'State', 'Country', 'Zip Code', 'Lat Long', 'Latitude', 'Longitude']
    df_processed = df_processed.drop(columns=[col for col in location_cols if col in df_processed.columns])
    
    # Rename columns if needed (handle both original and processed column names)
    column_mapping = {
        'tenure': 'YearsWithBank',
        'monthly_charges': 'MonthlyBankFees',
        'total_charges': 'TotalBalance',
        'churn': 'Churn Value',
        'churn_label': 'Churn Label',
        'Tenure Months': 'YearsWithBank',
        'Monthly Charges': 'MonthlyBankFees',
        'Total Charges': 'TotalBalance',
        'Churn': 'Churn Value'
    }
    
    # Only rename columns that exist and need renaming
    for old_col, new_col in column_mapping.items():
        if old_col in df_processed.columns and new_col not in df_processed.columns:
            df_processed = df_processed.rename(columns={old_col: new_col})
    
    # Convert YearsWithBank from months to years if needed
    if 'YearsWithBank' in df_processed.columns:
        if df_processed['YearsWithBank'].mean() > 50:  # Likely in months
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
    numerical_cols = [col for col in numerical_cols if col in df_processed.columns]
    
    if numerical_cols:
        scaler = StandardScaler()
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
        transformers['numerical_scaler'] = scaler
    
    # Ensure Churn Value exists
    if 'Churn' in df_processed.columns and 'Churn Value' not in df_processed.columns:
        df_processed['Churn Value'] = (df_processed['Churn'] == 'Yes').astype(int)
    
    if 'Churn Label' not in df_processed.columns and 'Churn' in df_processed.columns:
        df_processed['Churn Label'] = df_processed['Churn']
    
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