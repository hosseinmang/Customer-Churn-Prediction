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
    
    # Define column types
    numerical_cols = ['Count', 'Latitude', 'Longitude', 'Senior Citizen', 
                     'Tenure Months', 'Monthly Charges', 'Total Charges',
                     'Churn Value', 'Churn Score', 'CLTV']
    
    categorical_cols = ['Country', 'State', 'City', 'Zip Code', 'Gender', 
                       'Partner', 'Dependents', 'Phone Service', 'Multiple Lines',
                       'Internet Service', 'Online Security', 'Online Backup',
                       'Device Protection', 'Tech Support', 'Streaming TV',
                       'Streaming Movies', 'Contract', 'Paperless Billing',
                       'Payment Method', 'Churn Label', 'Churn Reason']
    
    # Initialize transformers dictionary
    transformers = {}
    
    # Handle numerical features
    for col in numerical_cols:
        if col in df_processed.columns:
            # Convert to numeric, coerce errors to NaN
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Scale numerical features (excluding Churn Value, Churn Score which are targets)
    scale_cols = [col for col in numerical_cols if col in df_processed.columns 
                 and col not in ['Churn Value', 'Churn Score', 'CLTV']]
    
    if scale_cols:
        scaler = StandardScaler()
        # Fill NaN values with mean before scaling
        df_processed[scale_cols] = df_processed[scale_cols].fillna(df_processed[scale_cols].mean())
        df_processed[scale_cols] = scaler.fit_transform(df_processed[scale_cols])
        transformers['numerical_scaler'] = scaler
    
    # Handle categorical features
    for col in categorical_cols:
        if col in df_processed.columns:
            # Fill NaN values with 'Unknown'
            df_processed[col] = df_processed[col].fillna('Unknown')
            # Convert everything to string first
            df_processed[col] = df_processed[col].astype(str)
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            transformers[f'{col}_encoder'] = le
    
    # Special handling for Churn Value if it doesn't exist
    if 'Churn Value' not in df_processed.columns and 'Churn Label' in df_processed.columns:
        df_processed['Churn Value'] = (df_processed['Churn Label'] == '1').astype(int)
    
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
    # Define core features to use (excluding location and ID columns)
    feature_cols = [
        'Tenure Months', 'Monthly Charges', 'Total Charges',
        'Gender', 'Senior Citizen', 'Partner', 'Dependents',
        'Phone Service', 'Multiple Lines', 'Internet Service',
        'Online Security', 'Online Backup', 'Device Protection',
        'Tech Support', 'Streaming TV', 'Streaming Movies',
        'Contract', 'Paperless Billing', 'Payment Method'
    ]
    
    # Remove any columns that don't exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].values
    y = df['Churn Value'].values if 'Churn Value' in df.columns else None
    
    return X, y, feature_cols