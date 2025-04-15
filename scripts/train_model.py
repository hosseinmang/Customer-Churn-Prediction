import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from preprocessing import preprocess_data, prepare_features

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def train_model():
    """Train and save the churn prediction model"""
    try:
        print("Loading data...")
        # Load data
        data_path = os.path.join(PROJECT_ROOT, 'data', 'Telco_customer_churn.xlsx')
        raw_df = pd.read_excel(data_path)
        print(f"Raw data type: {type(raw_df)}")
        print(f"Raw data shape: {raw_df.shape}")
        
        print("\nPreprocessing data...")
        # Preprocess the data
        df = preprocess_data(raw_df)
        print(f"Preprocessed data type: {type(df)}")
        print(f"Preprocessed data shape: {df.shape}")
        print(f"Preprocessed columns: {df.columns.tolist()}")
        
        print("\nPreparing features...")
        # Prepare features for modeling
        X = prepare_features(df)
        y = df['Churn Label'].map({'Yes': 1, 'No': 0})
        
        print(f"Feature shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        print("\nSplitting data...")
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Scaling features...")
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        print("Training model...")
        # Train the model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        print("Saving model...")
        # Save the model and scaler
        model_dir = os.path.join(PROJECT_ROOT, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(model, os.path.join(model_dir, 'churn_model.joblib'))
        joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
        
        # Print model performance
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        print(f"Model Training Score: {train_score:.4f}")
        print(f"Model Test Score: {test_score:.4f}")
        
        # Save feature names for later use
        feature_names = list(X.columns)
        joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.joblib'))
        
        print("Model training completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    train_model() 