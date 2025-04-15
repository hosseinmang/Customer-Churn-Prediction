import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap

class ChurnPredictor:
    def __init__(self, model_type='xgboost'):
        """
        Initialize the ChurnPredictor with specified model type
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('logistic', 'random_forest', or 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'xgboost':
            self.model = xgb.XGBClassifier(random_state=42)
        else:
            raise ValueError("model_type must be 'logistic', 'random_forest', or 'xgboost'")
    
    def train(self, X, y, feature_names=None):
        """
        Train the model on the given data
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Target variable
        feature_names : list
            List of feature names
        """
        self.feature_names = feature_names
        self.model.fit(X, y)
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            True labels
            
        Returns:
        --------
        dict
            Dictionary containing various performance metrics
        """
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        return metrics
    
    def get_feature_importance(self, X):
        """
        Calculate feature importance using SHAP values
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
            
        Returns:
        --------
        numpy.ndarray
            SHAP values for feature importance
        """
        explainer = shap.TreeExplainer(self.model) if self.model_type in ['random_forest', 'xgboost'] \
                   else shap.LinearExplainer(self.model, X)
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return shap_values
    
def train_evaluate_model(X, y, feature_names, model_type='xgboost', test_size=0.2):
    """
    Train and evaluate a model with cross-validation
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target variable
    feature_names : list
        List of feature names
    model_type : str
        Type of model to use
    test_size : float
        Proportion of dataset to include in the test split
        
    Returns:
    --------
    ChurnPredictor
        Trained model
    dict
        Dictionary containing performance metrics
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Initialize and train the model
    model = ChurnPredictor(model_type)
    model.train(X_train, y_train, feature_names)
    
    # Get performance metrics
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model.model, X, y, cv=5)
    
    metrics = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    return model, metrics