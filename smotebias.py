# COMPLETE SMOTE BIAS MITIGATION IMPLEMENTATION
# Heart Disease Prediction with Gender Bias Mitigation

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    MetricFrame, selection_rate,
    demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def implement_smote_bias_mitigation(data_file):
    """
    Complete SMOTE bias mitigation implementation for heart disease prediction
    
    Args:
        data_file: Path to heart_disease_uci.csv file
    
    Returns:
        Dictionary with baseline and SMOTE model results
    """
    
    # 1. DATA PREPROCESSING
    df = pd.read_csv(data_file)
    df_clean = df[df['dataset'] == 'Cleveland'].copy()
    
    # Create binary target and sex encoding
    df_clean['target'] = (df_clean['num'] > 0).astype(int)
    df_clean['sex_encoded'] = (df_clean['sex'] == 'Male').astype(int)
    
    # Select features and handle missing values
    numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    df_processed = df_clean.dropna(subset=numeric_features + ['sex_encoded', 'target'])
    
    X = df_processed[numeric_features + ['sex_encoded']]
    y = df_processed['target']
    A = df_processed['sex_encoded']  # Protected attribute
    
    # 2. TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A, test_size=0.25, random_state=42, stratify=y
    )
    
    # 3. BASELINE MODEL
    baseline_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    
    # 4. SMOTE BIAS MITIGATION
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    smote_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    smote_model.fit(X_train_smote, y_train_smote)
    smote_pred = smote_model.predict(X_test)
    
    # 5. EVALUATION
    def evaluate_model(y_true, y_pred, sensitive_features, model_name):
        metrics = {'accuracy': accuracy_score, 'selection_rate': selection_rate}
        mf = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, 
                        sensitive_features=sensitive_features)
        
        dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
        dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features)
        eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
        
        return {
            'model': model_name,
            'overall_accuracy': mf.overall['accuracy'],
            'female_accuracy': mf.by_group['accuracy'].iloc[0],
            'male_accuracy': mf.by_group['accuracy'].iloc[1],
            'female_selection_rate': mf.by_group['selection_rate'].iloc[0],
            'male_selection_rate': mf.by_group['selection_rate'].iloc[1],
            'dp_difference': abs(dp_diff),
            'dp_ratio': dp_ratio,
            'eo_difference': abs(eo_diff)
        }
    
    baseline_results = evaluate_model(y_test, baseline_pred, A_test, 'Baseline')
    smote_results = evaluate_model(y_test, smote_pred, A_test, 'SMOTE')
    
    # 6. RETURN RESULTS
    return {
        'baseline': baseline_results,
        'smote': smote_results,
        'improvements': {
            'dp_improvement_pct': ((baseline_results['dp_difference'] - smote_results['dp_difference']) 
                                 / baseline_results['dp_difference']) * 100,
            'eo_improvement_pct': ((baseline_results['eo_difference'] - smote_results['eo_difference']) 
                                 / baseline_results['eo_difference']) * 100,
            'accuracy_change': smote_results['overall_accuracy'] - baseline_results['overall_accuracy']
        }
    }

# USAGE EXAMPLE:
results = implement_smote_bias_mitigation('heart_disease_uci.csv')
print("Baseline DP Difference:", results['baseline']['dp_difference'])
print("SMOTE DP Difference:", results['smote']['dp_difference'])
print("Improvement:", results['improvements']['dp_improvement_pct'], "%")
