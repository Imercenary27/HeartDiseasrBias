# Complete bias analysis code for heart disease dataset - Final Version
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from fairlearn.metrics import (
    MetricFrame, selection_rate,
    demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ===== PROBLEM FORMULATION =====
print("=== HEART DISEASE BIAS ANALYSIS FOR SDG 3 ===")
print("Problem: Predicting heart disease without sex-based discrimination")
print("Objective: Build fair AI model for equitable healthcare")
print("SDG Relevance: Advances Target 3.4 - reduce premature mortality from NCDs")

# ===== DATA COLLECTION & EXPLORATION =====
df = pd.read_csv('heart_disease_uci.csv')

# Data preprocessing
df_clean = df.dropna(subset=['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca'])
df_clean['sex_encoded'] = (df_clean['sex'] == 'Male').astype(int)  # 0=Female, 1=Male
df_clean['target'] = (df_clean['num'] > 0).astype(int)  # 0=no disease, 1=disease

# Feature selection
numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
X = df_clean[numeric_features + ['sex_encoded']]
y = df_clean['target']
A = df_clean['sex_encoded']  # Protected attribute

print(f"\nDataset: {X.shape[0]} patients, {X.shape[1]} features")
print(f"Sex distribution: Female={sum(A==0)} ({sum(A==0)/len(A)*100:.1f}%), Male={sum(A==1)} ({sum(A==1)/len(A)*100:.1f}%)")

# Exploratory analysis
prevalence_by_sex = pd.crosstab(df_clean['sex'], df_clean['target'], normalize='index')
print(f"\nHeart disease prevalence:")
print(f"Female: {prevalence_by_sex.loc['Female', 1]:.3f}")
print(f"Male: {prevalence_by_sex.loc['Male', 1]:.3f}")

# ===== BIAS IDENTIFICATION =====
print(f"\n=== BIAS IDENTIFICATION ===")
print(f"1. REPRESENTATION BIAS: {abs(sum(A==0)/len(A) - 0.5)*100:.1f}% deviation from 50-50 split")
print(f"2. OUTCOME BIAS: {abs(prevalence_by_sex.loc['Female', 1] - prevalence_by_sex.loc['Male', 1]):.3f} difference in disease rates")

# Train/test split
X_tr, X_te, y_tr, y_te, A_tr, A_te = train_test_split(
    X, y, A, stratify=y, test_size=0.25, random_state=42
)

# ===== BASELINE MODEL =====
baseline_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
baseline_model.fit(X_tr, y_tr)
baseline_pred = baseline_model.predict(X_te)

# Baseline fairness evaluation
baseline_metrics = {'accuracy': accuracy_score, 'selection_rate': selection_rate}
baseline_mf = MetricFrame(
    metrics=baseline_metrics, 
    y_true=y_te, 
    y_pred=baseline_pred, 
    sensitive_features=A_te
)

baseline_dp_diff = demographic_parity_difference(y_te, baseline_pred, sensitive_features=A_te)
baseline_dp_ratio = demographic_parity_ratio(y_te, baseline_pred, sensitive_features=A_te)
baseline_eo_diff = equalized_odds_difference(y_te, baseline_pred, sensitive_features=A_te)

print(f"\n=== BASELINE MODEL RESULTS ===")
print(f"Overall Accuracy: {baseline_mf.overall['accuracy']:.3f}")
print(f"Accuracy by Sex: Female={baseline_mf.by_group['accuracy'][0]:.3f}, Male={baseline_mf.by_group['accuracy'][1]:.3f}")
print(f"Selection Rate by Sex: Female={baseline_mf.by_group['selection_rate'][0]:.3f}, Male={baseline_mf.by_group['selection_rate'][1]:.3f}")
print(f"Demographic Parity Difference: {baseline_dp_diff:.3f}")
print(f"Equalized Odds Difference: {baseline_eo_diff:.3f}")
print(f"3. ALGORITHMIC BIAS: DP difference of {abs(baseline_dp_diff):.3f} indicates unfair predictions")

# ===== BIAS MITIGATION TECHNIQUE 1: SMOTE =====
print(f"\n=== BIAS MITIGATION 1: SMOTE OVERSAMPLING ===")
smote = SMOTE(random_state=42)
X_tr_smote, y_tr_smote = smote.fit_resample(X_tr, y_tr)

smote_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
smote_model.fit(X_tr_smote, y_tr_smote)
smote_pred = smote_model.predict(X_te)

smote_mf = MetricFrame(
    metrics=baseline_metrics, 
    y_true=y_te, 
    y_pred=smote_pred, 
    sensitive_features=A_te
)
smote_dp_diff = demographic_parity_difference(y_te, smote_pred, sensitive_features=A_te)
smote_eo_diff = equalized_odds_difference(y_te, smote_pred, sensitive_features=A_te)

print(f"SMOTE Results:")
print(f"Overall Accuracy: {smote_mf.overall['accuracy']:.3f}")
print(f"Demographic Parity Difference: {smote_dp_diff:.3f}")
print(f"Improvement: {((abs(baseline_dp_diff)-abs(smote_dp_diff))/abs(baseline_dp_diff)*100):.1f}%")

# ===== BIAS MITIGATION TECHNIQUE 2: CLASS WEIGHT BALANCING =====
print(f"\n=== BIAS MITIGATION 2: CLASS WEIGHT BALANCING ===")
balanced_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])
balanced_model.fit(X_tr, y_tr)
balanced_pred = balanced_model.predict(X_te)

balanced_mf = MetricFrame(
    metrics=baseline_metrics, 
    y_true=y_te, 
    y_pred=balanced_pred, 
    sensitive_features=A_te
)
balanced_dp_diff = demographic_parity_difference(y_te, balanced_pred, sensitive_features=A_te)
balanced_eo_diff = equalized_odds_difference(y_te, balanced_pred, sensitive_features=A_te)

print(f"Balanced Results:")
print(f"Overall Accuracy: {balanced_mf.overall['accuracy']:.3f}")
print(f"Demographic Parity Difference: {balanced_dp_diff:.3f}")
print(f"Improvement: {((abs(baseline_dp_diff)-abs(balanced_dp_diff))/abs(baseline_dp_diff)*100):.1f}%")

# ===== COMPARATIVE ANALYSIS =====
comparison_results = pd.DataFrame({
    'Method': ['Baseline', 'SMOTE', 'Balanced'],
    'Accuracy': [
        baseline_mf.overall['accuracy'], 
        smote_mf.overall['accuracy'], 
        balanced_mf.overall['accuracy']
    ],
    'DP_Difference': [abs(baseline_dp_diff), abs(smote_dp_diff), abs(balanced_dp_diff)],
    'EO_Difference': [abs(baseline_eo_diff), abs(smote_eo_diff), abs(balanced_eo_diff)]
})

print(f"\n=== COMPARATIVE ANALYSIS ===")
print(comparison_results.round(3))

# Best model selection
best_idx = comparison_results['DP_Difference'].idxmin()
best_method = comparison_results.iloc[best_idx]['Method']

print(f"\n=== RECOMMENDATIONS ===")
print(f"Best Method: {best_method}")
print(f"Achieves lowest bias: {comparison_results.iloc[best_idx]['DP_Difference']:.3f}")
print(f"With accuracy: {comparison_results.iloc[best_idx]['Accuracy']:.3f}")

print(f"\n=== SDG 3 CONTRIBUTION ===")
print("This analysis advances SDG 3 by:")
print("• Identifying sex bias in cardiovascular risk prediction")
print("• Implementing fairness-aware machine learning techniques")
print("• Ensuring equitable healthcare AI for all patients")
print("• Reducing health disparities through unbiased algorithms")
print("• Supporting Target 3.4: reduce premature mortality from NCDs")

# Export results
comparison_results.to_csv('heart_disease_bias_results.csv', index=False)
print(f"\nResults exported for visualization and reporting")
