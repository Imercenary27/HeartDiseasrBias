# COMPLETE REWEIGHTING IMPLEMENTATION FOR HEART DISEASE BIAS MITIGATION
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class HeartDiseaseReweighting:
    """
    Reweighting implementation specifically designed for heart disease prediction
    with gender bias mitigation following Kamiran & Calders methodology.
    """
    
    def __init__(self, method='kamiran_calders'):
        """
        Initialize reweighting algorithm.
        
        Args:
            method (str): Reweighting strategy - 'kamiran_calders', 'inverse_proportion', 
                         or 'equal_opportunity'
        """
        self.method = method
        self.weights_ = None
        self.reweight_factors_ = {}
        
    def fit(self, X, y, sensitive_attr):
        """
        Calculate optimal sample weights for bias mitigation.
        
        Args:
            X (pd.DataFrame): Training features
            y (array-like): Training labels (0=no disease, 1=disease)
            sensitive_attr (array-like): Protected attribute (0=female, 1=male)
        
        Returns:
            self: Fitted reweighter
        """
        y = np.array(y)
        sensitive_attr = np.array(sensitive_attr)
        
        n_total = len(y)
        weights = np.ones(len(y))
        
        if self.method == 'kamiran_calders':
            # Original Kamiran & Calders reweighting formula
            for group in [0, 1]:  # Female, Male
                for cls in [0, 1]:  # No disease, Disease
                    group_mask = (sensitive_attr == group)
                    class_mask = (y == cls)
                    group_class_mask = group_mask & class_mask
                    
                    n_group = np.sum(group_mask)
                    n_class = np.sum(class_mask)
                    n_group_class = np.sum(group_class_mask)
                    
                    if n_group_class > 0:
                        weight_factor = (n_class * n_group) / (n_total * n_group_class)
                        weights[group_class_mask] = weight_factor
                        self.reweight_factors_[(group, cls)] = weight_factor
                        
        elif self.method == 'inverse_proportion':
            # Weight inversely proportional to group-class frequency
            for group in [0, 1]:
                for cls in [0, 1]:
                    mask = (sensitive_attr == group) & (y == cls)
                    count = np.sum(mask)
                    if count > 0:
                        weight = n_total / (4 * count)  # 4 = number of combinations
                        weights[mask] = weight
                        
        elif self.method == 'equal_opportunity':
            # Balance positive prediction rates across groups
            total_pos = np.sum(y == 1)
            for group in [0, 1]:
                group_mask = (sensitive_attr == group)
                group_pos_mask = group_mask & (y == 1)
                group_neg_mask = group_mask & (y == 0)
                
                if np.sum(group_pos_mask) > 0:
                    pos_weight = total_pos / (2 * np.sum(group_pos_mask))
                    weights[group_pos_mask] = pos_weight
                    
                if np.sum(group_neg_mask) > 0:
                    neg_weight = (n_total - total_pos) / (2 * np.sum(group_neg_mask))
                    weights[group_neg_mask] = neg_weight
        
        # Normalize weights to maintain total sample importance
        weights = weights * n_total / np.sum(weights)
        self.weights_ = weights
        return self
        
    def get_weights(self):
        """Return calculated sample weights."""
        if self.weights_ is None:
            raise ValueError("Reweighter must be fitted before getting weights")
        return self.weights_
    
    def get_weight_summary(self, sensitive_attr, y):
        """Print summary of weight distribution."""
        print(f"=== {self.method.upper()} REWEIGHTING SUMMARY ===")
        for group in [0, 1]:
            group_name = "Female" if group == 0 else "Male"
            for cls in [0, 1]:
                outcome = "No Disease" if cls == 0 else "Disease"
                mask = (sensitive_attr == group) & (y == cls)
                if np.any(mask):
                    weight = self.weights_[mask][0]
                    count = np.sum(mask)
                    print(f"{group_name}, {outcome}: {count} samples, weight={weight:.3f}")

def calculate_bias_metrics(y_true, y_pred, sensitive_attr):
    """Calculate comprehensive bias and performance metrics."""
    metrics = {}
    
    # Overall performance
    metrics['accuracy'] = np.mean(y_true == y_pred)
    
    # Group-specific metrics
    for group in [0, 1]:
        group_name = "female" if group == 0 else "male"
        group_mask = (sensitive_attr == group)
        
        if np.any(group_mask):
            metrics[f'accuracy_{group_name}'] = np.mean(y_true[group_mask] == y_pred[group_mask])
            metrics[f'selection_rate_{group_name}'] = np.mean(y_pred[group_mask])
            
            # True positive rate (sensitivity)
            pos_mask = group_mask & (y_true == 1)
            if np.any(pos_mask):
                metrics[f'tpr_{group_name}'] = np.mean(y_pred[pos_mask])
            else:
                metrics[f'tpr_{group_name}'] = 0
    
    # Fairness metrics
    sr_female = metrics.get('selection_rate_female', 0)
    sr_male = metrics.get('selection_rate_male', 0)
    metrics['demographic_parity_diff'] = abs(sr_male - sr_female)
    
    tpr_female = metrics.get('tpr_female', 0)
    tpr_male = metrics.get('tpr_male', 0)
    metrics['equalized_odds_diff'] = abs(tpr_male - tpr_female)
    
    return metrics

# ===== MAIN IMPLEMENTATION =====
def implement_reweighting_bias_mitigation(data_file_or_df):
    """
    Complete pipeline for implementing reweighting bias mitigation.
    
    Args:
        data_file_or_df: Path to CSV file or pandas DataFrame with heart disease data
    
    Returns:
        dict: Comprehensive results comparing different reweighting methods
    """
    
    # Load and prepare data
    if isinstance(data_file_or_df, str):
        df = pd.read_csv(data_file_or_df)
    else:
        df = data_file_or_df.copy()
    
    # Assume standard heart disease dataset format
    # Adjust column names based on your specific dataset
    feature_cols = [col for col in df.columns if col not in ['target', 'sex']]
    X = df[feature_cols]
    y = df['target'] if 'target' in df.columns else df['num']
    A = df['sex']  # Protected attribute
    
    print("=== HEART DISEASE REWEIGHTING BIAS MITIGATION ===")
    print(f"Dataset: {len(df)} patients, {len(feature_cols)} features")
    print(f"Disease prevalence: {np.mean(y):.3f}")
    print(f"Gender distribution: Female={np.sum(A==0)}, Male={np.sum(A==1)}")
    
    # Check initial bias
    initial_bias = pd.crosstab(A, y, normalize='index')
    print(f"Initial bias: Female disease rate={initial_bias.loc[0,1]:.3f}, Male disease rate={initial_bias.loc[1,1]:.3f}")
    
    # Train/test split
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A, test_size=0.25, random_state=42, stratify=y
    )
    
    # Test different reweighting methods
    methods = ['kamiran_calders', 'inverse_proportion', 'equal_opportunity']
    results = {}
    
    # Baseline model (no reweighting)
    baseline_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    results['baseline'] = calculate_bias_metrics(y_test, baseline_pred, A_test)
    
    print(f"\n=== TESTING REWEIGHTING METHODS ===")
    
    # Test each reweighting method
    for method in methods:
        print(f"\n--- {method.replace('_', ' ').title()} Method ---")
        
        # Apply reweighting
        reweighter = HeartDiseaseReweighting(method=method)
        reweighter.fit(X_train, y_train, A_train)
        sample_weights = reweighter.get_weights()
        reweighter.get_weight_summary(A_train, y_train)
        
        # Train reweighted model
        reweighted_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        reweighted_model.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        reweighted_pred = reweighted_model.predict(X_test)
        
        # Calculate metrics
        results[method] = calculate_bias_metrics(y_test, reweighted_pred, A_test)
        results[method]['sample_weights'] = sample_weights
    
    # Create comparison summary
    comparison_df = []
    for method_name, metrics in results.items():
        if method_name != 'sample_weights':
            row = {
                'Method': method_name.replace('_', ' ').title(),
                'Accuracy': metrics['accuracy'],
                'DP_Difference': metrics['demographic_parity_diff'],
                'EO_Difference': metrics['equalized_odds_diff'],
                'Female_Accuracy': metrics.get('accuracy_female', 0),
                'Male_Accuracy': metrics.get('accuracy_male', 0)
            }
            comparison_df.append(row)
    
    comparison_df = pd.DataFrame(comparison_df)
    
    print(f"\n=== REWEIGHTING EFFECTIVENESS SUMMARY ===")
    print(comparison_df.round(3).to_string(index=False))
    
    # Calculate improvements over baseline
    baseline_dp = results['baseline']['demographic_parity_diff']
    baseline_eo = results['baseline']['equalized_odds_diff']
    
    print(f"\n=== BIAS REDUCTION EFFECTIVENESS ===")
    for method in methods:
        method_display = method.replace('_', ' ').title()
        dp_reduction = ((baseline_dp - results[method]['demographic_parity_diff']) / baseline_dp) * 100
        eo_reduction = ((baseline_eo - results[method]['equalized_odds_diff']) / baseline_eo) * 100
        acc_change = (results[method]['accuracy'] - results['baseline']['accuracy']) * 100
        
        print(f"{method_display}:")
        print(f"  Demographic Parity Reduction: {dp_reduction:.1f}%")
        print(f"  Equalized Odds Reduction: {eo_reduction:.1f}%")
        print(f"  Accuracy Change: {acc_change:+.1f}%")
    
    return {
        'results': results,
        'comparison_df': comparison_df,
        'best_method': comparison_df.loc[comparison_df['DP_Difference'].idxmin(), 'Method']
    }

# Example usage with your heart disease dataset:
results = implement_reweighting_bias_mitigation('heart_disease_uci.csv')
