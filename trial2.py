# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Data
url_mat = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv"
url_por = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-por.csv"
mat = pd.read_csv(url_mat, sep=';')
por = pd.read_csv(url_por, sep=';')

# 2. Combine datasets
data = pd.concat([mat, por], ignore_index=True)
print(f"Combined dataset shape: {data.shape}")

# 3. EDA: Structure and basic stats
print(data.info())
print(data.describe())

# 4. Visualizations
sns.countplot(x='sex', data=data)
plt.title("Gender Distribution")
plt.show()

sns.countplot(x='address', data=data)
plt.title("Urban vs Rural")
plt.show()

sns.boxplot(x='sex', y='G3', data=data)
plt.title("Final Grade by Gender")
plt.show()

# 5. Identify Bias: Example—Gender gap in pass rates
data['pass'] = (data['G3'] >= 10).astype(int)
rate_f = data.loc[data.sex=='F','pass'].mean()
rate_m = data.loc[data.sex=='M','pass'].mean()
chi2, p, dof, expected = stats.chi2_contingency(
    pd.crosstab(data.sex, data.pass)
)
print(f"Female pass rate: {rate_f:.2f}, Male pass rate: {rate_m:.2f}, χ² p‐value={p:.3f}")

# 6. Preprocess for modeling
#    Drop original grades and target G3; encode categorical variables
X = data.drop(columns=['G1','G2','G3','pass'])
y = data['pass']
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 7. Baseline model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Baseline Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8. Bias Mitigation Method 1: Stratified Resampling to balance gender
train = pd.concat([X_train, y_train], axis=1)
# add sex column back
train['sex'] = data.loc[X_train.index,'sex']
groups = []
for sex in ['F','M']:
    grp = train[train.sex==sex]
    grp_res = resample(grp, 
                       replace=True,
                       n_samples=len(train)//2,
                       random_state=42)
    groups.append(grp_res)
balanced = pd.concat(groups)
y_bal = balanced['pass']
X_bal = balanced.drop(columns=['pass','sex'])

clf2 = RandomForestClassifier(random_state=42)
clf2.fit(X_bal, y_bal)
y2 = clf2.predict(X_test)
print("Resampled Accuracy:", accuracy_score(y_test, y2))
print(classification_report(y_test, y2))

# 9. Bias Mitigation Method 2: Threshold Adjustment for demographic parity
probs = clf.predict_proba(X_test)[:,1]
# pick separate thresholds
thr_F = np.percentile(probs[data.loc[X_test.index,'sex']=='F'], 50)
thr_M = np.percentile(probs[data.loc[X_test.index,'sex']=='M'], 50)
y_adj = []
for i,p in enumerate(probs):
    sex = data.loc[X_test.index[i],'sex']
    y_adj.append(int(p >= (thr_F if sex=='F' else thr_M)))
print("Threshold‐Adjusted Accuracy:", accuracy_score(y_test, y_adj))
print(classification_report(y_test, y_adj))

# 10. Summary of Results
results = {
    "Model": ["Baseline", "Resampled", "Threshold_Adjusted"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y2),
        accuracy_score(y_test, y_adj),
    ],
    "Female_Pass_Rate": [
        rate_f,
        np.mean(y2)[data.loc[X_test.index,'sex']=='F'].mean() if False else None,
        None
    ]
}
print(pd.DataFrame(results))
