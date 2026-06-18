import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

df = pd.read_csv('../data/diabetic_features.csv')
X = df.drop(columns=['readmitted_30'])
y = df['readmitted_30']

with open('../models/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# SHAP explainer - use a sample for speed
X_sample = X.sample(2000, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# 1. Global feature importance
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig('../models/shap_summary.png', bbox_inches='tight')
plt.close()
print("SHAP summary plot saved.")

# 2. Waterfall for single high-risk patient
high_risk_idx = X_sample.index[np.argmax(shap_values.sum(axis=1))]
shap_exp = shap.Explanation(
    values=shap_values[X_sample.index.get_loc(high_risk_idx)],
    base_values=explainer.expected_value,
    data=X_sample.loc[high_risk_idx].values,
    feature_names=X.columns.tolist()
)
plt.figure()
shap.plots.waterfall(shap_exp, show=False)
plt.tight_layout()
plt.savefig('../models/shap_waterfall.png', bbox_inches='tight')
plt.close()
print("SHAP waterfall plot saved.")

# 3. Top 10 most important features
mean_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 features by SHAP importance:")
print(mean_shap.head(10))