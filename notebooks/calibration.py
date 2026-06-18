import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt

df = pd.read_csv('../data/diabetic_features.csv')
X = df.drop(columns=['readmitted_30'])
y = df['readmitted_30']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with open('../models/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Apply Platt scaling
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated.fit(X_train, y_train)

# Compare before vs after
y_prob_raw = model.predict_proba(X_test)[:, 1]
y_prob_cal = calibrated.predict_proba(X_test)[:, 1]

print("Before calibration:")
print("  AUROC:", round(roc_auc_score(y_test, y_prob_raw), 4))
print("  Brier:", round(brier_score_loss(y_test, y_prob_raw), 4))

print("\nAfter calibration:")
print("  AUROC:", round(roc_auc_score(y_test, y_prob_cal), 4))
print("  Brier:", round(brier_score_loss(y_test, y_prob_cal), 4))

# Plot both curves
fig, ax = plt.subplots(figsize=(6, 6))
for probs, label in [(y_prob_raw, 'XGBoost raw'), (y_prob_cal, 'XGBoost calibrated')]:
    frac, mean = calibration_curve(y_test, probs, n_bins=10)
    ax.plot(mean, frac, marker='o', label=label)
ax.plot([0,1],[0,1], linestyle='--', label='Perfect')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_title('Calibration: Before vs After Platt Scaling')
ax.legend()
plt.tight_layout()
plt.savefig('../models/calibration_comparison.png')
print("\nPlot saved.")

with open('../models/xgb_calibrated.pkl', 'wb') as f:
    pickle.dump(calibrated, f)
print("Calibrated model saved.")