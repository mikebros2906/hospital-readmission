import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import matplotlib.pyplot as plt

df = pd.read_csv('../data/diabetic_features.csv')

X = df.drop(columns=['readmitted_30'])
y = df['readmitted_30']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
print("Positive rate in test:", y_test.mean().round(3))

# Class imbalance ratio
scale = (y_train == 0).sum() / (y_train == 1).sum()
print("scale_pos_weight:", round(scale, 2))

# XGBoost model
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=50)

# Evaluate
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auroc = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

print("\nAUROC:", round(auroc, 4))
print("Brier Score:", round(brier, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calibration curve
fraction_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
plt.figure(figsize=(6, 6))
plt.plot(mean_pred, fraction_pos, marker='o', label='XGBoost')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
plt.legend()
plt.tight_layout()
plt.savefig('../models/calibration_curve.png')
print("\nCalibration curve saved to models/")

# Save model
import pickle
with open('../models/xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to models/xgb_model.pkl")