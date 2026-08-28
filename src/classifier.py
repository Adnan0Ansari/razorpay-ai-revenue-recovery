"""
Trains a model to predict whether a retry on a failed transaction
will succeed, based on transaction features.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib  # for saving the trained model to disk

df = pd.read_csv("data/failed_transactions.csv")

# --- Encode categorical columns into numbers, since ML models need numeric input ---
categorical_cols = ["payment_method", "bank", "failure_reason"]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le  # we save these so we can decode predictions back to human-readable labels later

feature_cols = [
    "amount", "hour_of_day", "is_repeat_customer", "previous_failed_attempts",
    "payment_method_enc", "bank_enc", "failure_reason_enc",
]

X = df[feature_cols]
y = df["retry_successful"]

# --- Split into training and test sets ---
# WHY: we train on 80% of the data and test on the other 20% the model has NEVER
# seen, to check if it actually learned general patterns rather than memorizing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# --- Feature importance: which factors drive the prediction most ---
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)

# --- Save the model and encoders so other scripts can reuse them ---
joblib.dump(model, "src/retry_model.pkl")
joblib.dump(encoders, "src/encoders.pkl")
print("\nSaved model -> src/retry_model.pkl")