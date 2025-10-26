import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
import joblib

# Load data
df = pd.read_csv("synthetic_employee_weeks_20k.csv")

# Prepare regression data
X_reg = df.drop(columns=["productivity_index", "burnout_risk", "burnout_score"])
y_reg = df["productivity_index"]

# Prepare classification data
X_clf = df.drop(columns=["productivity_index", "burnout_score", "burnout_risk"])
y_clf = df["burnout_risk"].map({"High": 0, "Low": 1, "Medium": 2})

# Split
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Train models
print("Training productivity regression model...")
reg_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
reg_model.fit(Xr_train, yr_train)

print("Training burnout classification model...")
clf_model = RandomForestClassifier(n_estimators=200, random_state=42)
clf_model.fit(Xc_train, yc_train)

# Save models
joblib.dump(reg_model, "productivity_model.pkl")
joblib.dump(clf_model, "burnout_model.pkl")
print("✅ Models saved successfully!")
