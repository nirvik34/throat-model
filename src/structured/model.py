import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset

df = pd.read_csv("data/respiratory/human_vital_signs_dataset_2024.csv/human_vital_signs_dataset_2024.csv")

# Add missing user-input columns if not present
extra_cols = ["breathlessness", "chest_tightness", "cough", "smoker"]
for col in extra_cols:
    if col not in df.columns:
        df[col] = 0

# Feature columns
FEATURES = [
    "Respiratory Rate",
    "Oxygen Saturation",
    "Body Temperature",
    "Heart Rate",
    "Systolic Blood Pressure",
    "Age",
    "Derived_BMI",
    "breathlessness",
    "chest_tightness",
    "cough",
    "smoker"
]

X = df[FEATURES]
y = df["Risk Category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "models/respiratory_model.pkl")
print("✅ Model saved as models/respiratory_model.pkl")
