import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocessing import load_and_preprocess

DATA_PATH = "data/diabates/Healthcare-Diabetes.csv"
MODEL_PATH = "models/diabetes_model.pkl"

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, MODEL_PATH)
print("✅ Model saved as models/diabetes_model.pkl")
