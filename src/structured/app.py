import joblib
import pandas as pd

MODEL_PATH = "models/respiratory_model.pkl"

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

# Load trained model
model = joblib.load(MODEL_PATH)

def predict_risk(user_input: dict):
    """
    user_input example:
    {
        "Respiratory Rate": 22,
        "Oxygen Saturation": 92,
        "Body Temperature": 38.1,
        "Heart Rate": 105,
        "Systolic Blood Pressure": 130,
        "Age": 45,
        "Derived_BMI": 27.3,
        "breathlessness": 1,
        "chest_tightness": 0,
        "cough": 1,
        "smoker": 0
    }
    """

    df = pd.DataFrame([user_input], columns=FEATURES)
    prediction = model.predict(df)[0]
    return prediction


# 🔹 Test run (can remove later)
if __name__ == "__main__":
    sample_input = {
        "Respiratory Rate": 24,
        "Oxygen Saturation": 90,
        "Body Temperature": 38.5,
        "Heart Rate": 110,
        "Systolic Blood Pressure": 135,
        "Age": 50,
        "Derived_BMI": 29.1,
        "breathlessness": 1,
        "chest_tightness": 1,
        "cough": 1,
        "smoker": 0
    }

    result = predict_risk(sample_input)
    print("🩺 Predicted Risk Category:", result)
