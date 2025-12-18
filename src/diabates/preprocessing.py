import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

TARGET = "Outcome"

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Drop ID (not useful for learning)
    if "Id" in df.columns:
        df.drop(columns=["Id"], inplace=True)

    # Columns where 0 is medically invalid
    zero_invalid_cols = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    # Replace 0 with median of non-zero values
    for col in zero_invalid_cols:
        non_zero_values = df[df[col] != 0][col]
        if not non_zero_values.empty:
            median_val = non_zero_values.median()
            df[col] = df[col].replace(0, median_val)
        # If all are 0, leave as 0 (or handle differently if needed)

    X = df[FEATURES]
    y = df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
