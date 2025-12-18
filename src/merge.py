import pandas as pd

# ----------------------------
# Load datasets
# ----------------------------
vitals = pd.read_csv("data/respiratory/human_vital_signs_dataset_2024.csv/human_vital_signs_dataset_2024.csv")
symptoms = pd.read_csv("data/respiratory/respiratory symptoms and treatment.csv/respiratory symptoms and treatment.csv")

# ----------------------------
# Standardize column names
# ----------------------------
vitals.rename(columns={
    "Gender": "Sex",
    "Oxygen Saturation": "SpO2"
}, inplace=True)

symptoms.rename(columns={
    "Sex": "Sex",
    "Symptoms": "Symptom"
}, inplace=True)

# ----------------------------
# Clean data
# ----------------------------
# Lowercase text for consistency
symptoms["Symptom"] = symptoms["Symptom"].str.lower().str.strip()
symptoms["Sex"] = symptoms["Sex"].str.lower()
vitals["Sex"] = vitals["Sex"].str.lower()

# Drop rows with missing age (cannot merge without it)
symptoms = symptoms.dropna(subset=["Age"])

# ----------------------------
# Encode respiratory symptoms
# ----------------------------
symptoms["Breathlessness"] = symptoms["Symptom"].apply(
    lambda x: 1 if isinstance(x, str) and "shortness of breath" in x else 0
)

symptoms["Chest Tightness"] = symptoms["Symptom"].apply(
    lambda x: 1 if isinstance(x, str) and ("chest" in x or "tight" in x) else 0
)

# Keep only required columns
symptoms_final = symptoms[[
    "Age", "Sex", "Breathlessness", "Chest Tightness", "Disease"
]]

# ----------------------------
# Merge datasets
# ----------------------------
merged = pd.merge(
    vitals,
    symptoms_final,
    how="inner",
    on=["Age", "Sex"]
)

# ----------------------------
# Select final headers
# ----------------------------
final_df = merged[[
    "Patient ID",
    "Age",
    "Sex",
    "Heart Rate",
    "Respiratory Rate",
    "Body Temperature",
    "SpO2",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Breathlessness",
    "Chest Tightness",
    "Disease",
    "Risk Category"
]]

# ----------------------------
# Save output
# ----------------------------
final_df.to_csv("merged_respiratory_dataset.csv", index=False)

print("✅ Respiratory dataset merged successfully!")
