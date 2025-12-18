import os
import cv2
import numpy as np
import tensorflow as tf
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------
# App Initialization
# ---------------------------
app = FastAPI(
    title="Throat Disease Screening API",
    description="AI-assisted throat disease risk screening using TensorFlow Lite",
    version="1.0.0"
)

# Allow Node.js / frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load TFLite Model ONCE
# ---------------------------
MODEL_PATH = "models/throat_disease_cnn.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 224

# ---------------------------
# Utility Functions
# ---------------------------
def preprocess_image(image_path: str):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def get_risk_bucket(prob: float):
    if prob < 0.40:
        return "LOW", "Likely healthy throat"
    elif prob < 0.70:
        return "MEDIUM", "Possible abnormality detected"
    else:
        return "HIGH", "Strong indicators of throat disease"

# ---------------------------
# Health Check
# ---------------------------
@app.get("/health")
def health_check():
    return {
        "status": "UP",
        "model": "throat-disease-cnn (TFLite)"
    }

# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict/throat")
async def predict_throat(file: UploadFile = File(...)):
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        img = preprocess_image(tmp_path)

        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()

        prob = interpreter.get_tensor(output_details[0]["index"])[0][0]
        risk, message = get_risk_bucket(float(prob))

        return {
            "model": "throat-disease-cnn",
            "probability": round(float(prob), 3),
            "risk_level": risk,
            "assessment": message,
            "disclaimer": "This is an AI-assisted screening tool, not a medical diagnosis."
        }

    finally:
        os.remove(tmp_path)
