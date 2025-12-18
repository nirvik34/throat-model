import tensorflow as tf
import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

interpreter = tf.lite.Interpreter(
    model_path="models/throat_disease_cnn.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load image
img_path = "data/throat.v1-fliph.folder/train/THROAT/7ddvfbgnhjhtbnf_JPG.rf.f2ff94d2ebe8f6151ee61ad44cea244b.jpg"
if not os.path.exists(img_path):
    raise FileNotFoundError("❌ test.jpg not found")

img = cv2.imread(img_path)
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0).astype(np.float32)

# Inference
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

prob = interpreter.get_tensor(output_details[0]['index'])[0][0]

# Risk Buckets
if prob < 0.40:
    risk = "🟢 LOW RISK"
    message = "Likely healthy throat"
elif prob < 0.70:
    risk = "🟡 MEDIUM RISK"
    message = "Possible abnormality detected"
else:
    risk = "🔴 HIGH RISK"
    message = "Strong indicators of throat disease"

print("\n=== THROAT HEALTH SCREENING RESULT ===")
print(f"Risk Level   : {risk}")
print(f"Confidence   : {prob:.2f}")
print(f"Assessment  : {message}")
print("\n⚠ This is an AI-assisted screening tool, not a medical diagnosis.")
