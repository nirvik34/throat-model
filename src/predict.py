import tensorflow as tf
import cv2
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model = tf.keras.models.load_model("models/throat_disease_cnn.h5")

img_path = "data/throat.v1-fliph.folder/train/THROAT/7ddvfbgnhjhtbnf_JPG.rf.f2ff94d2ebe8f6151ee61ad44cea244b.jpg"

if not os.path.exists(img_path):
    raise FileNotFoundError("❌ test.jpg not found")

img = cv2.imread(img_path)
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)[0][0]

if prediction > 0.5:
    print(f"⚠ Disease Detected (confidence: {prediction:.2f})")
else:
    print(f"✅ Healthy Throat (confidence: {1 - prediction:.2f})")
