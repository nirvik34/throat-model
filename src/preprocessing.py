import cv2
import numpy as np

IMG_SIZE = 224

def preprocess_image(img_path: str):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("❌ Could not read image")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img
