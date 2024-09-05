import tensorflow
from tensorflow.keras.models import load_model
import numpy as np
import keras
import cv2
import os

model = load_model('./archer/model.h5')

data_path = './archer/dataset'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]

img_size = 256

def inference(img):
    # Store the original image for annotation
    original_image = img.copy()  # Use .copy() to avoid modifying the original image
    
    # Resize the image to the model's input size
    img_single_resized = cv2.resize(img, (img_size, img_size))  # Resize to model input size
    
    # Convert to grayscale if needed and expand dimensions
    if img_single_resized.shape[-1] == 3:  # Check if the image is in RGB format
        img_single_resized = cv2.cvtColor(img_single_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        
    img_single_resized = np.expand_dims(img_single_resized, axis=-1)  # Add channel dimension
    img_single_resized = np.expand_dims(img_single_resized, axis=0)  # Add batch dimension

    # Predict
    predictions_single = model.predict(img_single_resized)
    predicted_label = categories[np.argmax(predictions_single)]
    print('A.I predicts:', predicted_label)

    # Annotate the image
    cv2.putText(original_image, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Annotate with green text

    return original_image, predicted_label
