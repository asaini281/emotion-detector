# -*- coding: utf-8 -*-
"""
Created on Wed May  7 11:21:08 2025

@author: user
"""

import cv2
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load model and define labels/colors
model = load_model('emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = {
    'Angry': (0, 0, 255), 'Disgust': (0, 255, 0), 'Fear': (128, 0, 128),
    'Happy': (0, 255, 255), 'Sad': (255, 0, 0),
    'Surprise': (0, 165, 255), 'Neutral': (192, 192, 192)
}

# List of test images
test_images = ['photo1.png','photo2.png','photo3.png','photo4.png',]  # Add yours here
# Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Loop through images
for image_path in test_images:
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read {image_path}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi, verbose=0)[0]
        label = emotion_labels[np.argmax(prediction)]
        color = emotion_colors.get(label, (255, 255, 255))

        # Draw face box and label
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Optional expression overlays
        center_x, center_y = x + w // 2, y + h + 10
        if label == "Happy":
            cv2.ellipse(image, (center_x, center_y), (25, 10), 0, 0, 180, color, 2)  # Smile
        elif label == "Sad":
            cv2.ellipse(image, (center_x, center_y + 10), (25, 10), 0, 180, 360, color, 2)  # Frown
        elif label == "Angry":
            # Angry eyebrows
            cv2.line(image, (x+10, y+10), (x+30, y+5), color, 2)  # Left
            cv2.line(image, (x+w-10, y+10), (x+w-30, y+5), color, 2)  # Right

    # Display image with resizing
    resized = cv2.resize(image, (425, 600))
    cv2.imshow("Emotion Detection (Press 'N' for next or 'Q' to quit)", resized)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()



