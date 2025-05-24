# -*- coding: utf-8 -*-
"""
Created on Tue May  6 14:17:21 2025

@author: user
"""

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('emotion_model.h5')

# FER-2013 emotion labels (adjust if different in your training script)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

emotion_colors = {
    'Angry':    (0, 0, 255),       # Red
    'Disgust':  (0, 255, 0),       # Green
    'Fear':     (128, 0, 128),     # Purple
    'Happy':    (0, 255, 255),     # Cyan
    'Sad':      (255, 0, 0),       # Blue
    'Surprise': (0, 165, 255),     # Orange
    'Neutral':  (192, 192, 192)    # Gray
}

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam (try 0 if 1 doesn't work)
cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Preprocess the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Predict emotion
        prediction = model.predict(roi, verbose=0)[0]
        label = emotion_labels[np.argmax(prediction)]
        color = emotion_colors.get(label, (255, 255, 255))  # default white if label not found
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw colored text label
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Optional visual expression drawing
        center_x, center_y = x + w // 2, y + h + 10
        if label == "Happy":
            cv2.ellipse(frame, (center_x, center_y), (25, 10), 0, 0, 180, color, 2)  # Smile
        elif label == "Sad":
            cv2.ellipse(frame, (center_x, center_y + 10), (25, 10), 0, 180, 360, color, 2)  # Frown
        elif label == "Angry":
            cv2.line(frame, (x+10, y+10), (x+30, y+5), color, 2)  # Eyebrow left
            cv2.line(frame, (x+w-10, y+10), (x+w-30, y+5), color, 2)  # Eyebrow right

    # Show the frame
    cv2.imshow("Real-time Emotion Detection", frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

 