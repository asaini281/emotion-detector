# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:55:40 2025

@author: user
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam

# Set paths
train_dir = 'fer2013/train'
test_dir = 'fer2013/test'

# Image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(48, 48), color_mode='grayscale',
    batch_size=64, class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(48, 48), color_mode='grayscale',
    batch_size=64, class_mode='categorical')

# Model architecture
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, epochs=25, validation_data=test_generator)

# Save the model
model.save('emotion_model.h5')
print("✅ Model trained and saved as emotion_model.h5")


