# ai_training.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- Configuration ---
DATA_DIR = 'data'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 400

# --- 1. Load Data ---
# Use ImageDataGenerator for loading images and data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Use 20% of data for validation
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary', # For 'correct' and 'bad'
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# --- 2. Build the Model with Transfer Learning ---
# Load the pre-trained MobileNetV2 model without its top classification layer
base_model = MobileNetV2(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')

# Freeze the base model layers so we don't retrain them
base_model.trainable = False

# Create our new model on top
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid') # Binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 3. Train the Model ---
print("\n--- Starting Model Training from Image Data ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# --- 4. Save the Model ---
model.save('image_classifier_model.h5')
print("\n--- Model successfully saved as image_classifier_model.h5 ---")

# Optional: Print out the class indices
print("\nClass Indices:", train_generator.class_indices)