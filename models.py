import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os


# 1. SET UP PATHS

train_dir = 'Datasets/chest_xray/chest_xray/train'
test_dir = 'Datasets/chest_xray/chest_xray/test'
val_dir = 'Datasets/chest_xray/chest_xray/val'

# 2. IMAGE PREPROCESSING & AUGMENTATION
# This prepares the images and creates variations to help the AI learn better
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalizes pixels (0-255 to 0-1)
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# For testing, we only rescale (no flipping)
test_val_datagen = ImageDataGenerator(rescale=1./255)

# 3. LOAD THE DATASET
print("--- Loading Images ---")
train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_set = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# 4. BUILD THE NEURAL NETWORK (CNN)
model = models.Sequential([
    # Layer 1: Detecting edges
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),

    # Layer 2: Detecting shapes
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Layer 3: Detecting complex patterns
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Flatten and Decision Making
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Helps prevent overfitting
    layers.Dense(1, activation='sigmoid') # Binary output: 0 or 1
])

# 5. COMPILE
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 6. START TRAINING
print("\n--- Training Started (This may take several minutes) ---")
model.fit(
    train_set,
    epochs=10, 
    validation_data=test_set
)

# 7. SAVE THE FINISHED BRAIN
model.save('pneumonia_model.h5')
print("\nSuccess! Your model is saved as 'pneumonia_model.h5'")
