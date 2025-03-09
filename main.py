# Importing general-purpose libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Importing machine learning libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Importing image handling libraries
import cv2

# Function to retrieve image file paths from a given directory
def get_image_paths(data):
    paths = []
    valid_extension = ".jpg"

    # Walk through every subfolder in the directory
    for dirpath, dirnames, filenames in os.walk(data):
        for filename in filenames:
            # Convert filename to lower case and check if it ends with ".jpg"
            if filename.lower().endswith(valid_extension):
                # Construct the full path to the file
                full_path = os.path.join(dirpath, filename)
                paths.append(full_path)
    
    return paths

# Function to create training and validation data generators with augmentation
def train_data_gen(train_dir, img_size=(224,224), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.1  # Reserve 10% for validation
    )
    
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    val_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    return train_generator, val_generator

# Function to create test data generator (only rescaling; no augmentation)
def test_data_gen(test_dir, img_size=(224,224), batch_size=32):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    return test_generator

# Function to build a CNN model for image classification
def build_model(input_shape=(224, 224, 3)):
    model = tf.keras.Sequential([
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Third Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Flattening and Fully Connected Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model with Adam optimizer and binary cross-entropy loss
    model.compile(
        optimizer = 'adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Callbacks for efficient training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-7
)

# Main function to execute the training pipeline
def main() :
    # Retrieve training and testing image file paths
    train_paths = get_image_paths("data/train")
    test_paths = get_image_paths("data/test")

    print(f"Found {len(train_paths)} training images.")
    print(f"Found {len(test_paths)} testing images.")

    # Create training, validation, and test data generators
    train_generator, val_generator = train_data_gen("data/train")
    test_generator = test_data_gen("data/test")

    # Build the CNN model
    model = build_model(input_shape=(224, 224, 3))

    # Train the model using the data generators
    model.fit(
        train_generator,
        epochs=25,
        validation_data=val_generator,
        callbacks=[early_stopping, reduce_lr],
        batch_size=32
    )

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(test_generator)
    # Print accuracy summary
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)
    # Save the trained model
    model.save("cat_dog_classifier.h5")  # Saves the model in HDF5 format

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()