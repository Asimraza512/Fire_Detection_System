import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import pickle


def load_dataset():
    """
    Load and prepare dataset from folder structure
    """
    fire_path = "../fire_dataset/fire_images"
    non_fire_path = "../fire_dataset/non_fire_images"

    print(f"Loading from: {fire_path} and {non_fire_path}")

    # Check if folders exist
    if not os.path.exists(fire_path):
        print(f"ERROR: Folder '{fire_path}' not found!")
        print("Current directory:", os.getcwd())
        print("Available folders in parent directory:", os.listdir('..'))
        print("Available folders in fire_dataset:", os.listdir('../fire_dataset'))
        return None, None

    if not os.path.exists(non_fire_path):
        print(f"ERROR: Folder '{non_fire_path}' not found!")
        print("Available folders in fire_dataset:", os.listdir('../fire_dataset'))
        return None, None

    images = []
    labels = []

    # Load fire images
    print("Loading fire images...")
    fire_count = 0
    for img_file in os.listdir(fire_path):
        img_path = os.path.join(fire_path, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(1)  # Fire
            fire_count += 1
        else:
            print(f"Warning: Could not load image {img_file}")

    # Load non-fire images
    print("Loading non-fire images...")
    non_fire_count = 0
    for img_file in os.listdir(non_fire_path):
        img_path = os.path.join(non_fire_path, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(0)  # Non-Fire
            non_fire_count += 1
        else:
            print(f"Warning: Could not load image {img_file}")

    print(f"Loaded {fire_count} fire images and {non_fire_count} non-fire images")
    print(f"Total images: {len(images)}")

    return np.array(images), np.array(labels)


def create_model():
    """
    Create and compile the model
    """
    # Load MobileNetV2 as base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze base model layers
    base_model.trainable = False

    # Create custom model on top
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():
    print("Starting Fire Detection Model Training...")

    # Load data
    print("Loading dataset...")
    X, y = load_dataset()

    if X is None or y is None:
        print("Failed to load dataset. Exiting...")
        return

    # Normalize pixel values to [-1, 1]
    X = (X / 127.5) - 1.0

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Create model
    print("Creating model...")
    model = create_model()

    # Display model summary
    print("Model Summary:")
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
    ]

    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=20,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate model
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Create models folder if not exists
    if not os.path.exists('../models'):
        os.makedirs('../models')

    # Save model
    model.save('../models/keras_Model.h5')
    print("Model saved as '../models/keras_Model.h5'")

    # Save training history
    with open('../models/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    print("Training completed successfully! ✅")


if __name__ == "__main__":
    main()