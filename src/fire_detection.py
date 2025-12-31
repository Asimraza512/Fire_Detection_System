import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# Try different backend
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

class FireDetector:
    def __init__(self, model_path="../models/keras_Model.h5"):
        self.model = keras.models.load_model(model_path)
        self.img_size = (224, 224)
        self.class_names = ['Non-Fire', 'Fire']
        print("Model loaded successfully!")

    def preprocess_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, self.img_size)
        normalized = (resized / 127.5) - 1.0
        return np.expand_dims(normalized, axis=0)

    def predict(self, frame):
        processed_frame = self.preprocess_frame(frame)
        prediction = self.model.predict(processed_frame, verbose=0)
        confidence = prediction[0][0]
        class_id = 1 if confidence > 0.5 else 0
        actual_confidence = confidence if class_id == 1 else 1 - confidence
        return self.class_names[class_id], actual_confidence


def main():
    detector = FireDetector()

    # Try different camera indexes
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Webcam found at index {i}")
            break
    else:
        print("Error: Could not open any webcam")
        return

    print("Starting real-time fire detection...")
    print("Press 'q' to exit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            prediction, confidence = detector.predict(frame)

            # Display in console
            if prediction == "Fire":
                print(f"🚨 FIRE DETECTED! Confidence: {confidence:.2%} 🚨")
            else:
                print(f"✅ Safe - Confidence: {confidence:.2%}", end='\r')

            # Try to show window
            try:
                text = f"{prediction}: {confidence:.2%}"
                color = (0, 0, 255) if prediction == "Fire" else (0, 255, 0)
                cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow('Fire Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                # If window fails, continue without it
                continue

    except KeyboardInterrupt:
        print("\nStopping...")

    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass
    print("Detection stopped.")


if __name__ == "__main__":
    main()