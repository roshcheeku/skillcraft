import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mediapipe as mp

# Step 1: Define dataset paths and parameters
train_dir = r'C:\Users\dhanyashree\Downloads\archive (11)\HandGesture\images\train'
test_dir = r'C:\Users\dhanyashree\Downloads\archive (11)\HandGesture\images\testing'
image_size = (128, 128)
batch_size = 32
epochs = 15

# Step 2: Data loading and preprocessing
train_data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)

test_data_gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_data_gen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
)

test_data = test_data_gen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
)

class_labels = list(train_data.class_indices.keys())  


base_model = MobileNetV2(input_shape=(image_size[0], image_size[1], 3), include_top=False, weights='imagenet')
base_model.trainable = False 

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


print("Training the model...")
model.fit(train_data, validation_data=test_data, epochs=epochs)
print("Training complete!")

# Save the model
model.save("gesture_recognition_model.h5")
print("Model saved successfully!")

# Step 5: Real-time gesture recognition
# Load the saved model
model = tf.keras.models.load_model("gesture_recognition_model.h5")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box for the hand
            h, w, c = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w) - 20
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h) - 20
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w) + 20
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h) + 20

            # Ensure bounding box coordinates are within the frame
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            # Extract and preprocess ROI
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                img = cv2.resize(roi, image_size)
                img = np.expand_dims(img, axis=0) / 255.0

                # Predict the gesture
                pred = model.predict(img)
                confidence = np.max(pred)
                if confidence > 0.7:  # Set a confidence threshold
                    gesture = class_labels[np.argmax(pred)]
                else:
                    gesture = "Unknown"

                # Display the prediction on the frame
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()