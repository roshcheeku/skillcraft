import os
import numpy as np
import warnings
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image

warnings.filterwarnings("ignore", message=".*Truncated File Read*")

data_dir = "sct_ml_03/PetImages"
img_size = (128, 128)  # Resizing for faster processing

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
data_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

def extract_features(generator, model, batch_size=32):
    features = []
    labels = []
    num_batches = generator.samples // batch_size  # Total number of batches
    batch_count = 0  # Initialize batch counter

    for _ in range(num_batches):
        try:
            imgs, lbls = next(generator)
            feats = model.predict(imgs, verbose=0)
            features.extend(feats.reshape(feats.shape[0], -1))
            labels.extend(lbls)
            batch_count += 1  # Increment batch counter

            # Print progress for each batch processed
            print(f"Batch {batch_count}/{num_batches} processed.")
        except Exception as e:
            print(f"Error processing batch {batch_count + 1}: {e}")
            continue  # Skip batches with corrupted images
    return np.array(features), np.array(labels)

# Extract features and labels
X, y = extract_features(data_generator, model)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Save the trained SVM model
model_path = "svm_pet_classifier.joblib"
joblib.dump(svm, model_path)
print(f"Model saved to {model_path}")
