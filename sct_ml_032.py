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

# Suppress warnings
warnings.filterwarnings("ignore", message=".*Truncated File Read*")

# Define directories and image size
data_dir = "sct_ml_03/PetImages"
img_size = (128, 128)  # Resize for faster processing

# Load VGG16 model for feature extraction
print("Loading VGG16 model for feature extraction...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Image data generator
print("Initializing image data generator...")
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
data_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

# Feature extraction function with batch processing count
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
    print(f"Total batches processed: {batch_count}/{num_batches}")
    return np.array(features), np.array(labels)

# Extract features and labels
print("Extracting features and labels...")
X, y = extract_features(data_generator, model)

# Split data for training and testing
print("Splitting data for training and testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
print("Training SVM classifier...")
svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)

# Predict and evaluate
print("Evaluating model...")
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Save the trained SVM model
model_path = "svm_pet_classifier.joblib"
joblib.dump(svm, model_path)
print(f"Model saved to {model_path}")

# Function to predict using the trained SVM model
def predict_image(image_path, model, svm_classifier):
    print("Processing image for prediction...")
    try:
        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.resize(img_size)  # Resize to match the model's input size
        img_array = np.array(img)
        
        # Check and handle image channels
        print("Checking image channels...")
        if img_array.shape[-1] == 4:  # If the image has an alpha channel (RGBA), remove it
            img_array = img_array[:, :, :3]
        
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess for VGG16

        # Extract features using the VGG16 model
        print("Extracting features for prediction...")
        features = model.predict(img_array, verbose=1)
        features = features.reshape(features.shape[0], -1)  # Flatten features

        # Predict using the trained SVM classifier
        print("Making prediction with SVM classifier...")
        prediction = svm_classifier.predict(features)
        
        # Translate prediction to label
        label = "Cat" if prediction[0] == 0 else "Dog"
        return label
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Prediction Failed"

# User input for testing image
test_image_path = input("Enter the path to the test image (e.g., 'test_image.jpg'): ")

# Check if the image file exists
if not os.path.exists(test_image_path):
    print(f"The file {test_image_path} does not exist. Please provide a valid image path.")
else:
    # Make a prediction
    prediction = predict_image(test_image_path, model, svm)
    print(f"Prediction: {prediction}")  # Output the prediction label (Cat or Dog)
