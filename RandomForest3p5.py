# chatGPT version 3.5 (public) of the Random Forest
import time
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image

# Step 2: Load the dataset and extract the features

# Define the paths for the dataset
train_dir = 'chest_xray/train'
test_dir = 'chest_xray/test'

# Function to load and preprocess the images
def load_images_from_directory(directory):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        for filename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, filename)
            image = Image.open(image_path).convert('RGB')
            image = image.resize((256, 256))
            image_array = np.array(image) / 255.0  # Normalize pixel values
            images.append(image_array)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load the training set
X_train, y_train = load_images_from_directory(train_dir)

# Load the testing set
X_test, y_test = load_images_from_directory(test_dir)

# Flatten the image data
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Step 6: Create a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Start the timer
start_time = time.time()

# Step 7: Train the random forest classifier on the training data
rf.fit(X_train, y_train)

# Step 8: Make predictions on the test data
y_pred = rf.predict(X_test)

# Stop the timer
end_time = time.time()

# Step 9: Calculate and print the accuracy using pandas
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate the execution time in seconds and round to two decimal places
execution_time = round(end_time - start_time, 2)
print("Execution time:", execution_time, "seconds")
