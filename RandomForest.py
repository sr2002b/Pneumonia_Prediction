# created using chatgpt-4 per assignment instructions
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# Load the dataset
dataset_path = 'chest_xray'
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

# Preprocess the data
def preprocess_data(data_path, img_size=(150, 150)):
    images = []
    labels = []
    for label in ['NORMAL', 'PNEUMONIA']:
        label_path = os.path.join(data_path, label)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize(img_size)
            img = np.array(img) / 255.0
            images.append(img)
            labels.append(0 if label == 'NORMAL' else 1)
    return np.array(images), np.array(labels)

train_images, train_labels = preprocess_data(train_path)
val_images, val_labels = preprocess_data(val_path)
test_images, test_labels = preprocess_data(test_path)

train_images = train_images.reshape(-1, 150 * 150)
val_images = val_images.reshape(-1, 150 * 150)
test_images = test_images.reshape(-1, 150 * 150)

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(train_images, train_labels)

# Evaluate the model
train_preds = rf_classifier.predict(train_images)
print("Training Set Evaluation:")
print(classification_report(train_labels, train_preds, zero_division=1))
print(confusion_matrix(train_labels, train_preds))
print("Training Accuracy:", accuracy_score(train_labels, train_preds))

val_preds = rf_classifier.predict(val_images)
print("Validation Set Evaluation:")
print(classification_report(val_labels, val_preds, zero_division=1))
print(confusion_matrix(val_labels, val_preds))
print("Validation Accuracy:", accuracy_score(val_labels, val_preds))

test_preds = rf_classifier.predict(test_images)
print("Test Set Evaluation:")
print(classification_report(test_labels, test_preds, zero_division=1))
print(confusion_matrix(test_labels, test_preds))
print("Test Accuracy:", accuracy_score(test_labels, test_preds))

# Create a DataFrame to store the results
results = pd.DataFrame(columns=['Set', 'Accuracy'])
results.loc[0] = ['Training', accuracy_score(train_labels, train_preds)]
results.loc[1] = ['Validation', accuracy_score(val_labels, val_preds)]
results.loc[2] = ['Test', accuracy_score(test_labels, test_preds)]
print("\nAccuracy Results:")
print(results)
