# -*- coding: utf-8 -*-
"""AI-Powered Face Recognition System for Look-Alike Celebrities.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Lqa4eO8EX6Lz3JKlm39jlInarCZQ_62y

### **This project implements an AI-based face recognition system that identifies celebrities who resemble the person in a given image. It utilizes a Convolutional Neural Network (CNN) trained on a dataset of celebrity images to compute similarity scores between the input image and known celebrities.**

### **First of All**, The following packages are necessary to run the generated project on Google Colab:

### **opencv-python:** For image processing tasks like resizing, background removal, and visualization.
### **keras**: As a high-level API for building and training neural networks.
### **tensorflow**: The underlying framework for Keras, providing essential building blocks for deep learning models.
### **deepface**: A library specifically designed for deep learning-based face analysis tasks, including face recognition.
### **imutils**: A set of convenience functions for image processing tasks.
### **tqdm**: A progress bar library to visualize the progress of long-running operations.
### **scikit-learn**: A machine learning library for tasks like preprocessing, model evaluation, and data analysis.
"""

!pip install opencv-python keras tensorflow deepface imutils tqdm scikit-learn

"""# **1) Prepare the Data and Images**"""

import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import max_norm
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from google.colab import drive
import os

# Data Path Configuration
data_dir = '/content/drive/MyDrive/celebrities_photos_jpeg_only'
labels_dir = "/content/drive/MyDrive/My_look_alike_celebrities_model_completed/labels"

# Function to remove background (optional, explore alternative methods)
def remove_background(image):
    """
    This function attempts to remove the background from an image using basic thresholding.
    Consider exploring more advanced background removal techniques (e.g., segmentation models)
    if needed for your specific dataset.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        thresh_inv = cv2.bitwise_not(thresh)
        contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [max_contour], -1, (255), thickness=cv2.FILLED)
        person_only = cv2.bitwise_and(image, image, mask=mask)
        return person_only
    except Exception as e:
        print(f"Background removal exception: {e}")
        return image

# Function to create labels database
def create_labels_db(labels, file_name):
    data = {'Index': [i for i in range(len(labels))], 'Label': labels}
    df = pd.DataFrame(data)
    df.to_csv(labels_dir + "/" + file_name, index=False)

# Function to generate image paths
def get_image_paths(root_dir):
    celebrity_paths = [os.path.join(root_dir, subfolder) for subfolder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subfolder))]
    image_paths = []
    for celeb_path in celebrity_paths:
        for image in os.listdir(celeb_path):
            if image.endswith('.jpg'):
                image_path = os.path.join(celeb_path, image)
                image_name, _ = os.path.splitext(os.path.basename(image_path))
                image_name = ''.join(c for c in image_name if c.isascii())  # Remove non-ASCII characters
                image_dir = os.path.dirname(image_path)
                new_image_path = os.path.join(image_dir, image_name + ".jpg")
                os.rename(image_path, new_image_path)  # Rename for consistency
                image_paths.append(new_image_path)
    return image_paths

# Function to preprocess image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    # Consider using background removal if needed
    # img = remove_background(img)  # Uncomment if using background removal
    img = cv2.resize(img, (128, 128))  # Resize to desired input shape
    return img

# Load images and labels
images = []
labels = []
for image_path in get_image_paths(data_dir):
    try:
        img = preprocess_image(image_path)
        images.append(img)
        label = os.path.basename(os.path.dirname(image_path))[:-1]  # Extract label from directory name
        labels.append(label)

"""# **`2) Data preparation, model training, and evaluation`**"""

# Create labels database
create_labels_db(labels, "main_labels")

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create labels databases for training and testing sets
create_labels_db(y_train, "trained_labels")
create_labels_db(y_test, "test_labels")

# Convert data to NumPy arrays
X_train_resized = np.array(X_train)
X_test_resized = np.array(X_test)

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)

# Define the CNN model (adjust architecture as needed)
num_classes = len(np.unique(y_train))
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_resized, y_train_encoded, epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_resized, y_test_encoded)

# Save the model
model_path = '/content/drive/MyDrive/My_look_alike_celebrities_model_completed/models/celebrities22.h5'
model.save(model_path)

"""**Key improvements of the second Section:**

Data Splitting: Clearly separated the dataset into training and testing sets for better evaluation.
Label Encoding: Used LabelEncoder to convert categorical labels into numerical values for model training.
Model Architecture: Provided a basic CNN architecture as a starting point. You can customize it based on your dataset's complexity and requirements.
Training and Evaluation: Included training and evaluation steps to assess the model's performance.

# **3. Prediction and Visualization**

**1. Hyperparameter Tuning**: Experiment with different hyperparameters (e.g., learning rate, batch size, number of epochs) to optimize model performance.
**2. Data Augmentation**: Consider techniques like image rotation, flipping, and cropping to increase the dataset's diversity and improve generalization.
**3. Model Evaluation Metrics**: Explore additional metrics beyond accuracy (e.g., precision, recall, F1-score) to evaluate the model's performance in different aspects.
**4. Deployment**: If satisfied with the results, deploy the model for real-world use.
Remember to adjust the code based on your specific dataset and requirements. Experimentation and fine-tuning are essential for building a robust and effective face recognition model.
"""

# Load the saved model
model = load_model(model_path)

# Function to preprocess a single image
def preprocess_single_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Adjust size as needed
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

# Function to predict similar celebrities
def predict_similar_celebrities(image_path, top_n=5):
    input_image = preprocess_single_image(image_path)
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    predictions = model.predict(input_image)
    top_indices = np.argsort(predictions[0])[::-1][:top_n]

    # Assuming you have a mapping from index to label
    label_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}

    for i, idx in enumerate(top_indices):
        label = label_mapping[idx]
        # Get image of celebrity (replace with your image data)
        celebrity_image = load_celebrity_image(label)  # Assuming you have a function to load celebrity images
        celebrity_image = np.asarray(celebrity_image)
        celebrity_image = cv2.cvtColor(celebrity_image, cv2.COLOR_BGR2RGB)
        plt.subplot(1, top_n, i+1)
        plt.imshow(celebrity_image)
        plt.title(label)
        plt.axis('off')

# Example usage
image_path = '/content/drive/MyDrive/photos/mohamed_said.jpg'  # Path to input image
predict_similar_celebrities(image_path)
plt.show()

"""## **Key improvements of the Third Section**:
**Function for Preprocessing Single Image**: Added a function to preprocess a single image for prediction.
**Label Mapping**: Used label_encoder.classes_ to get the mapping between index and label.
**Visualization**: Improved the visualization by displaying the predicted celebrity images along with their labels.

# **Additional Considerations of the Third Section:**

**Similarity Metric**: Consider using different similarity metrics (e.g., Euclidean distance) based on your specific requirements.
**Model Refinement**: If the results are not satisfactory, experiment with different model architectures, hyperparameters, or data augmentation techniques.
**Real-Time Applications**: For real-time applications, optimize the code for speed and consider using a GPU.

# **Additional Features and Enhancements**

# **4. Predict similar celebrities with similarity scores**
"""

# Function to load celebrity images
def load_celebrity_image(label):
    for img_path in image_paths:
        if label in img_path:
            return cv2.imread(img_path)  # Load image directly
    return None

# Function to calculate similarity score
def calculate_similarity(input_embedding, celebrity_embedding):
    return cosine_similarity(input_embedding.reshape(1, -1), celebrity_embedding.reshape(1, -1))[0][0]

# Function to predict similar celebrities with similarity scores
def predict_similar_celebrities_with_scores(image_path, top_n=5):
    input_image = preprocess_single_image(image_path)
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    input_embedding = model.predict(input_image)[0]  # Get embedding for input image

    celebrity_embeddings = model.predict(np.array(images))  # Get embeddings for all celebrities

    similarities = [calculate_similarity(input_embedding, celebrity_embedding) for celebrity_embedding in celebrity_embeddings]
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    top_labels = [labels[idx] for idx in top_indices]
    top_scores = [similarities[idx] for idx in top_indices]

    return top_labels, top_scores

# Example usage
image_path = '/content/drive/MyDrive/photos/mohamed_said.jpg'  # Path to input image
top_labels, top_scores = predict_similar_celebrities_with_scores(image_path)

print("Top 3 similar celebrities:")
for label, score in zip(top_labels, top_scores):
    print("Celebrity:", label, "Similarity Score:", score)

"""### **Key improvements of the Fourth Section:**
**Celebrity Image Loading**: Directly loads celebrity images using cv2.imread.
**Similarity Calculation**: Added a function to calculate the cosine similarity between embeddings.
**Prediction with Scores**: Modified the prediction function to return top similar celebrities along with their similarity scores.

# **Additional Considerations of the Fourth Section:**

**Feature Extraction**: Experiment with different feature extraction techniques (e.g., pre-trained models like VGGFace) to improve accuracy.
**Ensemble Methods**: Consider combining multiple models (e.g., using ensemble techniques like bagging or boosting) to improve performance.
**Real-Time Deployment**: For real-time applications, optimize the code for speed and explore GPU acceleration.

### **General Customization**
**Dataset**: Replace the dataset path with your own.
**Model Architecture**: Experiment with different CNN architectures or pre-trained models.
**Hyperparameters**: Tune hyperparameters like learning rate, batch size, and number of epochs.
**Evaluation Metrics**: Use additional metrics like precision, recall, and F1-score to evaluate performance.
"""