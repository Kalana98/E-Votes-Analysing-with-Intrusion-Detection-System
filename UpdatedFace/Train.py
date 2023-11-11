import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set the directory for the face dataset
data_dir = 'facedataset'

# Define the dimensions of the images
img_width, img_height = 150, 150

# Get the number of classes (i.e., the number of users)
num_classes = len(os.listdir(data_dir))

# Initialize the training data and labels
training_data = []
training_labels = []

# Loop through each subdirectory (i.e., each user)
for i, sub_dir in enumerate(os.listdir(data_dir)):
    # Loop through each image in the subdirectory
    for file_name in os.listdir(os.path.join(data_dir, sub_dir)):
        # Read the image and resize it
        img = cv2.imread(os.path.join(data_dir, sub_dir, file_name))
        img = cv2.resize(img, (img_width, img_height))

        # Add the image to the training data and the corresponding label to the training labels
        training_data.append(img)
        training_labels.append(i)

# Convert the training data and labels to numpy arrays
training_data = np.array(training_data)
training_labels = np.array(training_labels)

# Normalize the training data
training_data = training_data / 255.0

# Convert the training labels to one-hot encoding
training_labels = to_categorical(training_labels, num_classes)

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(training_data, training_labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('face_detection_model.h5')
