import cv2
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model('face_detection_model.h5')

# Set the directory for the face dataset
data_dir = 'facedataset'

# Get the user names from the subdirectory names
user_names = sorted(os.listdir(data_dir))

# Create a dictionary mapping the index to the corresponding user name
label_dict = {i: name for i, name in enumerate(user_names)}

# Define the dimensions of the images
img_width, img_height = 150, 150

# Load the face detection algorithm
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the data from the CSV file
csv_file = 'facedataset.csv'
df = pd.read_csv(csv_file)

# Open a video capture device (use 0 for the default camera)
cap = cv2.VideoCapture(0)

# Loop over the frames from the video capture
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop through each face
    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Resize the face region to match the input size of the CNN model
        face = cv2.resize(face, (img_width, img_height))

        # Normalize the face region
        face = face / 255.0

        # Reshape the face region to match the input shape of the CNN model
        face = face.reshape((1, img_width, img_height, 3))

        # Classify the face using the trained CNN model
        prediction = model.predict(face)

        # Get the predicted label
        label_index = np.argmax(prediction)
        label = label_dict[label_index]

        # Get the mobile and NIC details from the CSV file
        row = df.loc[df['Name'] == label].iloc[0]
        mobile = row['Mobile']
        NIC = row['NIC']

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Put the predicted label and other details on the rectangle
        text = f'{label}, {mobile}, {NIC}'
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(label)
        print(mobile)
        print(NIC)

    # Display the frame
    cv2.imshow('Face detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
