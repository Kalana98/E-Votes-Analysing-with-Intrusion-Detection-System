import cv2
import os
import pandas as pd

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get the name, mobile number, and NIC details from the user
name = input("Enter the name: ")
mobile = input("Enter the mobile number: ")
NIC = input("Enter the NIC details: ")

# Create an empty DataFrame to store the details
df = pd.DataFrame(columns=['Name', 'Mobile', 'NIC'])

# Create a directory to save the images
if not os.path.exists('facedataset'):
    os.makedirs('facedataset')

# Create a subdirectory with the given name to save the images
if not os.path.exists(f'facedataset/{name}'):
    os.makedirs(f'facedataset/{name}')

# Check if the CSV file already exists
if os.path.exists('facedataset.csv'):
    # Read the existing data from the CSV file
    df = pd.read_csv('facedataset.csv')

# Check if the name already exists in the DataFrame
if name not in df['Name'].values:
    # Create a new DataFrame with the details
    new_df = pd.DataFrame({'Name': [name], 'Mobile': [mobile], 'NIC': [NIC]})

    # Concatenate the new DataFrame with the existing one
    df = pd.concat([df, new_df], ignore_index=True)

    # Save the DataFrame to CSV after adding a new user
    df.to_csv('facedataset.csv', index=False)

count = 0
while count < 200:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(frame, f"Name: {name}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Mobile: {mobile}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"NIC: {NIC}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Count: {count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Press 'c' to capture the current frame
    # Press 'c' to capture the current frame
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        # Check if the name already exists in the DataFrame
        if name not in df['Name'].values:
            # Create a new DataFrame with the details
            new_df = pd.DataFrame({'Name': [name], 'Mobile': [mobile], 'NIC': [NIC]})

            # Concatenate the new DataFrame with the existing one
            df = pd.concat([df, new_df], ignore_index=True)

            # Save the DataFrame to CSV after adding a new user
            if not os.path.exists('facedataset.csv'):
                df.to_csv('facedataset.csv', index=False)
            else:
                # Check if the name already exists in the CSV file
                csv_df = pd.read_csv('facedataset.csv')
                if name not in csv_df['Name'].values:
                    # Append the new DataFrame to the CSV file
                    df.to_csv('facedataset.csv', mode='a', header=False, index=False)

        # Save the current frame as an image
        cv2.imwrite(f'facedataset/{name}/{name}_{count}.jpg', gray[y:y + h, x:x + w])
        count += 1

