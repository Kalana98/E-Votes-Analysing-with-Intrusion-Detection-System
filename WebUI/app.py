from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

#-------------------1
# Load the trained model and vectorizer

import os
app.config['UPLOAD_FOLDER'] = 'uploads'

classifier = joblib.load('vote_feedback_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/Voter_personage_analyzer', methods=['GET', 'POST'])
def index():
    percentage_likely_to_vote = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'No file part'
        else:
            file = request.files['file']
            if file.filename == '':
                error = 'No selected file'
            else:
                new_data = pd.read_csv(file)
                X_new_tfidf = tfidf_vectorizer.transform(new_data['Voter_Commands'])
                predictions = classifier.predict(X_new_tfidf)
                percentage_likely_to_vote = (sum(predictions) / len(predictions)) * 100

    return render_template('Voter_personage_analyzer.html', percentage_likely_to_vote=percentage_likely_to_vote)


#----------------------2
from flask import Flask, render_template, request
import pandas as pd
from joblib import load
# Load the model and encoder
loaded_classifier = load("cyber_attack_model.joblib")
encoder = load("encoder.joblib")

@app.route("/Predict_cyber_Attack", methods=["GET", "POST"])
def Predict_cyber_Attack():
    if request.method == "POST":
        import pandas as pd
        from sklearn.preprocessing import OneHotEncoder
        from joblib import dump, load

        # Load the trained model
        loaded_classifier = load("cyber_attack_model.joblib")

        # Load the dataset
        dataset = pd.read_csv("cyber_attack_dataset.csv")

        # Separate the features and labels
        X = dataset.drop("Label", axis=1)
        y = dataset["Label"]

        # List of categorical feature names
        categorical_features = ["IP_address", "Time_of_Day", "Protocol"]  # Replace with your categorical feature names

        # Create an encoder
        encoder = OneHotEncoder(sparse=False)
        X_encoded = encoder.fit_transform(X[categorical_features])

        feature_names = []
        for i, feature in enumerate(categorical_features):
            unique_values = X[feature].unique()
            encoded_names = [f"{feature}_{value}" for value in unique_values]
            feature_names.extend(encoded_names)

        X_encoded = pd.DataFrame(X_encoded, columns=feature_names)
        X.drop(categorical_features, axis=1, inplace=True)
        X = pd.concat([X, X_encoded], axis=1)

        # Save the encoder for future use
        dump(encoder, "encoder.joblib")

        # Get user input
        user_input = {
            "IP_address": request.form["IP_address"],
            "Time_of_Day": request.form["Time_of_Day"],
            "Number_of_Packets": int(request.form["Number_of_Packets"]),
            "Protocol":  request.form["Protocol"]
        }

        # Create a DataFrame from user input
        user_input_df = pd.DataFrame([user_input])

        # Apply the same preprocessing steps on user input
        user_input_encoded = encoder.transform(user_input_df[categorical_features])
        user_input_encoded_df = pd.DataFrame(user_input_encoded, columns=feature_names)
        user_input_df.drop(categorical_features, axis=1, inplace=True)
        user_input_df = pd.concat([user_input_df, user_input_encoded_df], axis=1)

        # Make predictions on user input
        user_predictions = loaded_classifier.predict(user_input_df)
        if (user_predictions[0] == 1):
            user_predictions = "Cyber Attack Prediction result: Possible"
        else:
            user_predictions = "Cyber Attack Prediction result: Not Possible"
        return render_template("Predict_cyber_Attack.html", prediction=user_predictions)

    # packets , protocol , IP
    import psutil
    import socket

    # Get network statistics
    net_stats = psutil.net_io_counters()

    # Get the number of packets sent and received
    packets_sent = net_stats.packets_sent
    packets_received = net_stats.packets_recv

    # Determine the protocol based on the statistics (assumes TCP for demonstration)
    protocol = "TCP" if packets_sent > packets_received else "UDP"

    print(f'Packets Received: {packets_received}')
    print(f'Protocol: {protocol}')

    # Get the local IP address
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f'Local IP Address: {local_ip}')

    from datetime import datetime

    # Get the current time
    current_time = datetime.now().time()

    # Define time ranges
    morning_start = datetime.strptime('06:00:00', '%H:%M:%S').time()
    afternoon_start = datetime.strptime('12:00:00', '%H:%M:%S').time()
    evening_start = datetime.strptime('18:00:00', '%H:%M:%S').time()

    # Compare the current time with the defined ranges
    if morning_start <= current_time < afternoon_start:
        print("Morning")
        timeing="Morning"
    elif afternoon_start <= current_time < evening_start:
        print("Afternoon")
        timeing="Afternoon"
    else:
        print("Evening")
        timeing="Evening"

    return render_template("Predict_cyber_Attack.html", prediction=None, local_ip=local_ip ,timeing=timeing,packets_received=packets_received,protocol=protocol)


# face
# Load the trained CNN model
model = load_model('face_detection_model.h5')

# Set the directory for the face dataset
data_dir = 'static/facedataset'

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

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (img_width, img_height))
            face = face / 255.0
            face = face.reshape((1, img_width, img_height, 3))

            prediction = model.predict(face)
            label_index = np.argmax(prediction)
            label = label_dict[label_index]

            row = df.loc[df['Name'] == label].iloc[0]
            mobile = row['Mobile']
            NIC = row['NIC']

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f'{label}, {mobile}, {NIC}'
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/Face')
def Face():
    return render_template('Face.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# face Register...................................
import cv2
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a directory to save the images
if not os.path.exists('static/facedataset'):
    os.makedirs('static/facedataset')

# Create an empty DataFrame to store the details
df = pd.DataFrame(columns=['Name', 'Mobile', 'NIC'])

# Initialize the video capture object as a global variable
cap = None



@app.route('/Voter_Register')
def Voter_Register():
    return render_template('Voter_Register.html')

@app.route('/capture', methods=['POST'])
def capture():
    global df, cap  # Declare df and cap as global to access the outer scope variables

    name = request.form['name']
    mobile = request.form['mobile']
    NIC = request.form['nic']

    count = 0
    if cap is None:
        cap = cv2.VideoCapture(0)  # Initialize the camera if it's not already

    while count < 200:
        ret, frame = cap.read()

        if not ret:
            # Handle the case when frame capture fails, e.g., the camera is not available
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(frame, f"Name: {name}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Mobile: {mobile}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"NIC: {NIC}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Count: {count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('c'):
            if name not in df['Name'].values:
                new_df = pd.DataFrame({'Name': [name], 'Mobile': [mobile], 'NIC': [NIC]})
                df = pd.concat([df, new_df], ignore_index=True)
                if not os.path.exists('facedataset.csv'):
                    df.to_csv('facedataset.csv', index=False)
                else:
                    csv_df = pd.read_csv('facedataset.csv')
                    if name not in csv_df['Name'].values:
                        df.to_csv('facedataset.csv', mode='a', header=False, index=False)

            # Ensure the directory for saving images exists
            img_dir = os.path.join('static/facedataset', name)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            cv2.imwrite(os.path.join(img_dir, f'{name}_{count}.jpg'), gray[y:y + h, x:x + w])
            count += 1

    # Release the camera and close the OpenCV window at the end of the capture
    cap.release()
    cv2.destroyAllWindows()

    # Set the `cap` variable to `None` after the user has finished capturing images
    cap = None

    return redirect(url_for('Voter_Register'))



# load voter data

df = pd.read_csv('facedataset.csv')

@app.route('/RegisteredData')
def RegisteredData():
    return render_template('RegisteredData.html', data=df)


if __name__ == "__main__":
    app.run(debug=True)
