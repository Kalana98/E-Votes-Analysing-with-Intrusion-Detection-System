import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load the trained model
loaded_classifier = joblib.load("voting_classifier_model.pkl")

# Load the dataset from the CSV file
dataset = pd.read_csv("vote_codes_dataset.csv")

# Separate the features (vote codes, candidate, voted time) and labels (candidate)
X = dataset.drop("Candidate", axis=1)
y = dataset["Candidate"]

# Define the column transformer for one-hot encoding
categorical_features = ["VoteCode", "VotedTime"]
preprocessor = ColumnTransformer(
    transformers=[
        ("encoder", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder="passthrough"
)

# Apply the column transformer to encode the categorical features
X_encoded = preprocessor.fit_transform(X)

# User input
user_input = {
    "VoteCode": "A123",
    "VotedTime": "2023-08-30"
}

# Create a DataFrame from user input
user_input_df = pd.DataFrame([user_input])

# Apply the same column transformer to user input data
user_input_encoded = preprocessor.transform(user_input_df)

# Make predictions using the loaded model
user_prediction = loaded_classifier.predict(user_input_encoded)

print("Predicted Candidate:", user_prediction[0])
