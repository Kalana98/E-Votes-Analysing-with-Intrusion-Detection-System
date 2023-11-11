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
    "IP_address": "10.0.0.1",         # Replace with your IP address
    "Time_of_Day": "Afternoon",      # Replace with your time of day
    "Number_of_Packets": 5,          # Replace with your numerical value
    "Protocol": "ICMP"               # Replace with your protocol
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
print("Predictions for user input:", user_predictions)
