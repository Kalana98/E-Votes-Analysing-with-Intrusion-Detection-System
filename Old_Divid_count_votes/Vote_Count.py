import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load the dataset from the CSV file
dataset = pd.read_csv("vote_codes_dataset.csv")

# Separate the features (vote codes, candidate, voted time) and labels (candidate)
X = dataset.drop("Candidate", axis=1)
y = dataset["Candidate"]

# Define the column transformer for one-hot encoding
categorical_features = ["VoteCode", "VotedTime"]
preprocessor = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), categorical_features)],
    remainder="passthrough"
)

# Apply the column transformer to encode the categorical features
X_encoded = preprocessor.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
classifier = RandomForestClassifier()

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test)


# Save the trained model to a file
joblib.dump(classifier, "voting_classifier_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
