import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from joblib import dump
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
dataset = pd.read_csv("cyber_attack_dataset.csv")

# Separate the features and labels
X = dataset.drop("Label", axis=1)
y = dataset["Label"]

# Perform the categorical features
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X[categorical_features])

# Create feature names for the encoded features
feature_names = []
for i, feature in enumerate(categorical_features):
    unique_values = X[feature].unique()
    encoded_names = [f"{feature}_{value}" for value in unique_values]
    feature_names.extend(encoded_names)

X_encoded = pd.DataFrame(X_encoded, columns=feature_names)
X.drop(categorical_features, axis=1, inplace=True)
X = pd.concat([X, X_encoded], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# use  Random Forest classifier
classifier = RandomForestClassifier()

# Perform hyperparameter
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model
best_classifier = grid_search.best_estimator_

# Save the best model
dump(best_classifier, "cyber_attack_model.joblib")

# Make predictions
y_pred = best_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
