import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# preprocess the dataset
data = pd.read_csv('Voter_Commands.csv')
data['Voter_Commands'] = data['Voter_Commands'].str.lower()
data['Likelihood_to_Vote'] = data['Voter_Commands'].str.contains('vote|voting').astype(int)

# Split
X = data['Voter_Commands']
y = data['Likelihood_to_Vote']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.85,
    min_df=5
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a logistic regression  with custom options
classifier = LogisticRegression(
    C=1.0,
    penalty='l2',
    max_iter=100,
    random_state=42
)
classifier.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = classifier.predict(X_test_tfidf)
report = classification_report(y_test, y_pred)

# Save the model and vectorizer
joblib.dump(classifier, 'vote_feedback_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("Model Saved")
