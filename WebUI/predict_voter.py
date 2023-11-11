import pandas as pd
import joblib

# new data from the CSV file
new_data = pd.read_csv('new_voter_data.csv')

# Load  trained model
classifier = joblib.load('vote_feedback_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
X_new_tfidf = tfidf_vectorizer.transform(new_data['Voter_Commands'])

#  predict the likelihood of voting
predictions = classifier.predict(X_new_tfidf)

# Calculate the percentage of people likely to vote
percentage_likely_to_vote = (sum(predictions) / len(predictions)) * 100
print(f"Percentage likely to vote: {percentage_likely_to_vote:.2f}%")
