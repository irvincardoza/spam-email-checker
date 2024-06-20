import pickle

# Load the pickled model and vectorizer
vectorizer = pickle.load(open('spamapp/models/vectorizer.pkl', 'rb'))
model = pickle.load(open('spamapp/models/model.pkl', 'rb'))

# Test data
test_message = "Congratulations! You've won a free ticket to the Bahamas!"
data = [test_message]
vect = vectorizer.transform(data).toarray()
prediction = model.predict(vect)
prediction_text = 'Spam' if prediction[0] == 1 else 'Ham'

print(f"Message: {test_message}")
print(f"Vectorized: {vect}")
print(f"Prediction: {prediction}")
print(f"Prediction Text: {prediction_text}")
