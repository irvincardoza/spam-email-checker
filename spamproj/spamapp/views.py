import pickle
from django.shortcuts import render
from django.http import JsonResponse

# Load the pickled model and vectorizer
vectorizer = pickle.load(open('spamapp/models/vectorizer.pkl', 'rb'))
model = pickle.load(open('spamapp/models/model.pkl', 'rb'))

def index(request):
    return render(request, 'spamapp/index.html')

def predict(request):
    if request.method == 'POST':
        message = request.POST['message']
        data = [message]
        vect = vectorizer.transform(data).toarray()
        prediction = model.predict(vect)
        prediction_text = 'Spam' if prediction[0] == 1 else 'Not Spam'
        return render(request, 'spamapp/result.html', {'prediction': prediction_text})
    else:
        return render(request, 'spamapp/index.html')
