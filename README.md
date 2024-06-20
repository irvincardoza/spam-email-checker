# Email/SMS Spam Classifier

This project implements a machine learning model to classify messages (emails or SMS) as either spam or ham (not spam). It uses natural language processing (NLP) techniques to preprocess the text data and a machine learning model trained on a dataset of labeled messages.

## Project Structure

- **`spamapp/`**: Django application directory.
  - **`models/`**: Contains trained model (`model.pkl`) and vectorizer (`vectorizer.pkl`) files.
  - **`templates/`**: HTML templates for the web application.
    - **`index.html`**: Home page with a form to input messages for classification.
    - **`result.html`**: Result page displaying the classification result.
  - **`static/`**: Static files (CSS, JS) for styling and functionality.

- **`manage.py`**: Django's command-line utility for administrative tasks.

## Important


The `spamproj` folder contains the Django application, which includes the web interface for the Email/SMS Spam Classifier. The actual Python script used to train the model and any related data files are located outside of this folder. 

**Note**: The model included (`model.pkl`) is trained on a dataset that may not be the best for general use cases, which can lead to errors in classification. Further training on a more diverse and representative dataset is recommended for improved performance.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/irvincardoza/spam-email-checker.git
   cd spamproj
   ```
2. Install the requirements needed
   
3. Run the django project
   ```bash
   python manage.py migrate
   python manage.py runserver
   ```
## Contributions are welcome
