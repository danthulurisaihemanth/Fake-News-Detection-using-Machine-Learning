from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import re
import nltk

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and vectorizer
model = LogisticRegression()
vectorizer = TfidfVectorizer()

# Sample model training (you can load your pre-trained model and vectorizer)
def train_model():
    # Preprocessing function as provided earlier
    def preprocess_text(text_data):
        preprocessed_text = []
        for sentence in text_data:
            sentence = re.sub(r'[^\w\s]', '', sentence)
            preprocessed_text.append(' '.join(token.lower()
                                              for token in str(sentence).split()
                                              if token not in stopwords.words('english')))
        return preprocessed_text

    # Load and preprocess your data (replace with your data processing)
    data = pd.read_csv('News.csv', index_col=0)
    data = data.drop(["title", "subject", "date"], axis=1)
    preprocessed_review = preprocess_text(data['text'].values)
    data['text'] = preprocessed_review

    # Vectorize and train the model
    vectorizer.fit(data['text'])
    x_train = vectorizer.transform(data['text'])
    model.fit(x_train, data['class'])

# Route to home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to classify news
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        
        # Preprocess the input news text
        processed_text = [re.sub(r'[^\w\s]', '', news_text.lower())]
        
        # Vectorize the text using the trained vectorizer
        input_vector = vectorizer.transform(processed_text)
        
        # Predict the news class (real or fake)
        prediction = model.predict(input_vector)[0]
        
        # Return the result
        if prediction == 1:
            result = "Real News"
        else:
            result = "Fake News"
        
        return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    train_model()  # Train the model before starting the server
    app.run(debug=True)
