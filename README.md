
# 📰 Fake News Detection using Machine Learning

A machine learning project built using Python and NLP techniques to classify news articles as **Fake** or **Real**, with a web interface deployed via **Flask**.

---

## 🚀 Project Objective

The aim of this project is to develop a supervised machine learning model that can predict whether a given news article is fake or real based on its content. This is crucial in today’s digital age where misinformation spreads rapidly on social media.

---

## 🧠 Tech Stack & Tools

| Category | Tools/Technologies |
|----------|--------------------|
| Language | Python |
| Libraries | Pandas, NumPy, Scikit-learn, NLTK |
| NLP | Stopword removal, Stemming, TF-IDF |
| ML Models | Logistic Regression, Decision Tree |
| Web App | Flask |
| Deployment | Localhost via Flask |
| Others | HTML, CSS |

---

## 📁 Folder Structure

```
Fake-News-Detection-using-Machine-Learning/
│
├── static/                # CSS files for the web UI
├── templates/             # HTML files for rendering the frontend
│   └── index.html
├── vectorizer.pkl         # TF-IDF vectorizer (saved)
├── model.pkl              # Trained ML model (Logistic Regression)
├── app.py                 # Flask app to serve predictions
├── prediction.py          # Prediction logic (model loading, text processing)
├── requirements.txt       # Required libraries
└── README.md              # You are here!
```

---

## 📊 Dataset

We used a public dataset from **Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)**.

| Feature | Description |
|---------|-------------|
| `text` | Full news article |
| `label` | 0 = Fake, 1 = Real |

---

## 🧪 Machine Learning Workflow

1. **Text Preprocessing**
   - Lowercasing
   - Stopword removal (NLTK)
   - Stemming (PorterStemmer)
   - Tokenization
   - TF-IDF vectorization

2. **Model Building**
   - Logistic Regression (Used for deployment)
   - Decision Tree (For comparison)

3. **Model Evaluation**
   - Accuracy: ~92% with Logistic Regression
   - Confusion Matrix & Classification Report (to be added in notebook)

4. **Web Deployment**
   - Built a Flask app with an input form to enter news text
   - Model prediction shown on the browser

---

## 💻 How to Run the Project Locally

```bash
# 1. Clone the repository
git clone https://github.com/danthulurisaihemanth/Fake-News-Detection-using-Machine-Learning.git
cd Fake-News-Detection-using-Machine-Learning

# 2. Create virtual environment and install dependencies
pip install -r requirements.txt

# 3. Run the Flask app
python app.py

# 4. Open your browser at:
http://127.0.0.1:5000
```

---

## 📈 Sample Output Screenshots

*Add screenshots here of the app running, input form, and prediction output.*

---

## 🧪 Results & Model Performance

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | ~92%     | High      | High   | High     |
| Decision Tree       | ~88%     | Medium    | Medium | Medium   |

---

## 📌 Future Enhancements

- Add deep learning models (e.g., LSTM or BERT).
- Integrate real-time news article fetching (e.g., from RSS feeds or APIs).
- Improve UI with Bootstrap or React.
- Deploy using Streamlit or Dockerized app on cloud (Heroku, Render, or AWS EC2).

---

## 🙌 Acknowledgements

- [Kaggle Dataset - Fake and Real News](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Scikit-learn and NLTK Documentation
- Flask for web deployment

---

## 📬 Contact

**Danthuluri Sai Hemanth Varma**  
📧 saihemanthdanthuluri03@gmail.com  
🔗 [GitHub Profile](https://github.com/danthulurisaihemanth)

---

> ⚠ Note: This project is built for educational and internship purposes and does not reflect real-world deployment-scale fake news detection systems.
