# Fake News Detection using Machine Learning 📰🚫

This project focuses on detecting fake news using machine learning algorithms such as Logistic Regression and Decision Tree in Python. The dataset used for this project contains news articles labeled as either "real" or "fake." The objective of this project is to build a model that classifies news articles into one of these two categories.

## Project Overview🔍

The goal of this project is to implement two machine learning algorithms (Logistic Regression and Decision Tree) to predict whether a given news article is fake or real. 

## Technologies Used 💻

- Python 3
- Jupyter Notebook
- pandas
- seaborn
- matplotlib
- NLTK (Natural Language Toolkit)
- scikit-learn
- WordCloud
- tqdm

## Dataset 📊
[Download Dataset from Google Drive](https://drive.google.com/file/d/1EIe30Afif5kdhX4IOZalc135pEWUpVNZ/view?usp=sharing)

The dataset used in this project contains news articles labeled as "real" or "fake." It includes features such as the title, text, and date of the article. For the purpose of this project, the title, subject, and date columns were removed, as they were not helpful for classification.

## The dataset used for this project is too large to be uploaded to GitHub. You can download it from Google Drive using the link below:

[Download Dataset from Google Drive](https://drive.google.com/file/d/1EIe30Afif5kdhX4IOZalc135pEWUpVNZ/view?usp=sharing)

### Instructions
1. Download the file from the link above.
2. Place the file in the project directory, inside a folder named `data/`.
3. Ensure the file is named `News.csv` before running the code.

If you encounter any issues, ensure you have proper access to the file or contact the project owner.


## Installation ⚙️

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git

2.Install the necessary Python libraries:

pip install -r requirements.txt

## File Structure 📁

## notebooks/: Contains Jupyter notebooks with the code for data exploration, preprocessing, and model training.
## data/: Contains the dataset used for training and testing the models.
## requirements.txt: A list of Python dependencies required for this project.

## Usage 📜

1. Open the Jupyter notebook fake_news_detection.ipynb

2. Run the cells to perform the following steps:

· Load and preprocess the data.

. Perform exploratory data analysis (EDA) including visualizations like WordClouds and bar graphs.

· Preprocess text data by removing stopwords, punctuations, and irrelevant spaces.

. Train the Logistic Regression model and evaluate its performance.

. Split data into training and test sets and convert the text data into vectors using TfidfVectorizer.

. Evaluate the model's accuracy using accuracy_score

## PREPROCESSING AND VISUALIZATION 🛠️

. Data Cleaning: Drop unnecessary columns ( title,subject , and date ), handle null values,and shuffle the dataset to prevent model bias.

· Text Preprocessing: Remove stopwords and punctuation, tokenize text, and perform stemming.

. WordCloud: Visualized word clouds separately for "real" and "fake" news to identify frequent
terms.

. Top Words: Visualized the top 20 most frequent words using a bar chart.

## MODEL TRAINING & EVALUATION ⚖️
The model uses the Logistic Regression algorithm for classification. The data is vectorized using TfidfVectorizer before being fed into the model. After training, the model is evaluated on both the training and test datasets.

The results show a high accuracy:

## Training Accuracy: 99.38%
## Test Accuracy: 98.93%

## Flask App for Fake News Detection 🚀

This project also includes a basic Flask web app for fake news detection using the trained Logistic Regression model.

### How to Run the Flask App:

1. Clone the repository or download the project.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Navigate to the `flask_app` directory:
   ```bash
   cd flask_app
   ```
4. Run the Flask app:
   ```bash
   python app.py
   ```
5. Open a browser and go to `http://127.0.0.1:5001/` to use the fake news detection app.

### How It Works:
- You can enter any news text, and the model will predict whether it’s "Real News" or "Fake News".

## CONCLUSION 🎯
This project demonstrates the effectiveness of Logistic Regression for detecting fake news. Further improvements can be made by exploring other machine learning algorithms, fine-tuning the model, or using deep learning techniques.

## LICENSE 📝
This project is licensed under the MIT License - see the LICENSE file for details.

## Links 🌐
GitHub Repository: Your GitHub Link
Project Demo: Project Demo Link
