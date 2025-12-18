# cyberbullying_detection_using_logistic_regression.py
This project implements a Cyberbullying Detection System using Logistic Regression. It performs end-to-end text classification, including data loading, text preprocessing, exploratory data analysis, TF-IDF feature extraction, model training, evaluation, and visualization.
# 🚨 Cyberbullying Detection Using Logistic Regression

This project implements an **end-to-end Cyberbullying Detection system** using **Natural Language Processing (NLP)** and **Logistic Regression**.  
It classifies tweets into different cyberbullying categories using text preprocessing and TF-IDF features.

---

## 📌 Project Overview

Cyberbullying is a serious online issue.  
This project uses **machine learning and NLP techniques** to automatically detect cyberbullying content from social media text data.

The project covers:
- Text preprocessing
- Feature extraction using TF-IDF
- Logistic Regression model training
- Detailed model evaluation
- Visualization of results

---

## 📂 Project File

cyberbullying_detection_using_logistic_regression.py


---

## 🔍 Steps Covered in the Project

### 1️⃣ Data Loading
- Loaded cyberbullying tweets dataset using Pandas
- Inspected data structure and class labels

### 2️⃣ Exploratory Data Analysis (EDA)
- Class-wise word clouds (before & after preprocessing)
- Dataset size comparison
- Visualization of important terms

### 3️⃣ Text Preprocessing
- Lowercasing text
- Removing URLs, mentions, hashtags, emojis
- Tokenization
- Stopword removal
- Stemming
- Removing duplicates and null values
- Saving cleaned dataset

### 4️⃣ Feature Extraction
- TF-IDF Vectorization (unigrams & bigrams)
- Label encoding of cyberbullying classes
- PCA visualization of features

### 5️⃣ Model Training
- Logistic Regression classifier
- Stratified train-test split
- Training time measurement

### 6️⃣ Model Evaluation
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve (Micro & Macro average)
- MCC and Specificity

### 7️⃣ Model Saving
- Saved trained model using `joblib`
- Saved TF-IDF vectorizer and label encoder

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- WordCloud
- Joblib
- TQDM

---

## 🚀 How to Run the Project

1. Install required libraries:
   ```bash
   pip install numpy pandas nltk scikit-learn matplotlib seaborn wordcloud joblib tqdm
2. Download required NLTK resources (first run only):
   ```bash
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
3. Run the script:
   ```bash
   python cyberbullying_detection_using_logistic_regression.py

---

## 📊 Output
- Cleaned dataset saved as CSV
- TF-IDF features extracted
- Trained Logistic Regression model saved
- Confusion Matrix & ROC Curves
- Detailed performance metrics

---

## Concepts Learned
- NLP preprocessing pipeline
- TF-IDF feature engineering
- Multi-class text classification
- Logistic Regression
- Model evaluation metrics
- ROC & AUC analysis
