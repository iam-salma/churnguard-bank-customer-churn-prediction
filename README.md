# 🧠 Bank Customer Churn Prediction Using ANN

A professional end-to-end machine learning pipeline to predict bank customer churn using an **Artificial Neural Network (ANN)**, complete with **hyperparameter tuning** and a user-friendly **Streamlit web app**.

---

## 🚀 Project Overview

Banking institutions constantly face the challenge of customer retention. This project tackles this issue using deep learning. By analyzing customer behavior and demographics, our ANN-based model predicts the likelihood of a customer leaving the bank (churn).

- 🔍 **Problem Type**: Binary Classification
- 🏦 **Dataset**: Bank Customer Churn Data
- 🤖 **Model**: Artificial Neural Network (Keras + TensorFlow)
- 🎯 **Goal**: Predict whether a customer will churn based on personal and account details
- 🧪 **Tuning**: Hyperparameter tuning with `GridSearchCV`
- 🌐 **Deployment**: Streamlit app for real-time predictions

---

## 🧱 Tech Stack

- **Python 3.10**
- **TensorFlow / Keras**
- **scikit-learn**
- **scikeras**
- **pandas / numpy**
- **Streamlit**

---

## 🧠 ANN Model Architecture

- Input Layer: Features like credit score, age, gender, balance, etc.
- Hidden Layers: Configurable using hyperparameters (neurons and layers)
- Output Layer: Sigmoid activation for binary churn prediction

---

## 🔧 Hyperparameter Tuning

We used **GridSearchCV** (via `scikeras`) to find the optimal combination of:

- `neurons`: Number of neurons per layer
- `layers`: Number of hidden layers
- `epochs`: Training epochs
- `batch_size`: Batch size for training

This helped improve model performance through systematic search and cross-validation.

---

## 💡 Key Features

- 📊 Trained ANN with tuned hyperparameters
- 🔄 Preprocessing pipeline using label encoding, one-hot encoding, and scaling
- 🔮 Real-time prediction of churn probability
- 🌍 Simple UI built with Streamlit
- 📁 Organized model artifacts for reuse (`model.h5`, encoders, scaler)

---

## 🌐 Live App Demo (Streamlit)

The app allows users to input customer details and get a live prediction on their churn probability.

🌐 [Check out the live Streamlit app here](https://churnguard-bank-customer-churn-prediction.streamlit.app/)

## ▶️ Getting Started Locally

### Clone repo
```bash
git clone https://github.com/iam-salma/churnguard-bank-customer-churn-prediction.git
cd churnguard-bank-customer-churn-prediction
```

### Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run Streamlit app
```bash
streamlit run app.py
```

<a href="#top">Back to top</a>