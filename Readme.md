# 🏡 House Price Prediction App

A Streamlit web application to predict house sale prices using a trained machine learning model (Lasso-based feature selection + Linear Regression).

📍 **Live Demo**: [house-prediction-csi.streamlit.app](https://house-prediction-csi.streamlit.app)

---

## 📦 Features

- Upload or manually input house data
- Predict sale price based on selected features
- Interactive UI powered by Streamlit
- Model trained with:
  - `train_preprocessed.csv`
  - Lasso-based feature selection
  - Linear Regression (sklearn)

---

## 🚀 App Preview

![App Screenshot](https://raw.githubusercontent.com/your-username/your-repo-name/main/screenshot.png)

---

## 🧠 How It Works

1. **Data Preprocessing**
   - Handled missing values
   - Label encoded categoricals
   - Feature engineered TotalSF, HouseAge, RemodAge

2. **Feature Selection**
   - LassoCV → SelectFromModel → Chose top predictors

3. **Model**
   - Linear Regression trained on Lasso-selected features
   - Saved as `house_model.pkl`

---

## 🗂️ Project Structure
```
📁 House_Prediction_App/
├── app.py # Streamlit frontend
├── train_model.py # Script to train and save model
├── train.csv # Original training data
├── train_preprocessed.csv # Cleaned dataset used for training
├── house_model.pkl # Trained model
├── selected_features.pkl # Saved selected features
├── requirements.txt # Dependencies
└── README.md # This file
```
