# 🩺 Efficient Machine Learning Model for Diabetes Prediction

**Author:** Shreya Jadhav  
**Roll No:** 16014223077  

---

## 📘 Overview
This project reproduces and extends the research paper  
**“Efficient Machine Learning Model for Diabetes Prediction” (Khanam & Foo, 2021)**  
to develop an accurate and scalable diabetes prediction system using multiple machine learning algorithms.

Two datasets were used for experimentation:
1. **Pima Indians Diabetes Dataset** *(baseline model)*
2. **Kaggle BRFSS 2015 Diabetes Health Indicators Dataset** *(extended large-scale version)*

---

## 🧠 Objectives
- Implement and compare various machine learning algorithms for diabetes prediction.  
- Evaluate models based on multiple performance metrics.  
- Extend the methodology to a larger, real-world dataset for better generalization.  
- Integrate a simple Artificial Neural Network (ANN) for performance benchmarking.

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Replaced zero or missing values with **mean imputation**.  
- Removed outliers using the **IQR (Interquartile Range)** method.  
- Performed **feature selection** using Pearson correlation (> 0.2).  
- Normalized data with **MinMaxScaler (0–1 scaling)**.  
- Train-test split: **85% training, 15% testing** (stratified).

### 2. Implemented Models
- Logistic Regression  
- Naive Bayes (GaussianNB)  
- Support Vector Machine (SVM - Linear Kernel)  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbors (KNN)  
- AdaBoost  
- Artificial Neural Network *(2 hidden layers, 400 epochs, ReLU activations)*

### 3. Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC–AUC  

### 4. Visualizations
- Confusion Matrices  
- ROC Curves  
- Accuracy Comparison Bar Charts  
- Neural Network Training vs Validation Accuracy Plot  

---

## 📊 Key Findings
- Random Forest and Neural Network achieved the **best predictive performance**.  
- The Kaggle BRFSS dataset improved **model robustness and generalization**.  
- The results were consistent with the **original research paper’s conclusions**.  

---


## 📈 Future Improvements
- Add hyperparameter tuning (GridSearchCV).
- Implement cross-validation for robust evaluation.
- Integrate explainable AI for feature importance interpretation.
- Deploy as a web app for real-time diabetes risk prediction.

---

📚 References
- Khanam & Foo (2021). Efficient Machine Learning Model for Diabetes Prediction. ICT Express.
- Kaggle Dataset – Diabetes Health Indicators (BRFSS 2015)
- UCI Repository – Pima Indians Diabetes Database
