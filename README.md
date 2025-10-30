Efficient Machine Learning Model for Diabetes Prediction
Author: Shreya Jadhav

Roll No: 16014223077

📘 Overview

This project reproduces and extends the research paper
“Efficient Machine Learning Model for Diabetes Prediction” (Khanam & Foo, 2021)
to develop an accurate and scalable diabetes prediction system using multiple machine learning algorithms.

Two datasets were used for experimentation:

Pima Indians Diabetes Dataset (Baseline model)

Kaggle BRFSS 2015 Diabetes Health Indicators Dataset (Extended, large-scale version)

🧠 Objectives

Implement and compare various machine learning algorithms for diabetes prediction.

Evaluate models based on multiple performance metrics.

Extend the methodology to a larger, real-world dataset for better generalization.

Integrate a simple Artificial Neural Network (ANN) for performance benchmarking.

⚙️ Methodology
1. Data Preprocessing

Replaced zero or missing values with mean imputation.

Removed outliers using the IQR (Interquartile Range) method.

Performed feature selection using Pearson correlation (> 0.2).

Normalized data with MinMaxScaler (0–1 scaling).

Train-test split: 85% training, 15% testing (stratified).

2. Implemented Models

Logistic Regression

Naive Bayes (GaussianNB)

Support Vector Machine (SVM - Linear Kernel)

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

AdaBoost

Artificial Neural Network (2 hidden layers, 400 epochs, ReLU activations)

3. Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

ROC–AUC

4. Visualization

Confusion Matrices

ROC Curves

Accuracy Comparison Bar Charts

Neural Network Training vs Validation Accuracy Plot

🧾 Results Summary
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	~78%	-	-	-
Naive Bayes	~76%	-	-	-
Random Forest	~82–84%	-	-	-
Neural Network	~83–85%	-	-	-

(Exact values vary slightly between Pima and Kaggle datasets.)

📊 Key Findings

Random Forest and Neural Network achieved the best predictive performance.

The Kaggle BRFSS dataset improved model robustness and generalization.

The results aligned closely with the original study’s conclusions.

🧩 Project Structure
📁 diabetes_prediction/
│
├── 📄 diabetes_pima_model.py           # Pima Indians Diabetes implementation
├── 📄 diabetes_kaggle_brfss_model.py   # Kaggle BRFSS 2015 implementation
├── 📄 README.md                        # Project documentation
├── 📄 requirements.txt                 # Dependencies
└── 📁 results/                         # Plots, confusion matrices, ROC curves

🛠️ Technologies Used

Python 3.x

NumPy, Pandas, Matplotlib, Seaborn

Scikit-learn

TensorFlow / Keras

📈 Future Improvements

Add hyperparameter tuning (GridSearchCV).

Implement cross-validation for more robust metrics.

Integrate explainable AI techniques for feature importance analysis.

Deploy as a simple web app for real-time diabetes risk prediction.

📚 References

Khanam & Foo (2021). Efficient Machine Learning Model for Diabetes Prediction. ICT Express.

Kaggle Dataset: Diabetes Health Indicators (BRFSS 2015)

UCI Repository: Pima Indians Diabetes Database
