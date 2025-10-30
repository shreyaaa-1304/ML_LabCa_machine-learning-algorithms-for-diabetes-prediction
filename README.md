**Author:** Shreya Jadhav
**Roll No:** 16014223077

---

### 1. Problem Definition

Develop an efficient machine learning model for predicting diabetes using publicly available datasets. Compare multiple algorithms to identify the best-performing model for accurate diabetes prediction.

---

### 2. Literature Study

The methodology reproduces and extends the research paper *“Efficient Machine Learning Model for Diabetes Prediction” (Khanam & Foo, 2021)*. Various supervised learning algorithms were implemented to analyze diabetes prediction performance using real-world data.

---

### 3. Data Collection

* **Primary Dataset:** Pima Indians Diabetes Dataset (baseline implementation)
* **Extended Dataset:** Kaggle *Diabetes Health Indicators (BRFSS 2015)* dataset for scalability and realism.

---

### 4. Data Preprocessing

* Handled missing values using mean imputation.
* Removed outliers using the Interquartile Range (IQR) method.
* Performed feature selection using Pearson correlation (>0.2).
* Normalized features using **MinMaxScaler**.
* Split dataset into **85% training** and **15% testing** sets (stratified sampling).

---

### 5. Model Implementation

Implemented and trained the following machine learning models using **Scikit-learn** and **TensorFlow**:

* Logistic Regression
* Naive Bayes (GaussianNB)
* Support Vector Machine (SVM)
* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors (KNN)
* AdaBoost Classifier
* Artificial Neural Network (Sequential Model with 2 hidden layers, 400 epochs)

---

### 6. Hyperparameter Tuning

No explicit hyperparameter tuning (e.g., GridSearchCV) was performed. Default parameters were used for fairness and simplicity across all models.

---

### 7. Model Evaluation

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC–AUC

Visualizations included:

* Confusion matrices for all models
* ROC curves for model comparison
* Accuracy bar chart comparison

---

### 8. Result Analysis

All models were compared using evaluation metrics. Random Forest and Neural Network models achieved the highest accuracy. The results were consistent with findings from the original research paper.

---

### 9. Dataset Extension (Improvement)

The implementation was extended from the smaller **Pima Indians dataset** to the larger **Kaggle BRFSS 2015 dataset**, improving model scalability and robustness. Data preprocessing and evaluation steps were adapted to handle higher dimensionality.

---

### ✅ Summary

All steps — from literature study to model evaluation — were successfully implemented, except explicit hyperparameter tuning. The project effectively demonstrates reproducibility, adaptability, and comparative performance analysis of multiple ML models for diabetes prediction.

