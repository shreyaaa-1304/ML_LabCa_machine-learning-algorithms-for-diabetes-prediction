# Efficient Machine Learning Model for Diabetes Prediction
# Dataset: Kaggle - Diabetes Health Indicators (BRFSS 2015)

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# 2. Try Loading the Dataset

print(" Looking for Kaggle file upload...")

try:
    # Preferred: Upload from Kaggle manually (BRFSS 2015 dataset)
    data = pd.read_csv("/content/diabetes_binary_health_indicators_BRFSS2015.csv")
    print(" Loaded local Kaggle dataset successfully.")
except FileNotFoundError:
    # Fallback option if Kaggle file is not uploaded
    print(" Kaggle file not found. Using backup online dataset instead.")
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    data = pd.read_csv(url)

print("Shape:", data.shape)
print("\nDataset Preview:")
display(data.head())

# 3. Data Preprocessing

# If the dataset has 'Diabetes_binary' column (Kaggle BRFSS)
if 'Diabetes_binary' in data.columns:
    y = data['Diabetes_binary']
    X = data.drop(columns=['Diabetes_binary'])
else:
    # fallback dataset target
    y = data['Outcome']
    X = data.drop(columns=['Outcome'])

# Fill missing values if any
X = X.replace([np.inf, -np.inf], np.nan)
X.fillna(X.mean(), inplace=True)

# 4. Outlier Removal (IQR)
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
X = X[mask]
y = y[mask]

print("After removing outliers:", X.shape)

# 5. Feature Selection (Pearson Correlation > 0.2)
corr = pd.concat([X, y], axis=1).corr()['Diabetes_binary' if 'Diabetes_binary' in data.columns else 'Outcome'].abs()
selected_features = corr[corr > 0.2].index.tolist()
if 'Diabetes_binary' in selected_features:
    selected_features.remove('Diabetes_binary')
elif 'Outcome' in selected_features:
    selected_features.remove('Outcome')
print("Selected Features:", selected_features)

X = X[selected_features]

# 6. Normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)

# Insert this import in Step 1
from imblearn.over_sampling import SMOTE

# --- Modify Step 8: ML Models (with Class Weighting) ---

# Note: Not all models support class_weight='balanced'.
# We apply it to those that do (LR, SVM, RF, DT, AdaBoost)
models = {
    "Logistic Regression (Balanced)": LogisticRegression(max_iter=200, class_weight='balanced'),
    "Naive Bayes": GaussianNB(), # Does not support class_weight
    "SVM (Balanced)": SVC(kernel='linear', probability=True, class_weight='balanced'),
    "Decision Tree (Balanced)": DecisionTreeClassifier(class_weight='balanced'),
    "Random Forest (Balanced)": RandomForestClassifier(class_weight='balanced'),
    "KNN": KNeighborsClassifier(n_neighbors=7), # Does not support class_weight
    "AdaBoost (Balanced)": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced')),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append([name, acc, prec, rec, f1])

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
print("\n Model Performance Summary:\n")
display(results_df)

# 9. Confusion Matrices
for name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# 10. ROC Curves
plt.figure(figsize=(8,6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')

plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for ML Models (Diabetes Prediction)')
plt.legend()
plt.show()

# 11. Neural Network (2 Hidden Layers, 400 Epochs)
model_nn = Sequential([
    Dense(len(selected_features), input_dim=len(selected_features), activation='relu'),
    Dense(26, activation='relu'),
    Dense(5, activation='relu'),
    Dense(1, activation='sigmoid')
])

opt = SGD(learning_rate=0.01)
model_nn.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
history = model_nn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=400, verbose=0)

# 12. ANN Evaluation
loss, acc = model_nn.evaluate(X_test, y_test, verbose=0)
print(f"\n Neural Network Accuracy: {acc*100:.2f}%")

# 13. Plot NN Training Curve
plt.figure(figsize=(7,5))
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Neural Network Accuracy (Diabetes Dataset)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 14. Accuracy Comparison
plt.figure(figsize=(8,5))
sns.barplot(x="Model", y="Accuracy", data=results_df, palette="crest")
plt.title("Accuracy Comparison of ML Models (Diabetes Dataset)")
plt.xticks(rotation=45)
plt.show()
