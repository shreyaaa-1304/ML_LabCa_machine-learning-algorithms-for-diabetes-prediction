# A Comparison of Machine Learning Algorithms for Diabetes Prediction

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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

#  2. Load Dataset (Pima Indians Diabetes)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DPF","Age","Outcome"]
data = pd.read_csv(url, names=cols)

print("Initial shape:", data.shape)
data.head()

# 3. Handle Missing Values (replace 0s with mean)
zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
data[zero_cols] = data[zero_cols].replace(0, np.nan)
data.fillna(data.mean(), inplace=True)

# 4. Outlier Removal (IQR Method)
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
print("After removing outliers:", data.shape)

#  5. Feature Selection using Pearson Correlation (>0.2)
corr = data.corr()['Outcome'].abs().sort_values(ascending=False)
selected_features = corr[corr > 0.2].index.tolist()
selected_features.remove('Outcome')
print("Selected Features:", selected_features)

# 6. Normalization (0-1 scaling)
X = data[selected_features]
y = data['Outcome']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 7. Train-Test Split (85/15)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)

#  8. Train & Evaluate ML Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel='linear', probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "AdaBoost": AdaBoostClassifier()
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

# 9. Results Summary
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
print("\nPerformance Summary:\n")
display(results_df)

# 10. Confusion Matrices
for name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# 11. ROC Curves for All ML Models
plt.figure(figsize=(8,6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Machine Learning Models')
plt.legend()
plt.show()

# 2. Neural Network (2 Hidden Layers, 400 Epochs)
model_nn = Sequential([
    Dense(5, input_dim=len(selected_features), activation='relu'),
    Dense(26, activation='relu'),
    Dense(5, activation='relu'),
    Dense(1, activation='sigmoid')
])

opt = SGD(learning_rate=0.01)
model_nn.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
history = model_nn.fit(X_train, y_train, validation_data=(X_test, y_test),
                       epochs=400, verbose=0)

# 13. NN Performance
loss, acc = model_nn.evaluate(X_test, y_test, verbose=0)
print(f"\nNeural Network Accuracy: {acc*100:.2f}%")

# 14. Training vs Validation Accuracy Plot
plt.figure(figsize=(7,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Neural Network Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 15. Bar Chart Comparison
plt.figure(figsize=(8,5))
sns.barplot(x="Model", y="Accuracy", data=results_df, palette="crest")
plt.title("Accuracy Comparison of ML Models")
plt.xticks(rotation=45)
plt.show()
