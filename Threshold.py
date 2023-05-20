import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the heart disease dataset (assuming it's in a CSV file)
data = pd.read_csv("heart.csv")

# Split the dataset into features (X) and target (y)
X = data.drop("target", axis=1)
y = data["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize classifiers
logreg_model = LogisticRegression()
nn_model = MLPClassifier(hidden_layer_sizes=(16, 8))
rf_model = RandomForestClassifier()
svm_model = SVC(probability=True)

# Train classifiers
logreg_model.fit(X_train, y_train)
nn_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Predict probabilities for the positive class (class 1)
logreg_pred_prob = logreg_model.predict_proba(X_test)[:, 1]
nn_pred_prob = nn_model.predict_proba(X_test)[:, 1]
rf_pred_prob = rf_model.predict_proba(X_test)[:, 1]
svm_pred_prob = svm_model.predict_proba(X_test)[:, 1]

# Calculate the false positive rate (FPR) and true positive rate (TPR) for each classifier
logreg_fpr, logreg_tpr, logreg_thresholds = roc_curve(y_test, logreg_pred_prob)
nn_fpr, nn_tpr, nn_thresholds = roc_curve(y_test, nn_pred_prob)
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_pred_prob)
svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, svm_pred_prob)

# Calculate the threshold values based on the ROC curve
logreg_threshold = logreg_thresholds[np.argmax(logreg_tpr - logreg_fpr)]
nn_threshold = nn_thresholds[np.argmax(nn_tpr - nn_fpr)]
rf_threshold = rf_thresholds[np.argmax(rf_tpr - rf_fpr)]
svm_threshold = svm_thresholds[np.argmax(svm_tpr - svm_fpr)]

# Print the calculated threshold values
print("Logistic Regression Threshold:", logreg_threshold)
print("Neural Network Threshold:", nn_threshold)
print("Random Forest Threshold:", rf_threshold)
print("SVM Threshold:", svm_threshold)
