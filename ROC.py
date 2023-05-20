import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the heart disease dataset (assuming it's in a CSV file)
data = pd.read_csv("heart_cleveland_upload.csv")

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
logreg_fpr, logreg_tpr, _ = roc_curve(y_test, logreg_pred_prob)
nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_pred_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_prob)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_pred_prob)

# Calculate the area under the ROC curve (AUC)
logreg_auc = roc_auc_score(y_test, logreg_pred_prob)
nn_auc = roc_auc_score(y_test, nn_pred_prob)
rf_auc = roc_auc_score(y_test, rf_pred_prob)
svm_auc = roc_auc_score(y_test, svm_pred_prob)

# Plot the ROC curves
plt.plot(
    logreg_fpr,
    logreg_tpr,
    label="Logistic Regression (AUC = {:.2f})".format(logreg_auc),
)
plt.plot(nn_fpr, nn_tpr, label="Neural Network (AUC = {:.2f})".format(nn_auc))
plt.plot(rf_fpr, rf_tpr, label="Random Forest (AUC = {:.2f})".format(rf_auc))
plt.plot(svm_fpr, svm_tpr, label="SVM (AUC = {:.2f})".format(svm_auc))
plt.plot([0, 1], [0, 1], "k--")  # Diagonal line for random guessing
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curves")
plt.legend(loc="lower right")
plt.show()
