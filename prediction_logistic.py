from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
dataset = pd.read_csv('heart_cleveland_upload.csv')

# Split dataset into features and target variable
X = dataset.drop('target', axis=1)
y = dataset['target']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Create logistic regression model and fit it to the training set
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predict on the testing set
y_pred = lr.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy}')
