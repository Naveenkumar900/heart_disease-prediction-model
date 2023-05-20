from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
dataset = pd.read_csv('framingham_modified (1).csv')

# Split dataset into features and target variable
X = dataset.drop('target', axis=1)
y = dataset['target']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create decision tree model and fit it to the training set
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Predict on the testing set
y_pred = dt.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Decision Tree Accuracy: {accuracy}')
