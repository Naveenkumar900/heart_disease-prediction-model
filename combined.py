import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Framingham dataset
df = pd.read_csv('framingham_modified (1).csv')

# Drop columns with missing values
df = df.dropna()

# Split the data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
#models = [
    #('Logistic Regression', LogisticRegression()),
    #('KMeans', KMeans(n_clusters=2)),
    #('Decision Tree', DecisionTreeClassifier()),
    ###('Random Forest', RandomForestClassifier()),
    #('Neural Network', MLPClassifier(hidden_layer_sizes=(32,16), max_iter=50))
#]
models = [
    ('Logistic Regression', LogisticRegression()),
    ('Random Forest', RandomForestClassifier()),
    ('Neural Network', MLPClassifier(hidden_layer_sizes=(32,16), max_iter=50)),
    ('SVM',SVC(kernel='linear', random_state=0))
]

# Evaluate models
results = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append((name, accuracy, precision, recall, f1))

# Print results in tabular form
df_results = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
print(df_results)
