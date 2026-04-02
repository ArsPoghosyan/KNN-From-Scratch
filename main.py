import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN
from utils import accuracy

# Load real dataset (Iris)
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = KNN(k=5)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

print("Predictions:", predictions[:10])
print("Accuracy:", accuracy(y_test, predictions))

# Comparing with SKlearn
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

print("Sklearn Predictions:", clf.predict(X_test)[:10])
print("Sklearn Predictions Accuracy:", accuracy(y_test, clf.predict(X_test)))
