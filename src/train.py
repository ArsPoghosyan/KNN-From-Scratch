import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN
from utils import accuracy
# 1. Load a dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# 3. Instantiate and test our custom model
k = 3
clf = KNN(k=k)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# 4. Evaluate
print(f"Custom KNN classification accuracy: {accuracy(y_test, predictions) * 100:.2f}%")