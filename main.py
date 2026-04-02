from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.knn import KNN
from src.utils import accuracy


def main():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = KNN(k=5)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Custom KNN predictions:", predictions[:10])
    print(f"Custom KNN accuracy: {accuracy(y_test, predictions):.4f}")

    sklearn_model = KNeighborsClassifier(n_neighbors=5)
    sklearn_model.fit(X_train, y_train)

    sklearn_predictions = sklearn_model.predict(X_test)
    print("Sklearn KNN predictions:", sklearn_predictions[:10])
    print(f"Sklearn KNN accuracy: {accuracy(y_test, sklearn_predictions):.4f}")


if __name__ == "__main__":
    main()