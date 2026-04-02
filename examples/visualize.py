import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from src.knn import KNN


def main():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # first 2 features for 2D plotting
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = KNN(k=5)
    model.fit(X_train, y_train)

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        edgecolors="k",
        s=60,
        label="Training data"
    )
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        edgecolors="k",
        marker="^",
        s=90,
        label="Test data"
    )

    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.title("KNN Decision Boundary on Iris Dataset (k=5)")
    plt.legend()

    os.makedirs("images", exist_ok=True)
    plt.savefig("images/decision_boundary.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()