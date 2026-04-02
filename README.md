# 🔍 K-Nearest Neighbors (KNN) From Scratch

## 📌 Overview

This project implements the **K-Nearest Neighbors (KNN)** algorithm from scratch using **NumPy**, without relying on machine learning frameworks.

It demonstrates a clear understanding of:

* Distance-based learning
* Model evaluation
* Algorithm design from first principles

---

## 🚀 Features

* ✅ KNN Classifier (from scratch)
* ✅ Euclidean distance computation
* ✅ Majority voting
* ✅ Accuracy evaluation
* ✅ Real dataset support (Iris)
* ✅ Decision boundary visualization (matplotlib)
* ✅ Comparison with sklearn implementation

---

## 🧠 How KNN Works

For each input sample:

1. Compute distance to all training points
2. Select the **K nearest neighbors**
3. Perform **majority voting**
4. Assign the most common class

KNN is a **lazy learning algorithm** — it does not learn a model, it stores the data.

---

## 📊 Complexity

| Step       | Complexity |
| ---------- | ---------- |
| Training   | O(1)       |
| Prediction | O(n × d)   |

* `n` = number of samples
* `d` = number of features

---

## 📁 Project Structure

```
knn-from-scratch/
│
├── src/
│   ├── __init__.py
│   ├── knn.py
│   ├── train.py
│   └── utils.py
│
├── examples/
│   └── visualize.py
│
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/KNN-From-Scratch.git
cd KNN-From-Scratch

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## ▶️ Usage

### Run basic example

```bash
python examples/basic_test.py
```

### Run visualization (recommended)

```bash
python examples/visualize.py
```

---

## 🎨 Decision Boundary Visualization

This project includes visualization of how KNN separates data in feature space.

* Uses first 2 features of Iris dataset
* Shows how classification regions change
* Helps understand model behavior

---

## 📈 Example Output

```
Predictions: [1 0]
Accuracy: 0.95
```

---

## 🔬 Comparison with sklearn

The implementation is validated against sklearn’s KNN classifier to ensure correctness.

---

## 💡 Why This Project Matters

This project demonstrates:

* Strong understanding of core ML concepts
* Ability to implement algorithms from scratch
* Clean project structure and reproducibility

---

## 👤 Author

Arsen Poghosyan

