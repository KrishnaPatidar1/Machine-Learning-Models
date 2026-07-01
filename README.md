# Machine Learning Models from Scratch

This repository contains fully vectorized Machine Learning algorithms built entirely from scratch using pure `numpy`. It serves to demonstrate a deep, mathematical understanding of ML architecture, avoiding reliance on high-level libraries like `scikit-learn` for the core engines.

## 🧠 Current Models

### 1. Multiple Linear Regression (`/multiple_linear_regression`)
A complete, vectorized implementation of Multiple Linear Regression.
* **Vectorized Calculus:** Uses NumPy matrix multiplication (`@`) for simultaneous derivative calculations, completely avoiding standard `for` loops.
* **Custom Data Scaling:** Built-in Enum architecture to apply Z-Score Normalization, Min-Max Scaling, Mean Normalization, and Max Scaling.
* **Dynamic Gradient Descent:** Implements an irreducible-error tracking system to dynamically halt training upon convergence, preventing infinite loops on noisy data.
* **scikit-learn Verification:** Includes a benchmark testing script that verifies the custom math against `scikit-learn`'s production models.

### 2. Polynomial Regression
* An extension of Linear Regression to handle non-linear, curved data.
* Mathematically transforms existing features into higher-degree polynomials (e.g., $x^2$, $x^3$) to allow the model to bend the prediction line to fit complex patterns.

### 3. Logistic Regression
* A binary classification engine used to predict discrete categories (0 or 1).
* Built a custom **Sigmoid function** to squish linear outputs into probabilities.
* Implemented **Binary Cross-Entropy (Log Loss)** as the cost function, complete with `numpy` clipping to prevent floating-point `NaN` crashes.
* **Tested on:** Wisconsin Breast Cancer Dataset (Achieved ~97% accuracy on unseen test data).

## ⚙️ Core Engineering Features

Instead of just writing formulas, this repository is built like a modular machine learning framework. Key features include:

* **Custom Scaling Engine:** A built-in `Scaling` class that handles:
  * Z-Score Normalization (Standardization)
  * Min-Max Scaling
  * Mean Normalization
  * Max Scaling
* **Safe Train/Test Splitting:** The models properly memorize the scaling parameters (mean, standard deviation) from the training data and apply them safely to unseen test data to prevent data leakage.
* **Vectorized Math:** Heavily utilizes NumPy's C-level arrays and dot products for lightning-fast training, calculating slopes for thousands of data points instantly.
* **Custom Diagnostics:** Includes mathematical evaluations like Confusion Matrices to track True Positives, False Positives, True Negatives, and False Negatives.