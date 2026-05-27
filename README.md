# Machine Learning Models from Scratch

This repository contains fully vectorized Machine Learning algorithms built entirely from scratch using pure `numpy`. It serves to demonstrate a deep, mathematical understanding of ML architecture, avoiding reliance on high-level libraries like `scikit-learn` for the core engines.

## 🧠 Current Models

### 1. Multiple Linear Regression (`/multiple_linear_regression`)
A complete, vectorized implementation of Multiple Linear Regression.

**Core Features Implemented:**
* **Vectorized Calculus:** Uses NumPy matrix multiplication (`@`) for simultaneous derivative calculations, completely avoiding standard `for` loops.
* **Custom Data Scaling:** Built-in Enum architecture to apply Z-Score Normalization, Min-Max Scaling, Mean Normalization, and Max Scaling.
* **Dynamic Gradient Descent:** Implements an irreducible-error tracking system to dynamically halt training upon convergence, preventing infinite loops on noisy data.
* **scikit-learn Verification:** Includes a benchmark testing script that verifies the custom math against `scikit-learn`'s production models.

## 📂 Architecture

```text
📁 ml_models
 ├── 📁 multiple_linear_regression
 │   ├── mlr_model.py       # Core Math Engine
 │   ├── visuals.py         # Topographical mapping tool
 │   └── 📁 examples        # Data generators and benchmarking scripts
 └── 📁 logistic_regression # (In Progress)