from random_poly_data_generator import DataGenerator

import sys
from pathlib import Path

# importing Polynomial Regression Model
# 1. Getting the absolute path of the parent directory
parent_dir = str(Path(__file__).resolve().parent.parent)

# 2. Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# 3. Importing the class from the module
from poly_model import PolynomialRegressionModel


# main
if __name__ == "__main__":
    raw_X, Y = DataGenerator.generate_non_linear_data(100000)

    model = PolynomialRegressionModel(raw_X, Y, n = 3)

    model.ZScoreNormalization()
    model.gradient_descent()

    print("\nGradient Descent generated values")
    print(f"w1 = {model.w[0]}")
    print(f"w2 = {model.w[1]}")
    print(f"w3 = {model.w[2]}")
    print(f"b = {model.b}")