# Importing the mlr_model file from the parent directory
import sys
from pathlib import Path

# 1. Getting the absolute path of the parent directory
parent_dir = str(Path(__file__).resolve().parent.parent)

# 2. Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# 3. Importing the class from the module
from logistic_model import LogisticRegressionModel

# other modules
from sample_data_generator import ClassificationDataGenerator

# main
if __name__ == "__main__":
    X, Y = ClassificationDataGenerator.generate_simple_binary_data(40)
    model = LogisticRegressionModel(X, Y)
    model.ZScoreNormalization()
    model.gradient_descent()
    print(f"cost = {model.calculate_cost(model.w, model .b)}")

    probabilities = model.predict(model.X)
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = (predictions == Y).mean() * 100
    print(f"Custom Model Accuracy: {accuracy}%")