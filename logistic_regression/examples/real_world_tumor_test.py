# importing required modules
import sys
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# getting the absolute path of the parent directory
parent_dir = str(Path(__file__).resolve().parent.parent)

# add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# importing the class from the module
from logistic_model import LogisticRegressionModel

# main
if __name__ == "__main__":
    
    # loading breast cancer dataset from sklearn
    data = load_breast_cancer()
    X = data.data 
    Y = data.target

    # splitting the data into training and testing sets
    # 80% for training and 20% for testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # creating the model using training data
    model = LogisticRegressionModel(X_train, Y_train)
    
    # applying z-score normalization
    model.ZScoreNormalization()
    
    # finding the best weights and bias
    model.gradient_descent()
    
    # scaling the test data using training parameters
    X_test_scaled = model.scale_test_data(X_test)
    
    # predicting values for the test data
    probabilities = model.predict(X_test_scaled)
    
    # converting probabilities to 0 or 1 based on 0.5 threshold
    predictions = (probabilities >= 0.5).astype(int)
    
    # calculating accuracy of the model on test data
    accuracy = (predictions == Y_test).mean() * 100
    print(f"\nAccuracy on test data : {accuracy:.2f}%")

    # confusion matrix
    true_positives = np.sum((predictions == 1) & (Y_test == 1))
    true_negatives = np.sum((predictions == 0) & (Y_test == 0))
    false_positives = np.sum((predictions == 1) & (Y_test == 0)) 
    false_negatives = np.sum((predictions == 0) & (Y_test == 1)) 

    print("\nConfusion Matrix :")
    print(f"True Negatives : {true_negatives}")
    print(f"True Positives : {true_positives}")
    print(f"False Positives : {false_positives}")
    print(f"False Negatives : {false_negatives}")

# end of main