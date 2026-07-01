'''
Polynomial Regression (Single Feature)

Using MultipleLinearRegressionModel for single feature Polynomial Regression
'''
import numpy as np

# importing mlr_model
import sys
from pathlib import Path

# 1. Get the directory of the current script
current_dir = Path(__file__).resolve().parent

# 2. Navigate to the neighbor directory (folder_b)
neighbor_dir = current_dir.parent / "multiple_linear_regression"

# 3. Append the neighbor directory path to sys.path
sys.path.append(str(neighbor_dir))

# 4. Import your module normally
from mlr_model import MultipleLinearRegressionModel # type: ignore[reportMissingImports]


class PolynomialRegressionModel(MultipleLinearRegressionModel):
    '''
    Single feature polynomial regression model
    '''

    def __init__(self, X, Y, n: int):
        '''
        X : numpy 2d array containing features
            each row in X represents a data point
        Y : numpy 1d array containing targets
        n : exponent for polynomial regression
        '''

        self.n = n
        new_X = np.zeros((X.shape[0], self.n))

        for i in range(self.n):
            new_X[:, i] = X ** (i+1)
        
        super().__init__(new_X, Y)