import numpy as np
from enum import Enum


class Scaling(Enum):
    UNSCALED = 0
    MAX_SCALED = 1
    MIN_MAX_SCALED = 2
    MEAN_NORMALIZATION = 3
    Z_SCORE_NORMALIZATION = 4


class MultipleLinearRegressionModel:

    def __init__(self, X, Y):
        ''' X : numpy 2d array containing features
                each row in X represents a data point

            Y : numpy 1d array containing targets
        '''
        self.X = X
        self.Y = Y
        self.scalingStatus = Scaling.UNSCALED

    def maxScale_X(self):
        self.maxValues = np.max(self.X, axis=0)
        self.X = self.X / self.maxValues
        self.scalingStatus = Scaling.MAX_SCALED

    def minMaxScaling(self):
        self.maxValues = np.max(self.X, axis=0)
        self.minValues = np.min(self.X, axis=0)
        self.X = (self.X - self.minValues) / (self.maxValues - self.minValues)
        self.scalingStatus = Scaling.MIN_MAX_SCALED

    def meanNormalization(self):
        self.maxValues = np.max(self.X, axis=0)
        self.minValues = np.min(self.X, axis=0)
        self.mean = np.mean(self.X, axis=0)
        self.X = (self.X - self.mean) / (self.maxValues - self.minValues)
        self.scalingStatus = Scaling.MEAN_NORMALIZATION

    def ZScoreNormalization(self):
        self.mean = np.mean(self.X, axis=0)
        self.std = np.std(self.X, axis=0)
        self.X = (self.X - self.mean) / (self.std)
        self.scalingStatus = Scaling.Z_SCORE_NORMALIZATION

    def calculate_cost(self, w, b):
        ''' w : a numpy vector of weights
            b : bias
        The function calculates SQUARED ERROR cost function for the given weights and bias
        '''
        m = len(self.X)
        error = (self.X @ w) + b - self.Y
        return np.sum(error ** 2) / (2 * m)

    def calculate_derivatives(self, w, b):
        '''
        self.X : a 2d array with m rows and k columns
                 each row represents a training example with k features
        self.Y : an array of size m representing targets(y) of training examples
        w : a vector of weights(w)
        b : bias (constant term)

        X, Y and w are numpy ndarrays
        '''
        m = len(self.Y)  # number of training examples

        # an array of errors of m data points
        error = (self.X @ w) + b - self.Y

        # an array of k values
        # representing derivatives of cost function wrt. the weights
        dj_dw = (error @ self.X) / m
        dj_db = np.sum(error) / m
        return (dj_dw, dj_db)

    def gradient_descent(self):
        '''
        * finds values of weights and bias such that the cost function is minimun
        * sets the final weights and bias as self.w and self.b

        self.w is a numpy array of size k
        self.b is the final bias
        '''
        # print("\nGradient Descent function called")

        #constants
        alpha = 0.01
        EPSILON = 1e-3
        k = len(self.X[0]) # number of features

        self.w = np.array([0.0] * k)
        self.b = 0
        cost = self.calculate_cost(self.w, self.b)
        prev_cost =  cost + 5

        while( abs(prev_cost - cost) > EPSILON):
        
            prev_cost = cost

            dj_dw, dj_db = self.calculate_derivatives(self.w, self.b)

            self.w = self.w - alpha * dj_dw
            self.b = self.b - alpha * dj_db
            cost = self.calculate_cost(self.w, self.b)

        # end of gradient descent

# end of class MultipleLinearRegressionModel