import numpy as np
from enum import Enum


class Scaling(Enum):
    UNSCALED = 0
    MAX_SCALED = 1
    MIN_MAX_SCALED = 2
    MEAN_NORMALIZATION = 3
    Z_SCORE_NORMALIZATION = 4


class LogisticRegressionModel:

    def __init__(self, X, Y):
        ''' X : numpy 2d array containing features
                each row in X represents a data point

            Y : numpy 1d array containing targets
        '''
        self.X = X
        self.Y = Y
        self.w = np.array([0.0])
        self.b = 0

        # number of data points in training dataset
        self.m = self.X.shape[0]

        # number of features in the dataset
        self.n = self.X.shape[1]

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

    def predict(self, X, w = None, b = None):
        '''
        X : numpy 2d array containing features
            each row in X represents a data point
        w : a numpy vector of weights
        b : bias

        This function takes input features (already scaled) and returns the values predicted for the data set

        The function uses the weights and bias given as arguments if provided otherwise uses the object's weights and bias
        '''

        # if no weights and bias are provided
        # we use the object's weights and bias
        if w is None and b is None:
            w = self.w
            b = self.b

        z = (X @ w) + b
        predictions = 1 / (1 + np.exp(-z))
        return predictions
    
    def loss_function(self, f, Y):
        '''
        f : a numpy vector storing the values predicted by the model
            f(x)
        Y : a numpy vector storing the actual targets of the input dataset

        The function returns a numpy vector of loss functions for the dataset
        '''

        loss = ( Y * np.log(f) ) + ( (1 - Y) * np.log(1 - f) )
        loss = -loss
        return loss

    def calculate_cost(self, w, b):
        ''' w : a numpy vector of weights
            b : bias
        The function calculates LOG LOSS cost function for the given weights and bias
        '''

        f = self.predict(self.X, w, b)

        # the output of the sigmoid function may be 1 when the number is very large
        # and can be 0 when the number is very small
        # so we need to clip the values of f between EPSILON and 1 - EPSILON
        EPSILON = 1e-15
        f = np.clip(f, EPSILON, 1 - EPSILON)

        # OPTION 1
        loss = self.loss_function(f, self.Y)
        cost = np.sum(loss) / self.m

        # OPTION 2 (which i accidentally discovered)
        # (OPTIMIZED AND FASTER)
        # cost = (self.Y @ np.log(f) + (1 - self.Y) @ np.log(1 - f))
        # cost = -cost
        # here we directly use '@' operator and avoid the use of loss_function

        return cost
    
    def calculate_derivatives(self, w, b):
        '''
        self.X : a 2d array with m rows and k columns
                 each row represents a training example with k features
        self.Y : an array of size m representing targets(y) of training examples
        w : a vector of weights(w)
        b : bias (constant term)

        X, Y and w are numpy ndarrays

        The function returns the dj_dw and dj_db
        dj_dw : np vector of partial derivatives of cost function wrt. weights
        dj_db : partial derivative of cost function wrt. bias
        '''

        difference = self.predict(self.X, w, b) - self.Y
        dj_dw = (difference @ self.X) / self.m
        dj_db = np.sum(difference) / self.m
        return dj_dw, dj_db
    
    def gradient_descent(self):
        '''
        * finds values of weights and bias such that the cost function is minimun
        * sets the final weights and bias as self.w and self.b

        self.w is a numpy array of size k
        self.b is the final bias
        '''

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

    def scale_test_data(self, X_test):
        '''
        Scales the testing dataset using the parameters calculated from the training dataset
        '''
        if self.scalingStatus == Scaling.MAX_SCALED:
            return X_test / self.maxValues
            
        elif self.scalingStatus == Scaling.MIN_MAX_SCALED:
            return (X_test - self.minValues) / (self.maxValues - self.minValues)
            
        elif self.scalingStatus == Scaling.MEAN_NORMALIZATION:
            return (X_test - self.mean) / (self.maxValues - self.minValues)
            
        elif self.scalingStatus == Scaling.Z_SCORE_NORMALIZATION:
            return (X_test - self.mean) / self.std
            
        return X_test

    # end of LogisticRegressionModel