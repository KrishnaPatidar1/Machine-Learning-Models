import numpy as np

class ClassificationDataGenerator:
    @staticmethod
    def generate_simple_binary_data(m=40):
        '''
        Generates a small, easily separable dataset for testing binary classification.
        Returns X (m x 2 matrix) and Y (array of m binary targets).
        '''
        np.random.seed(42) # Keeps the random data the exact same every time you run it

        m_half = m // 2

        # -----------------------------------------------------
        # CATEGORY 0 (Failing Students: Low study, low sleep)
        # -----------------------------------------------------
        # Centers data around [3.0, 3.0] with a little bit of random spread (noise)
        X_0 = np.random.normal(loc=3.0, scale=1.0, size=(m_half, 2))
        Y_0 = np.zeros(m_half)

        # -----------------------------------------------------
        # CATEGORY 1 (Passing Students: High study, high sleep)
        # -----------------------------------------------------
        # Centers data around [7.0, 7.0]
        X_1 = np.random.normal(loc=7.0, scale=1.0, size=(m_half, 2))
        Y_1 = np.ones(m_half)

        # Combine the two groups into one dataset
        X = np.vstack((X_0, X_1))
        Y = np.concatenate((Y_0, Y_1))

        # Shuffle the data (Crucial!)
        # If we don't shuffle, Gradient Descent sees all 0s first, then all 1s, which confuses it.
        shuffle_indices = np.random.permutation(m)
        X = X[shuffle_indices]
        Y = Y[shuffle_indices]

        return X, Y