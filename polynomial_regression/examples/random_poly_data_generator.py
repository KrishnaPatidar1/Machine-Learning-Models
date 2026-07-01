import numpy as np

class DataGenerator:
    
    @staticmethod
    def generate_non_linear_data(m=500):
        np.random.seed(42)
        
        # 1. Generating one raw feature (e.g., House Size in 1000s of sq ft)
        X_raw = np.random.uniform(1.0, 5.0, m)
        
        # 2. The True Hidden Math (A Quadratic Curve: ax^2 + bx + c)
        true_w2 = 12.0
        true_w1 = 5.0
        true_b = 200.0
        
        # 3. Adding noise so it looks like real-world data
        noise = np.random.normal(0, 15, m)
        Y = (true_w2 * (X_raw ** 2)) + (true_w1 * X_raw) + true_b + noise
        
        # We return the raw 1D array
        return X_raw, Y