import numpy as np

class DataGenerator:

    @staticmethod
    def generate_random_housePrice_data(m: int):
        rg = np.random.default_rng(seed=42)

        area = rg.integers(750, 10001, m)
        bedrooms = rg.integers(1, 11, m)
        floors = rg.integers(1, 11, m)
        age = rg.integers(0, 51, m)

        X = np.column_stack((area, bedrooms, floors, age))

        # now generating Y
        noise = np.random.normal(0, 0.4, m)

        true_w = np.array([10000, 100000, 1000000, -100000])
        true_b = 100000

        Y = (X @ true_w) + true_b + noise

        return (X, Y)

#end of class DataGenerator