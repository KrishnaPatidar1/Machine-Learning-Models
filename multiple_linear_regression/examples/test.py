# Importing the mlr_model file from the parent directory
import sys
from pathlib import Path

# 1. Getting the absolute path of the parent directory
parent_dir = str(Path(__file__).resolve().parent.parent)

# 2. Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# 3. Importing the class from the module
from mlr_model import MultipleLinearRegressionModel 

# Importing others modules from the current directory
from random_data_generator import DataGenerator
from visuals import DataVisualizer
from verifier import Verifier
from time import perf_counter

# testing the code now
# main
if __name__ == "__main__":

    X, Y = DataGenerator.generate_random_housePrice_data(100000)
    model1 = MultipleLinearRegressionModel(X, Y)

    # starting timer
    initial_time = perf_counter()

    # model1.maxScale_X()
    # model1.minMaxScaling()
    # model1.meanNormalization()
    model1.ZScoreNormalization()

    # Measuring time and verifying correct answer using sklearn
    
    # ending measurement of time
    end_time = perf_counter()
    print(f"\nTime required for scaling : {end_time - initial_time : 0.6f} seconds")

    # starting to measure time for gradient descent
    initial_time = perf_counter()
    model1.gradient_descent()

    # ending measurement of time for gradient descent
    end_time = perf_counter()

    print(f'\nModel generated w : {model1.w}')
    print(f'Model generated b : {model1.b}')
    print(f'final cost : {model1.calculate_cost(model1.w, model1.b)}')

    actual_w, actual_b = Verifier.calculate_solution(model1.X, model1.Y)

    print(f'\nsklearn generated w : {actual_w}')
    print(f'sklearn generated b : {actual_b}')
    print(f'cost : {model1.calculate_cost(actual_w, actual_b)}')

    print(f"\nTime required for gradient descent : {end_time - initial_time : 0.6f} seconds")
    

    # Visualizing gradient descent for weight of feature 0 and bias
    DataVisualizer.run_and_visualize_gradient_descent(model1)


# end of main