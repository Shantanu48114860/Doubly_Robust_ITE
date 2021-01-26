from Constants import Constants
from Experiments import Experiments

if __name__ == '__main__':
    # Twins
    print("Using original data")
    running_mode = "original_data"
    original_exp = Experiments(running_mode)
    original_exp.run_all_experiments(iterations=1)
