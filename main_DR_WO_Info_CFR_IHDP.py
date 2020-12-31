from IHDP.Constants import Constants
from IHDP.Experiments import Experiments

if __name__ == '__main__':
    # IHDP
    train_path = "DR_WO_Info_CFR/IHDP/Dataset/ihdp_npci_1-1000.train.npz"
    test_path = "DR_WO_Info_CFR/IHDP/Dataset/ihdp_npci_1-1000.test.npz"
    print("Using original data")
    running_mode = "original_data"
    original_exp = Experiments(running_mode)
    original_exp.run_all_experiments(train_path, test_path,
                                     iterations=10)
