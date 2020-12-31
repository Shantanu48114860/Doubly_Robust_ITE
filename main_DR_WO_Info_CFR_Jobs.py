from Jobs.Constants import Constants
from Jobs.Experiments import Experiments

if __name__ == '__main__':
    train_path = "DR_WO_Info_CFR/Jobs/Dataset/jobs_DW_bin.new.10.train.npz"
    test_path = "DR_WO_Info_CFR/Jobs/Dataset/jobs_DW_bin.new.10.test.npz"
    print("Using original data")
    running_mode = "original_data"
    original_exp = Experiments(running_mode)
    original_exp.run_all_experiments(train_path, test_path,
                                     iterations=10)
