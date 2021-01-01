import numpy as np

from Utils import Utils


class DataLoader:
    @staticmethod
    def load_train_test_jobs(train_path, test_path, iter_id):
        train_arr = np.load(train_path)
        test_arr = np.load(test_path)
        np_train_X = train_arr['x'][:, :, iter_id]
        np_train_T = Utils.convert_to_col_vector(train_arr['t'][:, iter_id])
        np_train_e = Utils.convert_to_col_vector(train_arr['e'][:, iter_id])
        np_train_yf = Utils.convert_to_col_vector(train_arr['yf'][:, iter_id])

        # train_X = np.concatenate((np_train_X, np_train_e, np_train_yf), axis=1)

        np_test_X = test_arr['x'][:, :, iter_id]
        np_test_T = Utils.convert_to_col_vector(test_arr['t'][:, iter_id])
        np_test_e = Utils.convert_to_col_vector(test_arr['e'][:, iter_id])
        np_test_yf = Utils.convert_to_col_vector(test_arr['yf'][:, iter_id])

        # test_X = np.concatenate((np_test_X, np_test_e, np_test_yf), axis=1)

        print("Numpy Train Statistics:")
        print(np_train_X.shape)
        print(np_train_T.shape)

        # print("Numpy Val Statistics:")
        # print(val_X.shape)
        # print(val_T.shape)

        print(" Numpy Test Statistics:")
        print(np_test_X.shape)
        print(np_test_T.shape)

        # X -> x1.. x17, e, yf -> (19, 1)
        return np_train_X, np_train_T, np_train_e, np_train_yf, \
               np_test_X, np_test_T, np_test_e, np_test_yf
