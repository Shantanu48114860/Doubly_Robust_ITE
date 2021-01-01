import os

import numpy as np
import pandas as pd

from Utils import Utils


class DataLoader:
    @staticmethod
    def load_train_test_ihdp_shalit(train_path, test_path, iter_id):
        train_arr = np.load(train_path)
        test_arr = np.load(test_path)
        np_train_X = train_arr['x'][:, :, iter_id]
        np_train_T = Utils.convert_to_col_vector(train_arr['t'][:, iter_id])
        np_train_yf = Utils.convert_to_col_vector(train_arr['yf'][:, iter_id])
        np_train_ycf = Utils.convert_to_col_vector(train_arr['ycf'][:, iter_id])

        n_treated = np_train_T[np_train_T == 1]
        # print(n_treated.shape[0])
        # print(np_train_T.shape[0])

        n_treated = n_treated.shape[0]
        n_total = np_train_T.shape[0]

        # np_train_X, np_val_X, np_train_T, np_val_T, np_train_yf, np_val_yf, np_train_ycf, np_val_ycf = \
        #     Utils.test_train_split(np_train_X, np_train_T, np_train_yf, np_train_ycf, split_size=0.8)

        np_test_X = test_arr['x'][:, :, iter_id]
        np_test_T = Utils.convert_to_col_vector(test_arr['t'][:, iter_id])
        np_test_yf = Utils.convert_to_col_vector(test_arr['yf'][:, iter_id])
        np_test_ycf = Utils.convert_to_col_vector(test_arr['ycf'][:, iter_id])

        print("Numpy Train Statistics:")
        print(np_train_X.shape)
        print(np_train_T.shape)

        # print("Numpy Val Statistics:")
        # print(np_val_X.shape)
        # print(np_val_T.shape)
        # print(np_val_yf.shape)
        # print(np_val_ycf.shape)

        print("Numpy Temp Statistics:")
        print(np_test_X.shape)
        print(np_test_T.shape)

        # tensor_train = Utils.convert_to_tensor(np_train_X, np_train_T, np_train_yf, np_train_ycf)
        # tensor_test = Utils.convert_to_tensor(np_test_X, np_test_T, np_test_yf, np_test_ycf)
        # return np_train_X, np_train_T, np_train_yf, np_train_ycf, \
        #        np_val_X, np_val_T, np_val_yf, np_val_ycf, \
        #        np_test_X, np_test_T, np_test_yf, np_test_ycf, n_treated, n_total

        return np_train_X, np_train_T, np_train_yf, np_train_ycf, \
               np_test_X, np_test_T, np_test_yf, np_test_ycf, n_treated, n_total

    def load_train_test_ihdp_random(self, csv_path, split_size):
        # print(".. Data Loading ..")
        # data load
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), csv_path), header=None)
        np_covariates_X, np_treatment_T, np_outcomes_Y_f, np_outcomes_Y_cf = self.__convert_to_numpy(df)
        print("ps_np_covariates_X: {0}".format(np_covariates_X.shape))
        print("ps_np_treatment_Y: {0}".format(np_treatment_T.shape))

        np_train_X, np_test_X, np_train_T, np_test_T, np_train_yf, np_test_yf, np_train_ycf, np_test_ycf = \
            Utils.test_train_split(np_covariates_X, np_treatment_T, np_outcomes_Y_f,
                                   np_outcomes_Y_cf, split_size)

        print("Numpy Train Statistics:")
        print(np_train_X.shape)
        print(np_train_T.shape)
        n_treated = np_train_T[np_train_T == 1]
        # print(n_treated.shape[0])
        # print(np_train_T.shape[0])

        n_treated = n_treated.shape[0]
        n_total = np_train_T.shape[0]
        # print("Numpy Val Statistics:")
        # print(np_val_X.shape)
        # print(np_val_T.shape)
        # print(np_val_yf.shape)
        # print(np_val_ycf.shape)

        print("Numpy Temp Statistics:")
        print(np_test_X.shape)
        print(np_test_T.shape)

        return np_train_X, np_train_T, np_train_yf, np_train_ycf, \
               np_test_X, np_test_T, np_test_yf, np_test_ycf, n_treated, n_total

    @staticmethod
    def __convert_to_numpy(df):
        covariates_X = df.iloc[:, 5:]
        treatment_T = df.iloc[:, 0]
        outcomes_Y_f = df.iloc[:, 1]
        outcomes_Y_cf = df.iloc[:, 2]

        np_covariates_X = Utils.convert_df_to_np_arr(covariates_X)
        np_outcomes_Y_f = Utils.convert_to_col_vector(Utils.convert_df_to_np_arr(outcomes_Y_f))
        np_outcomes_Y_cf = Utils.convert_to_col_vector(Utils.convert_df_to_np_arr(outcomes_Y_cf))
        np_treatment_T = Utils.convert_to_col_vector(Utils.convert_df_to_np_arr(treatment_T))

        return np_covariates_X, np_treatment_T, np_outcomes_Y_f, np_outcomes_Y_cf
