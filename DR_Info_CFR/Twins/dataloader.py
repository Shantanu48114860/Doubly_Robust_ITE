import numpy as np
from scipy.special import expit

from Utils import Utils


class DataLoader:
    def load_train_test_twins_random(self, csv_path, split_size=0.8):
        # print(".. Data Loading ..")
        # data load
        np_covariates_X, np_treatment_T, np_outcomes_Y_f, np_outcomes_Y_cf, no \
            = self.preprocess_dataset_for_training(csv_path)
        print("ps_np_covariates_X: {0}".format(np_covariates_X.shape))
        print("ps_np_treatment_Y: {0}".format(np_treatment_T.shape))

        idx = np.random.permutation(no)
        train_idx = idx[:int(split_size * no)]
        test_idx = idx[int(split_size * no):]

        # np_train_X = np_covariates_X[train_idx, :]
        # np_train_T = np_treatment_T[train_idx, :]
        # np_train_yf = np_outcomes_Y_f[train_idx, :]
        # np_train_ycf = np_outcomes_Y_cf[train_idx, :]
        #
        # np_test_X = np_covariates_X[test_idx, :]
        # np_test_T = np_treatment_T[test_idx, :]
        # np_test_yf = np_outcomes_Y_f[test_idx, :]
        # np_test_ycf = np_outcomes_Y_cf[test_idx, :]

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

        print("Numpy test Statistics:")
        print(np_test_X.shape)
        print(np_test_T.shape)

        return np_train_X, np_train_T, np_train_yf, np_train_ycf, \
               np_test_X, np_test_T, np_test_yf, np_test_ycf, n_treated, n_total

    @staticmethod
    def preprocess_dataset_for_training(csv_path):
        data_X = np.loadtxt(csv_path, delimiter=",", skiprows=1)

        # Define features
        x = data_X[:, :30]
        no, dim = x.shape

        # Define potential outcomes
        potential_y = data_X[:, 30:]
        # Die within 1 year = 1, otherwise = 0
        potential_y = np.array(potential_y < 9999, dtype=float)

        # Assign treatment
        coef = np.random.uniform(-0.01, 0.01, size=[dim, 1])
        prob_temp = expit(np.matmul(x, coef) + np.random.normal(0, 0.01, size=[no, 1]))
        prob_t = prob_temp / (2 * np.mean(prob_temp))
        prob_t[prob_t > 1] = 1

        t = np.random.binomial(1, prob_t, [no, 1])
        t = t.reshape([no, ])
        t = Utils.convert_to_col_vector(t)

        y_f = np.transpose(t) * potential_y[:, 1] + np.transpose(1 - t) * potential_y[:, 0]
        y_f = np.reshape(np.transpose(y_f), [no, ])
        y_f = Utils.convert_to_col_vector(y_f)

        y_cf = np.transpose(1 - t) * potential_y[:, 1] + np.transpose(t) * potential_y[:, 0]
        y_cf = np.reshape(np.transpose(y_cf), [no, ])
        y_cf = Utils.convert_to_col_vector(y_cf)

        print(x.shape)
        print(potential_y.shape)
        print(y_f.shape)
        print(y_cf.shape)
        print(t.shape)

        # print(t[1])
        # print(potential_y[1])
        # print(y_f[1])
        # print(y_cf[1])

        return x, t, y_f, y_cf, no
