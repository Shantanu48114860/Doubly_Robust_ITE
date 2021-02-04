import os

import numpy as np
import pandas as pd
import pyro.distributions as dist
import torch

from Utils import Utils


class DataLoader:
    @staticmethod
    def load_train_test_ihdp_shalit(train_path, test_path, iter_id):
        iter_id = 136
        train_arr = np.load(train_path)
        test_arr = np.load(test_path)
        np_train_X = train_arr['x'][:, :, iter_id]
        np_train_T = Utils.convert_to_col_vector(train_arr['t'][:, iter_id])
        np_train_yf = Utils.convert_to_col_vector(train_arr['yf'][:, iter_id])
        np_train_ycf = Utils.convert_to_col_vector(train_arr['ycf'][:, iter_id])
        np_train_mu0 = Utils.convert_to_col_vector(train_arr['mu0'][:, iter_id])
        np_train_mu1 = Utils.convert_to_col_vector(train_arr['mu1'][:, iter_id])

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
        np_test_mu0 = Utils.convert_to_col_vector(test_arr['mu0'][:, iter_id])
        np_test_mu1 = Utils.convert_to_col_vector(test_arr['mu1'][:, iter_id])

        print("Numpy Train Statistics:")
        print(np_train_X.shape)
        print(np_train_T.shape)

        # print("Numpy Val Statistics:")
        # print(np_val_X.shape)
        # print(np_val_T.shape)
        # print(np_val_yf.shape)
        # print(np_val_ycf.shape)

        print("Numpy Test Statistics:")
        print(np_test_X.shape)
        print(np_test_T.shape)

        # tensor_train = Utils.convert_to_tensor(np_train_X, np_train_T, np_train_yf, np_train_ycf)
        # tensor_test = Utils.convert_to_tensor(np_test_X, np_test_T, np_test_yf, np_test_ycf)
        # return np_train_X, np_train_T, np_train_yf, np_train_ycf, \
        #        np_val_X, np_val_T, np_val_yf, np_val_ycf, \
        #        np_test_X, np_test_T, np_test_yf, np_test_ycf, n_treated, n_total

        return np_train_X, np_train_T, np_train_yf, np_train_ycf, np_train_mu0, np_train_mu1, \
               np_test_X, np_test_T, np_test_yf, np_test_ycf, np_test_mu0, np_test_mu1, n_treated, n_total

    def load_train_test_ihdp_random(self, csv_path, split_size):
        # print(".. Data Loading ..")
        # data load
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), csv_path), header=None)
        np_covariates_X, np_treatment_T, np_outcomes_Y_f, np_outcomes_Y_cf, \
        np_mu0, np_mu1 = self.__convert_to_numpy(df)
        print("ps_np_covariates_X: {0}".format(np_covariates_X.shape))
        print("ps_np_treatment_Y: {0}".format(np_treatment_T.shape))

        np_train_X, np_test_X, np_train_T, np_test_T, np_train_yf, np_test_yf, np_train_ycf, np_test_ycf, \
        np_train_mu0, np_test_mu0, np_train_mu1, np_test_mu1 = \
            Utils.test_train_split(np_covariates_X, np_treatment_T, np_outcomes_Y_f,
                                   np_outcomes_Y_cf, np_mu0, np_mu1, split_size)

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
               np_test_X, np_test_T, np_test_yf, np_test_ycf, \
               np_train_mu0, np_test_mu0, np_train_mu1, np_test_mu1, n_treated, n_total

    @staticmethod
    def __convert_to_numpy(df):
        covariates_X = df.iloc[:, 5:]
        treatment_T = df.iloc[:, 0]
        outcomes_Y_f = df.iloc[:, 1]
        outcomes_Y_cf = df.iloc[:, 2]
        mu0 = df.iloc[:, 3]
        mu1 = df.iloc[:, 4]

        np_covariates_X = Utils.convert_df_to_np_arr(covariates_X)
        np_outcomes_Y_f = Utils.convert_to_col_vector(Utils.convert_df_to_np_arr(outcomes_Y_f))
        np_outcomes_Y_cf = Utils.convert_to_col_vector(Utils.convert_df_to_np_arr(outcomes_Y_cf))
        np_treatment_T = Utils.convert_to_col_vector(Utils.convert_df_to_np_arr(treatment_T))
        np_mu0 = Utils.convert_to_col_vector(Utils.convert_df_to_np_arr(mu0))
        np_mu1 = Utils.convert_to_col_vector(Utils.convert_df_to_np_arr(mu1))

        return np_covariates_X, np_treatment_T, np_outcomes_Y_f, np_outcomes_Y_cf, np_mu0, np_mu1

    @staticmethod
    def custom_data_loader():
        N = 5000
        # sigma = dist.Uniform(-1, 1).sample([10, 10])
        sigma = np.random.uniform(-1, 1, (10, 10))
        # print(sigma+ (sigma.T))
        cov = 0.5 * (sigma.dot(sigma.T))
        # vary this
        mean_T = np.empty(10)
        mean_T.fill(7.5)
        x_treated = np.random.multivariate_normal(mean_T, cov, 5000)

        mean_C = np.zeros(10)
        x_control = np.random.multivariate_normal(mean_C, cov, 5000)

        X = np.concatenate((x_treated, x_control), axis=0)

        t_t = np.ones(N)
        t_c = np.zeros(5000)
        t = Utils.convert_to_col_vector(np.concatenate((t_t, t_c), axis=0))
        print(t.shape)

        bias_y = np.random.multivariate_normal(np.zeros(2), 0.1 * np.eye(2, 2), 2 * N)
        w_y = np.random.uniform(-0.1, 0.1, (10, 2))

        y = X.dot(w_y) + bias_y
        print(y.shape)

        np_X = np.concatenate((X, t, y), axis=1)
        # np.random.shuffle(X)
        np.save("Dataset/Custom_GANITE_8.npy", np_X)

        print("KL: ", Utils.kl_divergence(mean_T, mean_C, cov, cov))
        print("X statistics: ")
        print(np_X.shape)

    # def generate_data(N, alpha=0.25, beta=1, gamma=1):
    #     """
    #     This implements the generative process of [1], but using larger feature and
    #     latent spaces ([1] assumes ``feature_dim=1`` and ``latent_dim=5``).
    #     """
    #
    #     zc = dist.Bernoulli(0.5).sample([N])
    #     zt = dist.Bernoulli(0.5).sample([N])
    #     zy = dist.Bernoulli(0.5).sample([args.num_data])
    #
    #     # zc = dist.Normal(0,1).sample([args.num_data])
    #     # zt = dist.Normal(0,1).sample([args.num_data])
    #     # zy = dist.Normal(0,1).sample([args.num_data])
    #
    #     xc = dist.Normal(zc, 5 * zc + 3 * (1 - zc)).sample([args.synthetic_dim]).t()
    #     xt = dist.Normal(zt, 2 * zt + 0.5 * (1 - zt)).sample([args.synthetic_dim]).t()
    #     xy = dist.Normal(zy, 10 * zy + 6 * (1 - zy)).sample([args.synthetic_dim]).t()
    #
    #     x = torch.cat([xc, xt, xy], -1)
    #
    #     t = torch.mul(dist.Bernoulli(alpha * zt + (1 - alpha) * (1 - zt)).sample(),
    #                   dist.Bernoulli(alpha * zt + (1 - alpha) * (1 - zt)).sample())
    #
    #     y = dist.Normal(beta * (zc + gamma * (2 * t - 2)), 1).sample([1]).t().squeeze(-1) + dist.Normal(beta * zy,
    #                                                                                                     1).sample(
    #         [1]).t().squeeze(-1)
    #
    #     # Compute true ite for evaluation (via Monte Carlo approximation).
    #     t0_t1 = torch.tensor([[0.], [1.]])
    #
    #     y_t0, y_t1 = dist.Normal(beta * (zc + gamma * (2 * t0_t1 - 2)), 1).mean + dist.Normal(beta * zy, 1).mean
    #
    #     true_ite = y_t1 - y_t0

    def generate_data(N, alpha=0.25, beta=1, gamma=1):
        """
        This implements the generative process of [1], but using larger feature and
        latent spaces ([1] assumes ``feature_dim=1`` and ``latent_dim=5``).
        """

        zx = dist.Bernoulli(0.5).sample([N])
        zt = dist.Bernoulli(0.5).sample([N])
        zyf = dist.Bernoulli(0.5).sample([N])
        zycf = dist.Bernoulli(0.5).sample([N])

        x_x = dist.Normal(zx, 5 * zx + 3 * (1 - zx)).sample([7]).t()
        x_t = dist.Normal(zt, 2 * zt + 0.5 * (1 - zt)).sample([1]).t()
        x_yf = dist.Normal(zyf, 10 * zyf + 6 * (1 - zyf)).sample([1]).t()
        x_ycf = dist.Normal(zycf, 10 * zycf + 6 * (1 - zycf)).sample([1]).t()

        x = torch.cat([x_x, x_t, x_yf, x_ycf], -1)
        print("X: ", x.size())

        w = dist.Uniform(-0.1, 0.1).sample([10, 1])
        n_t = dist.Normal(0, 0.1).sample([1])

        print("W: ", w.size())
        print("n_t: ", n_t.size())
        print((torch.mm(x, w) + n_t).size())
        t = dist.Bernoulli(torch.sigmoid(torch.mm(x, w) + n_t)).sample().squeeze(-1)
        print("t: ", t.size())
        # print(t)

        # n_Y = dist.Normal(0, 0.1).sample([2])
        # print(n_Y.size())
        # w_y = dist.Uniform(-1, 1).sample([10, 2])
        # print((torch.mm(x, w_y).add(n_Y)).size())
        # y = (torch.mm(x, w_y) + n_Y).squeeze(-1)

        n_Y = dist.Normal(0, 0.1).sample([1])
        print(n_Y.size())
        w_y = dist.Uniform(-1, 1).sample([10, 1])
        print((torch.mm(x, w_y).add(n_Y)).size())
        y_t0 = (torch.mm(x, w_y) + n_Y).squeeze(-1)

        n_Y = dist.Normal(0, 0.1).sample([1])
        print(n_Y.size())
        w_y = dist.Uniform(-1, 1).sample([10, 1])
        print((torch.mm(x, w_y).add(n_Y)).size())
        y_t1 = (torch.mm(x, w_y) + n_Y).squeeze(-1)
        # print(y.size())

        # y_t0 = y[:, 0]
        # y_t1 = y[:, 1]

        y_f = t * y_t1 + (1 - t) * y_t0
        y_cf = (1 - t) * y_t1 + t * y_t0

        np_x = x.numpy()
        np_t = Utils.convert_to_col_vector(t.numpy())
        np_y_f = Utils.convert_to_col_vector(y_f.numpy())
        np_y_cf = Utils.convert_to_col_vector(y_cf.numpy())
        np_mu0 = Utils.convert_to_col_vector(y_t0.numpy())
        np_mu1 = Utils.convert_to_col_vector(y_t1.numpy())

        print(np_x.shape)
        print(np_t.shape)
        # print(t[t==1].shape)
        # print(t[t==0].shape)
        print(np_mu0.shape)
        print(np_mu1.shape)

        np_X = np.concatenate((np_x, np_t, np_y_f, np_y_cf, np_mu0, np_mu1), axis=1)
        print(np_X.shape)
        # np.random.shuffle(X)
        np.save("Dataset/Custom_GANITE_8.npy", np_X)

    def generate_data_cevae(N, alpha=0.75, beta=3, gamma=2):
        """
        This implements the generative process of [1], but using larger feature and
        latent spaces ([1] assumes ``feature_dim=1`` and ``latent_dim=5``).
        """

        z = dist.Bernoulli(0.5).sample([N])

        x = dist.Normal(z, (5 ** 2) * z + (3 ** 2) * (1 - z)).sample([5]).t()

        print(x.size())

        t = dist.Bernoulli(alpha * z + (1 - alpha) * (1 - z)).sample()
        print("T: ", t.size())
        y_f = dist.Bernoulli(logits=beta * (z + gamma * (2 * t - 1))).sample()
        y_cf = dist.Bernoulli(logits=beta * (z + gamma * (2 * t - 1))).sample()
        # print("Y", y.size())

        # Compute true ite for evaluation (via Monte Carlo approximation).
        t0_t1 = torch.tensor([[0.], [1.]])
        y_t0, y_t1 = dist.Bernoulli(logits=beta * (z + gamma * (2 * t0_t1 - 1))).mean

        print("Y, ", y_f.size())
        print("y_t0, ", y_t0.size())

        true_ite = y_t1 - y_t0
        print(true_ite)
        np_x = x.numpy()
        np_t = Utils.convert_to_col_vector(t.numpy())
        np_mu0 = Utils.convert_to_col_vector(y_t0.numpy())
        np_mu1 = Utils.convert_to_col_vector(y_t1.numpy())
        np_y_f = Utils.convert_to_col_vector(y_f.numpy())
        np_y_cf = Utils.convert_to_col_vector(y_cf.numpy())

        print(np_x.shape)
        print(np_t.shape)
        # print(t[t==1].shape)
        # print(t[t==0].shape)
        print(np_mu0.shape)
        print(np_mu1.shape)

        np_X = np.concatenate((np_x, np_t, np_y_f, np_y_cf, np_mu0, np_mu1), axis=1)
        print(np_X.shape)
        # np.random.shuffle(X)
        np.save("Dataset/Custom_GANITE_8.npy", np_X)

    @staticmethod
    def load_custom(iter_id):
        X = np.load("Dataset/Custom_GANITE_8.npy")
        covariates_x = X[:, 0:10]
        print(covariates_x.shape)
        t = Utils.convert_to_col_vector(X[:, -5])
        y_f = Utils.convert_to_col_vector(X[:, -4])
        y_cf = Utils.convert_to_col_vector(X[:, -3])
        np_mu0 = Utils.convert_to_col_vector(X[:, -2])
        np_mu1 = Utils.convert_to_col_vector(X[:, -1])

        np_train_X, np_test_X, np_train_T, np_test_T, np_train_yf, np_test_yf, np_train_ycf, \
        np_test_ycf, np_train_mu0, np_test_mu0, np_train_mu1, np_test_mu1 = \
            Utils.test_train_split_custom(covariates_x, t, y_f,
                                          y_cf, iter_id,
                                          np_mu0, np_mu1, split_size=0.8)

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

        # print("KL: ", Utils.kl_divergence(mean_T, mean_C, cov, cov))
        return np_train_X, np_train_T, np_train_yf, np_train_ycf, \
               np_test_X, np_test_T, np_test_yf, np_test_ycf, \
               np_train_mu0, np_test_mu0, np_train_mu1, np_test_mu1, \
               n_treated, n_total


# DataLoader.generate_data_cevae(10000)


# n_Y = dist.Normal(0, 0.1).sample([2])
# print(n_Y)
# w_y = dist.Uniform(-1, 1).sample([3, 2])
# x = torch.rand((3, 3))
# z = torch.mm(x, w_y)
# print(z)
# print(z+n_Y)
# y = (torch.mm(x, w_y) + n_Y).squeeze(-1)
# print("----")
DataLoader.generate_data(1000)

# DataLoader.custom_data_loader()
