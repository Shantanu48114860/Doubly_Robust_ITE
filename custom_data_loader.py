import numpy as np
import pyro.distributions as dist
import torch

from Utils import Utils


def custom_data_loader(N):
    zx = dist.Bernoulli(0.5).sample([N])
    zt = dist.Bernoulli(0.5).sample([N])
    zyf = dist.Bernoulli(0.5).sample([N])
    zycf = dist.Bernoulli(0.5).sample([N])

    z_t_1 = torch.cat((zx, zt, zyf, zycf), -1)
    print(z_t_1.size())
    print(z_t_1[0])

    zx = dist.Bernoulli(0.6).sample([N])
    zt = dist.Bernoulli(0.6).sample([N])
    zyf = dist.Bernoulli(0.6).sample([N])
    zycf = dist.Bernoulli(0.6).sample([N])

    z_t_0 = torch.cat((zx, zt, zyf, zycf), -1)
    print(z_t_0.size())
    print(z_t_0[0])

    sigma_t_1 = 5 * z_t_1 + 3 * (1 - z_t_1)
    sigma_t_0 = 5 * z_t_0 + 3 * (1 - z_t_0)
    x_t_1 = dist.Normal(z_t_1, sigma_t_1).sample([10]).t()
    x_t_0 = dist.Normal(z_t_0, sigma_t_0).sample([10]).t()

    print("KL: ", Utils.kl_divergence(z_t_1.numpy(), z_t_0.numpy(),
                                      sigma_t_1.numpy(), sigma_t_0.numpy()))

    x = torch.cat((x_t_1, x_t_0), dim=0)
    print(x.size())

    t_t = torch.ones(4 * N)
    t_c = torch.zeros(4 * N)
    t = torch.cat((t_t, t_c), dim=0)
    print(t.shape)

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


def generate_data_DR_VIDAL(N, alpha=0.25, beta=1, gamma=1):
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


# DataLoader.generate_data_DR_VIDAL(1000)

# DataLoader.custom_data_loader(1000)
