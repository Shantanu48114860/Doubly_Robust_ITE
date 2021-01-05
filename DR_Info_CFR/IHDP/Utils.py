from collections import namedtuple
from itertools import product

import numpy as np
import pandas as pd
import sklearn.model_selection as sklearn
import torch


class Utils:
    @staticmethod
    def convert_df_to_np_arr(data):
        return data.to_numpy()

    @staticmethod
    def convert_to_col_vector(np_arr):
        return np_arr.reshape(np_arr.shape[0], 1)

    @staticmethod
    def test_train_split(np_train_X, np_train_T, np_train_yf, np_train_ycf, split_size=0.8):
        return sklearn.train_test_split(np_train_X, np_train_T, np_train_yf, np_train_ycf,
                                        train_size=split_size)

    @staticmethod
    def convert_to_tensor_DR_net(X, T, Y_f, Y_cf, pi, mu):
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        tensor_T = torch.from_numpy(T)
        tensor_y_f = torch.from_numpy(Y_f)
        tensor_y_cf = torch.from_numpy(Y_cf)
        tensor_pi = torch.from_numpy(pi)
        tensor_mu = torch.from_numpy(mu)

        processed_dataset = torch.utils.data.TensorDataset(tensor_x,
                                                           tensor_T,
                                                           tensor_y_f,
                                                           tensor_y_cf,
                                                           tensor_pi,
                                                           tensor_mu)
        return processed_dataset

    @staticmethod
    def convert_to_tensor(X, T, Y_f, Y_cf):
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        tensor_T = torch.from_numpy(T)
        tensor_y_f = torch.from_numpy(Y_f)
        tensor_y_cf = torch.from_numpy(Y_cf)

        processed_dataset = torch.utils.data.TensorDataset(tensor_x,
                                                           tensor_T,
                                                           tensor_y_f,
                                                           tensor_y_cf)
        return processed_dataset

    @staticmethod
    def convert_to_tensor_pi_net(X, T):
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        tensor_T = torch.from_numpy(T)

        processed_dataset = torch.utils.data.TensorDataset(tensor_x,
                                                           tensor_T)
        return processed_dataset

    @staticmethod
    def convert_to_tensor_mu_net(X, T, yf):
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        tensor_T = torch.from_numpy(T)
        tensor_y_f = torch.from_numpy(yf)

        processed_dataset = torch.utils.data.TensorDataset(tensor_x,
                                                           tensor_T,
                                                           tensor_y_f)
        return processed_dataset

    @staticmethod
    def concat_np_arr(X, Y, axis=1):
        return np.concatenate((X, Y), axis)

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    @staticmethod
    def get_runs(params):
        """
        Gets the run parameters using cartesian products of the different parameters.
        :param params: different parameters like batch size, learning rates
        :return: iterable run set
        """
        Run = namedtuple("Run", params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

    @staticmethod
    def write_to_csv(file_name, list_to_write):
        pd.DataFrame.from_dict(
            list_to_write,
            orient='columns'
        ).to_csv(file_name)

    @staticmethod
    def vae_loss(mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """

    def __call__(self, noise_z, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (noise_z - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


class EarlyStopping_DCN:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0,
                 dr_net_shared_path="DR_WO_Info_CFR/IHDP/Models/dr_net_phi.pth",
                 dr_net_y1_path="DR_WO_Info_CFR/Models/IHDP/Models/dr_net_y1.pth",
                 dr_net_y0_path="DR_WO_Info_CFR/Models/IHDP/Models/dr_net_y0.pth",
                 pi_net_path="DR_WO_Info_CFR/Models/IHDP/Models/pi_net.pth",
                 mu_net_path="DR_WO_Info_CFR/Models/IHDP/Models/mu_net.pth",
                 trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dr_net_shared_path = dr_net_shared_path
        self.dr_net_y1_path = dr_net_y1_path
        self.dr_net_y0_path = dr_net_y0_path
        self.pi_net_path = pi_net_path
        self.mu_net_path = mu_net_path
        self.trace_func = trace_func

    def __call__(self, val_loss, dr_net_shared, dr_net_y1_model, dr_net_y0_model,
                 pi_net_model, mu_net_model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, dr_net_shared, dr_net_y1_model, dr_net_y0_model,
                                 pi_net_model, mu_net_model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, dr_net_shared, dr_net_y1_model, dr_net_y0_model,
                                 pi_net_model, mu_net_model)
            self.counter = 0

    def save_checkpoint(self, val_loss, dr_net_shared, dr_net_y1_model, dr_net_y0_model,
                        pi_net_model, mu_net_model):
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ...')
        torch.save(dr_net_shared.state_dict(), self.dr_net_shared_path)
        torch.save(dr_net_y1_model.state_dict(), self.dr_net_y1_path)
        torch.save(dr_net_y0_model.state_dict(), self.dr_net_y0_path)
        torch.save(pi_net_model.state_dict(), self.pi_net_path)
        torch.save(mu_net_model.state_dict(), self.mu_net_path)
        self.val_loss_min = val_loss
