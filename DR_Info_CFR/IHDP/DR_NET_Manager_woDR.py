from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from DRNet_Model import DRNetPhi, DRNetH_Y1, DRNetH_Y0


class DRNet_Manager_wo_DR_Net:
    def __init__(self, input_nodes, shared_nodes, outcome_nodes, device):
        self.dr_net_phi = DRNetPhi(input_nodes=input_nodes,
                                   shared_nodes=shared_nodes).to(device)

        self.dr_net_h_y1 = DRNetH_Y1(input_nodes=shared_nodes,
                                     outcome_nodes=outcome_nodes).to(device)

        self.dr_net_h_y0 = DRNetH_Y0(input_nodes=shared_nodes,
                                     outcome_nodes=outcome_nodes).to(device)

    def train_DR_NET(self, train_parameters, device):
        epochs = train_parameters["epochs"]
        batch_size = train_parameters["batch_size"]
        lr = train_parameters["lr"]
        weight_decay = train_parameters["lambda"]
        shuffle = train_parameters["shuffle"]
        train_dataset = train_parameters["train_dataset"]
        # val_dataset = train_parameters["val_dataset"]
        ALPHA = train_parameters["ALPHA"]
        BETA = train_parameters["BETA"]

        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=shuffle)

        # val_data_loader = torch.utils.data.DataLoader(val_dataset,
        #                                               shuffle=False)

        optimizer_W = optim.Adam(self.dr_net_phi.parameters(), lr=lr)
        optimizer_V1 = optim.Adam(self.dr_net_h_y1.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_V0 = optim.Adam(self.dr_net_h_y0.parameters(), lr=lr, weight_decay=weight_decay)

        loss_F_MSE = nn.MSELoss()
        loss_CF_MSE = nn.MSELoss()

        for epoch in range(epochs):
            epoch += 1
            total_loss_train = 0
            with tqdm(total=len(train_data_loader)) as t:
                self.dr_net_phi.train()
                self.dr_net_h_y0.train()
                self.dr_net_h_y1.train()
                for batch in train_data_loader:
                    covariates_X, T, y_f, y_cf, _, _ = batch
                    covariates_X = covariates_X.to(device)
                    T = T.to(device)

                    optimizer_W.zero_grad()
                    optimizer_V1.zero_grad()
                    optimizer_V0.zero_grad()

                    y1_hat = self.dr_net_h_y1(self.dr_net_phi(covariates_X))
                    y0_hat = self.dr_net_h_y0(self.dr_net_phi(covariates_X))

                    T_float = T.float()

                    y_f_hat = y1_hat * T_float + y0_hat * (1 - T_float)
                    y_cf_hat = y1_hat * (1 - T_float) + y0_hat * T_float

                    if torch.cuda.is_available():
                        loss_F = loss_F_MSE(y_f_hat.float().cuda(),
                                            y_f.float().cuda()).to(device)
                        loss_CF = loss_CF_MSE(y_cf_hat.float().cuda(),
                                              y_cf.float().cuda()).to(device)

                    else:
                        loss_F = loss_F_MSE(y_f_hat.float(),
                                            y_f.float()).to(device)
                        loss_CF = loss_CF_MSE(y_cf_hat.float(),
                                              y_cf.float()).to(device)

                    loss = loss_F + loss_CF
                    loss.backward()
                    total_loss_train += loss_F.item() + loss_CF.item()

                    optimizer_W.step()
                    optimizer_V1.step()
                    optimizer_V0.step()

                    t.set_postfix(epoch='{0}'.format(epoch), loss='{:05.3f}'.format(total_loss_train))
                    t.update()

    def test_DR_NET(self, test_parameters, device):
        eval_set = test_parameters["tensor_dataset"]
        self.dr_net_phi.eval()
        self.dr_net_h_y0.eval()
        self.dr_net_h_y1.eval()

        _data_loader = torch.utils.data.DataLoader(eval_set,
                                                   shuffle=False)

        y1_hat_list = []
        y0_hat_list = []
        y1_true_list = []
        y0_true_list = []

        ITE_dict_list = []

        for batch in _data_loader:
            covariates_X, T, yf, ycf, mu0, mu1 = batch
            covariates_X = covariates_X.to(device)
            y1_hat = self.dr_net_h_y1(self.dr_net_phi(covariates_X))
            y0_hat = self.dr_net_h_y0(self.dr_net_phi(covariates_X))

            T_float = T.float()

            y_f_hat = y1_hat * T_float + y0_hat * (1 - T_float)

            y1_hat_list.append(y1_hat.item())
            y0_hat_list.append(y0_hat.item())

            y1_true = T * yf + (1 - T) * ycf
            y0_true = (1 - T) * yf + T * ycf

            y1_true_list.append(mu1.item())
            y0_true_list.append(mu0.item())

            predicted_ITE = y1_hat - y0_hat
            true_ITE = mu1 - mu0
            if torch.cuda.is_available():
                diff_ite = abs(true_ITE.float().cuda() - predicted_ITE.float().cuda())
                diff_yf = abs(yf.float().cuda() - y_f_hat.float().cuda())
            else:
                diff_ite = abs(true_ITE.float() - predicted_ITE.float())
                diff_yf = abs(yf.float() - y_f_hat.float())

            ITE_dict_list.append(self.create_ITE_Dict(T.item(),
                                                      true_ITE.item(),
                                                      predicted_ITE.item(),
                                                      diff_ite.item(),
                                                      yf.item(),
                                                      y_f_hat.item(),
                                                      diff_yf.item()))

        return {
            "y1_hat_list": np.array(y1_hat_list),
            "y0_hat_list": np.array(y0_hat_list),
            "y1_true_list": np.array(y1_true_list),
            "y0_true_list": np.array(y0_true_list),
            "ITE_dict_list": ITE_dict_list
        }

    @staticmethod
    def create_ITE_Dict(T, true_ite, predicted_ite, diff_ite, yf_true, yf_hat, diff_yf):
        result_dict = OrderedDict()

        result_dict["Treatment"] = T
        result_dict["true_ite"] = true_ite
        result_dict["predicted_ite"] = predicted_ite
        result_dict["diff_ite"] = diff_ite
        result_dict["yf_true"] = yf_true
        result_dict["yf_hat"] = yf_hat
        result_dict["diff_yf"] = diff_yf

        return result_dict
