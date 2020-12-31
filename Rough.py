import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from IHDP import Utils
from IHDP.DRNet_Model import DRNetPhi, DRNetH_Y1, DRNetH_Y0, pi_net, mu_net


# from Utils import EarlyStopping_DCN


class DRNet_Manager:
    def __init__(self, input_nodes, shared_nodes, outcome_nodes, device):
        self.dr_net_phi = DRNetPhi(input_nodes=input_nodes,
                                   shared_nodes=shared_nodes).to(device)

        self.dr_net_h_y1 = DRNetH_Y1(input_nodes=shared_nodes,
                                     outcome_nodes=outcome_nodes).to(device)

        self.dr_net_h_y0 = DRNetH_Y0(input_nodes=shared_nodes,
                                     outcome_nodes=outcome_nodes).to(device)

        self.pi_net = pi_net(input_nodes=input_nodes,
                             outcome_nodes=outcome_nodes).to(device)

        self.mu_net = mu_net(input_nodes=input_nodes + 1,
                             shared_nodes=shared_nodes,
                             outcome_nodes=outcome_nodes).to(device)

    def train_DR_NET(self, train_parameters, device):
        epochs = train_parameters["epochs"]
        batch_size = train_parameters["batch_size"]
        lr = train_parameters["lr"]
        weight_decay = train_parameters["lambda"]
        shuffle = train_parameters["shuffle"]
        train_dataset = train_parameters["train_dataset"]
        val_dataset = train_parameters["val_dataset"]
        ALPHA = train_parameters["ALPHA"]
        BETA = train_parameters["BETA"]

        # treated_set_val = val_parameters["treated_set"]
        # control_set_val = val_parameters["control_set"]

        # treated_data_loader_val = torch.utils.data.DataLoader(treated_set_val,
        #                                                       shuffle=False)
        # control_data_loader_val = torch.utils.data.DataLoader(control_set_val,
        #                                                       shuffle=False)

        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=shuffle)

        val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                      shuffle=False)

        optimizer_W = optim.Adam(self.dr_net_phi.parameters(), lr=lr)
        optimizer_V1 = optim.Adam(self.dr_net_h_y1.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_V0 = optim.Adam(self.dr_net_h_y0.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_pi = optim.Adam(self.pi_net.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_mu = optim.Adam(self.mu_net.parameters(), lr=lr, weight_decay=weight_decay)

        loss_F_MSE = nn.MSELoss()
        loss_DR_MSE = nn.MSELoss()
        lossBCE = nn.BCELoss()

        early_stopping = Utils.EarlyStopping_DCN(patience=150, verbose=True,
                                                 dr_net_shared_path="DR_WO_Info_CFR/IHDP/Models/dr_net_phi.pth",
                                                 dr_net_y1_path="DR_WO_Info_CFR/IHDP/Models/dr_net_y1.pth",
                                                 dr_net_y0_path="DR_WO_Info_CFR/IHDP/Models/dr_net_y0.pth",
                                                 pi_net_path="DR_WO_Info_CFR/IHDP/Models/pi_net.pth",
                                                 mu_net_path="DR_WO_Info_CFR/IHDP/Models/mu_net.pth")

        valid_losses = []
        for epoch in range(epochs):
            epoch += 1
            total_loss_train = 0
            total_loss_val = 0
            with tqdm(total=len(train_data_loader)) as t:
                self.dr_net_phi.train()
                self.dr_net_h_y0.train()
                self.dr_net_h_y1.train()
                self.pi_net.train()
                self.mu_net.train()
                for batch in train_data_loader:
                    covariates_X, T, y_f, y_cf = batch
                    covariates_X = covariates_X.to(device)
                    T = T.to(device)

                    idx = (T == 1).squeeze()

                    covariates_X_treated = covariates_X[idx]
                    covariates_X_control = covariates_X[~idx]

                    treated_size = covariates_X_treated.size(0)
                    control_size = covariates_X_control.size(0)

                    optimizer_W.zero_grad()
                    optimizer_V1.zero_grad()
                    optimizer_V0.zero_grad()
                    optimizer_pi.zero_grad()
                    optimizer_mu.zero_grad()

                    pi = self.pi_net(covariates_X)
                    mu = self.mu_net(covariates_X, T)

                    y1_hat = self.dr_net_h_y1(self.dr_net_phi(covariates_X))
                    y0_hat = self.dr_net_h_y0(self.dr_net_phi(covariates_X))

                    T_float = T.float()

                    y_f_hat = y1_hat * T_float + y0_hat * (1 - T_float)
                    # y_cf = y1_hat * (1 - T_float) + y0_hat * T_float

                    y_f_dr = T_float * ((T_float * y1_hat - (T_float - pi) * mu) / pi) + \
                             (1 - T_float) * (((1 - T_float) * y0_hat - (T_float - pi) * mu) / (1 - pi))

                    # y_cf_dr = (1 - T_float) * (((1 - T_float) * y1_hat - (T_float - pi) * mu) / pi) + \
                    #          T_float * ((T_float * y0_hat - (T_float - pi) * mu) / (1 - pi))

                    loss_pi = lossBCE(pi, T_float).to(device)
                    if torch.cuda.is_available():
                        loss_F = loss_F_MSE(y_f_hat.float().cuda(),
                                            y_f.float().cuda()).to(device)
                        loss_DR = loss_DR_MSE(y_f_dr.float().cuda(),
                                              y_f.float().cuda()).to(device)
                    else:
                        loss_F = loss_F_MSE(y_f_hat.float(),
                                            y_f.float()).to(device)
                        loss_DR = loss_DR_MSE(y_f_dr.float(),
                                              y_f.float()).to(device)

                    loss = loss_F + ALPHA * loss_pi + BETA * loss_DR

                    # loss_pi.backward(retain_graph=True)
                    # loss_mu.backward(retain_graph=True)
                    # loss_F.backward(retain_graph=True)
                    # loss_DR.backward(retain_graph=True)

                    loss.backward()

                    total_loss_train += loss_F.item() + loss_DR.item() + loss_pi.item()

                    optimizer_pi.step()
                    optimizer_mu.step()
                    optimizer_W.step()

                    if treated_size > 0:
                        optimizer_V1.step()
                    if control_size > 0:
                        optimizer_V0.step()

                    t.set_postfix(epoch='{0}'.format(epoch), loss='{:05.3f}'.format(total_loss_train))
                    t.update()

            ######################
            # validate the model #
            ######################
            # prep model for evaluation
            self.dr_net_phi.eval()
            self.dr_net_h_y0.eval()
            self.dr_net_h_y1.eval()
            self.pi_net.eval()
            self.mu_net.eval()

            # val treated
            for batch in val_data_loader:
                covariates_X, T, y_f, y_cf = batch
                covariates_X = covariates_X.to(device)
                T = T.to(device)

                pi = self.pi_net(covariates_X)
                mu = self.mu_net(covariates_X, T)

                y1_hat = self.dr_net_h_y1(self.dr_net_phi(covariates_X))
                y0_hat = self.dr_net_h_y0(self.dr_net_phi(covariates_X))

                T_float = T.float()

                y_f_hat = y1_hat * T_float + y0_hat * (1 - T_float)

                y_f_dr = T_float * ((T_float * y1_hat - (T_float - pi) * mu) / pi) + \
                         (1 - T_float) * (((1 - T_float) * y0_hat - (T_float - pi) * mu) / (1 - pi))

                loss_pi = lossBCE(pi, T_float).to(device)
                if torch.cuda.is_available():
                    loss_F = loss_F_MSE(y_f_hat.float().cuda(),
                                        y_f.float().cuda()).to(device)
                    loss_DR = loss_DR_MSE(y_f_dr.float().cuda(),
                                          y_f.float().cuda()).to(device)
                else:
                    loss_F = loss_F_MSE(y_f_hat.float(),
                                        y_f.float()).to(device)
                    loss_DR = loss_DR_MSE(y_f_dr.float(),
                                          y_f.float()).to(device)

                loss = loss_F + ALPHA * loss_pi + BETA * loss_DR
                total_loss_val += loss.item()

            val_loss = total_loss_val
            valid_losses.append(val_loss)

            valid_loss = np.average(np.array(valid_losses))
            valid_losses = []
            early_stopping(valid_loss,
                           self.dr_net_phi,
                           self.dr_net_h_y0,
                           self.dr_net_h_y1,
                           self.pi_net,
                           self.mu_net)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if epoch % 100 == 0:
                print("---->>>[[epoch: {0}/3000]], loss, val: {1}"
                      .format(epoch,
                              valid_loss))

        torch.save(self.dr_net_phi.state_dict(), "DR_WO_Info_CFR/IHDP/Models/dr_net_phi.pth")
        torch.save(self.dr_net_h_y1.state_dict(), "DR_WO_Info_CFR/IHDP/Models/dr_net_y1.pth")
        torch.save(self.dr_net_h_y0.state_dict(), "DR_WO_Info_CFR/IHDP/Models/dr_net_y0.pth")
        torch.save(self.pi_net.state_dict(), "DR_WO_Info_CFR/IHDP/Models/pi_net.pth")
        torch.save(self.mu_net.state_dict(), "DR_WO_Info_CFR/IHDP/Models/mu_net.pth")

    def test(self, test_parameters, device):
        eval_set = test_parameters["tensor_dataset"]
        self.dr_net_phi.eval()
        self.dr_net_h_y0.eval()
        self.dr_net_h_y1.eval()
        self.pi_net.eval()
        self.mu_net.eval()

        _data_loader = torch.utils.data.DataLoader(eval_set,
                                                   shuffle=False)

        y1_hat_list = []
        y0_hat_list = []
        y1_true_list = []
        y0_true_list = []

        for batch in _data_loader:
            covariates_X, T, yf, ycf = batch
            covariates_X = covariates_X.to(device)
            y1_hat = self.dr_net_h_y1(self.dr_net_phi(covariates_X))
            y0_hat = self.dr_net_h_y0(self.dr_net_phi(covariates_X))

            y1_hat_list.append(y1_hat.item())
            y0_hat_list.append(y0_hat.item())

            y1_true = T * yf + (1 - T) * ycf
            y0_true = (1 - T) * yf + T * ycf

            y1_true_list.append(y1_true.item())
            y0_true_list.append(y0_true.item())

        return {
            "y1_hat_list": np.array(y1_hat_list),
            "y0_hat_list": np.array(y0_hat_list),
            "y1_true_list": np.array(y1_true_list),
            "y0_true_list": np.array(y0_true_list)
        }

    def train_pi_net(self, train_parameters, device):
        print(".. PS Training started ..")
        epochs = train_parameters["epochs"]
        batch_size = train_parameters["batch_size"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        train_set = train_parameters["train_set"]

        data_loader_train = torch.utils.data.DataLoader(train_set,
                                                        batch_size=batch_size,
                                                        shuffle=shuffle)

        optimizer = optim.Adam(self.pi_net.parameters(), lr=lr)
        lossBCE = nn.BCELoss()
        for epoch in range(epochs):
            epoch += 1
            self.pi_net.train()  # Set model to training mode
            total_loss = 0
            total_correct = 0
            train_set_size = 0

            for batch in data_loader_train:
                covariates, T = batch
                covariates = covariates.to(device)
                T = T.to(device).float()
                train_set_size += covariates.size(0)

                pi = self.pi_net(covariates)
                loss = lossBCE(pi, T).to(device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # pred_accuracy = total_correct / train_set_size
            if epoch % 100 == 0:
                print("Epoch: {0}, loss: {1}".
                      format(epoch, total_loss))
        print("Training Completed..")

    def eval_pi_net(self, eval_parameters, device):
        eval_set = eval_parameters["eval_set"]
        self.pi_net.eval()
        data_loader = torch.utils.data.DataLoader(eval_set, shuffle=False)
        eval_set_size = 0
        pi_list = []
        for batch in data_loader:
            covariates, T = batch
            covariates = covariates.to(device)
            eval_set_size += covariates.size(0)
            pi = self.pi_net(covariates)
            pi = pi.squeeze().item()
            pi_list.append(pi)

        np_pi = Utils.Utils.convert_to_col_vector(np.array(pi_list))
        print(np_pi)
        print("########")
        print(np.log(pi_list))
        return np_pi

    def train_mu_net(self, train_parameters, device):
        print(".. mu net Training started ..")
        epochs = train_parameters["epochs"]
        batch_size = train_parameters["batch_size"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        weight_decay = train_parameters["lambda"]
        train_set = train_parameters["train_set"]

        data_loader_train = torch.utils.data.DataLoader(train_set,
                                                        batch_size=batch_size,
                                                        shuffle=shuffle)

        optimizer_mu = optim.Adam(self.mu_net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_F_MSE = nn.MSELoss()

        for epoch in range(epochs):
            epoch += 1
            self.mu_net.train()  # Set model to training mode
            total_loss = 0
            train_set_size = 0

            for batch in data_loader_train:
                covariates, T, yf = batch
                covariates = covariates.to(device)
                T = T.to(device)

                mu = self.mu_net(covariates, T)
                if torch.cuda.is_available():
                    loss = loss_F_MSE(mu.float().cuda(), yf.float().cuda()).to(device)
                else:
                    loss = loss_F_MSE(mu.float(), yf.float()).to(device)

                optimizer_mu.zero_grad()
                loss.backward()
                optimizer_mu.step()

                total_loss += loss.item()

            # pred_accuracy = total_correct / train_set_size
            if epoch % 100 == 0:
                print("Epoch: {0}, loss: {1}".
                      format(epoch, total_loss))
        print("Training Completed..")

    def eval_mu_net(self, eval_parameters, device):
        eval_set = eval_parameters["eval_set"]
        self.mu_net.eval()
        data_loader = torch.utils.data.DataLoader(eval_set, shuffle=False)
        mu_list = []
        for batch in data_loader:
            covariates, T, _ = batch
            covariates = covariates.to(device)
            T = T.to(device)

            mu = self.mu_net(covariates, T)
            mu = mu.squeeze().item()
            mu_list.append(mu)

        np_mu = Utils.Utils.convert_to_col_vector(np.array(mu_list))
        return np_mu
