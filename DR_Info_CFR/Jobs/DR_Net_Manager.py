import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from DRNet_Model import DRNetPhi, DRNetH_Y1, DRNetH_Y0, pi_net, mu_net


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
        optimizer_pi = optim.Adam(self.pi_net.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_mu = optim.Adam(self.mu_net.parameters(), lr=lr, weight_decay=weight_decay)

        loss_F_MSE = nn.BCELoss()
        loss_CF_MSE = nn.BCELoss()
        loss_DR_F_MSE = nn.BCELoss()
        loss_DR_CF_MSE = nn.BCELoss()
        lossBCE = nn.BCELoss()

        for epoch in range(epochs):
            epoch += 1
            total_loss_train = 0
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
                    y_cf_hat = y1_hat * (1 - T_float) + y0_hat * T_float

                    y_f_dr = torch.sigmoid(T_float * ((T_float * y1_hat - (T_float - pi) * mu) / pi) + \
                             (1 - T_float) * (((1 - T_float) * y0_hat - (T_float - pi) * mu) / (1 - pi)))

                    y_cf_dr = torch.sigmoid((1 - T_float) * (((1 - T_float) * y1_hat - (T_float - pi) * mu) / pi) + \
                             T_float * ((T_float * y0_hat - (T_float - pi) * mu) / (1 - pi)))

                    loss_pi = lossBCE(pi, T_float).to(device)
                    if torch.cuda.is_available():
                        loss_F = loss_F_MSE(y_f_hat.float().cuda(),
                                            y_f.float().cuda()).to(device)
                        loss_CF = loss_CF_MSE(y_cf_hat.float().cuda(),
                                              y_cf.float().cuda()).to(device)

                        loss_DR_F = loss_DR_F_MSE(y_f_dr.float().cuda(),
                                                  y_f.float().cuda()).to(device)
                        loss_DR_CF = loss_DR_CF_MSE(y_cf_dr.float().cuda(),
                                                    y_cf.float().cuda()).to(device)
                    else:
                        loss_F = loss_F_MSE(y_f_hat.float(),
                                            y_f.float()).to(device)
                        loss_CF = loss_CF_MSE(y_cf_hat.float(),
                                              y_cf.float()).to(device)

                        loss_DR_F = loss_DR_F_MSE(y_f_dr.float(),
                                                  y_f.float()).to(device)
                        loss_DR_CF = loss_DR_CF_MSE(y_cf_dr.float(),
                                                    y_cf.float()).to(device)

                    loss = loss_F + loss_CF + ALPHA * loss_pi + BETA * (loss_DR_F + loss_DR_CF)
                    loss.backward()
                    total_loss_train += loss_F.item() + loss_CF.item() + loss_DR_F.item() + \
                                        loss_DR_CF.item() + loss_pi.item()

                    optimizer_pi.step()
                    optimizer_mu.step()
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
        self.pi_net.eval()
        self.mu_net.eval()

        _data_loader = torch.utils.data.DataLoader(eval_set,
                                                   shuffle=False)

        y_f_true_list = []
        y1_hat_list = []
        y0_hat_list = []
        e_list = []
        T_list = []

        for batch in _data_loader:
            covariates_X, T, e, y_f = batch
            covariates_X = covariates_X.to(device)
            y1_hat = self.dr_net_h_y1(self.dr_net_phi(covariates_X))
            y0_hat = self.dr_net_h_y0(self.dr_net_phi(covariates_X))

            y1_hat_list.append(y1_hat.item())
            y0_hat_list.append(y0_hat.item())

            y_f_true_list.append(y_f.item())
            e_list.append(e.item())
            T_list.append(T.item())

        return {
            "y1_hat_list": y1_hat_list,
            "y0_hat_list": y0_hat_list,
            "yf_list": y_f_true_list,
            "e_list": e_list,
            "T_list": T_list
        }
