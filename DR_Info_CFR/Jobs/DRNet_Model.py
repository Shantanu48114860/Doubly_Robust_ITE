import torch
import torch.nn as nn
import torch.nn.functional as F


class DRNetPhi(nn.Module):
    def __init__(self, input_nodes, shared_nodes=200):
        super(DRNetPhi, self).__init__()

        # shared layer
        self.shared1 = nn.Linear(in_features=input_nodes, out_features=shared_nodes)

        self.shared2 = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

        self.shared3 = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()
        # shared layers
        x = F.elu(self.shared1(x))
        x = F.elu(self.shared2(x))
        x = F.elu(self.shared3(x))

        return x


class DRNetH_Y1(nn.Module):
    def __init__(self, input_nodes=200, outcome_nodes=100):
        super(DRNetH_Y1, self).__init__()

        # potential outcome1 Y(1)
        self.hidden1_Y1 = nn.Linear(in_features=input_nodes, out_features=outcome_nodes)

        self.hidden2_Y1 = nn.Linear(in_features=outcome_nodes, out_features=outcome_nodes)

        self.out_Y1 = nn.Linear(in_features=outcome_nodes, out_features=1)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        # potential outcome1 Y(1)
        y1 = F.elu(self.hidden1_Y1(x))
        y1 = F.elu(self.hidden2_Y1(y1))
        y1 = torch.sigmoid(self.out_Y1(y1))

        return y1


class DRNetH_Y0(nn.Module):
    def __init__(self, input_nodes=200, outcome_nodes=100):
        super(DRNetH_Y0, self).__init__()

        # potential outcome1 Y(0)
        self.hidden1_Y0 = nn.Linear(in_features=input_nodes, out_features=outcome_nodes)

        self.hidden2_Y0 = nn.Linear(in_features=outcome_nodes, out_features=outcome_nodes)

        self.out_Y0 = nn.Linear(in_features=outcome_nodes, out_features=1)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        # potential outcome1 Y(0)
        y0 = F.elu(self.hidden1_Y0(x))
        y0 = F.elu(self.hidden2_Y0(y0))
        y0 = torch.sigmoid(self.out_Y0(y0))

        return y0


class pi_net(nn.Module):
    def __init__(self, input_nodes, outcome_nodes=200):
        super(pi_net, self).__init__()

        # propensity score
        self.hidden1_pi = nn.Linear(in_features=input_nodes, out_features=outcome_nodes)

        self.hidden2_pi = nn.Linear(in_features=outcome_nodes, out_features=outcome_nodes)

        self.out_pi = nn.Linear(in_features=outcome_nodes, out_features=1)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        # potential outcome1 Y(0)
        pi = F.elu(self.hidden1_pi(x))
        pi = F.elu(self.hidden2_pi(pi))
        pi = torch.sigmoid(self.out_pi(pi))

        return pi


class mu_net(nn.Module):
    def __init__(self, input_nodes, shared_nodes=200, outcome_nodes=100):
        super(mu_net, self).__init__()

        self.hidden1_mu = nn.Linear(in_features=input_nodes, out_features=shared_nodes)

        self.hidden2_mu = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

        self.hidden3_mu = nn.Linear(in_features=shared_nodes, out_features=outcome_nodes)

        self.hidden4_mu = nn.Linear(in_features=outcome_nodes, out_features=outcome_nodes)

        self.hidden5_mu = nn.Linear(in_features=outcome_nodes, out_features=outcome_nodes)

        self.out_mu = nn.Linear(in_features=outcome_nodes, out_features=1)

    def forward(self, x, t):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        x_t = torch.cat((x, t.float()), 1)

        mu = F.elu(self.hidden1_mu(x_t))
        mu = F.elu(self.hidden2_mu(mu))
        mu = F.elu(self.hidden3_mu(mu))

        mu = F.elu(self.hidden4_mu(mu))
        mu = F.elu(self.hidden5_mu(mu))
        mu = self.out_mu(mu)

        return mu


# class mu_net(nn.Module):
#     def __init__(self, input_nodes, shared_nodes=200, outcome_nodes=100):
#         super(mu_net, self).__init__()
#
#         self.hidden1_mu = nn.Linear(in_features=input_nodes, out_features=shared_nodes)
#
#         self.hidden2_mu = nn.Linear(in_features=shared_nodes, out_features=outcome_nodes)
#
#         # self.hidden3_mu = nn.Linear(in_features=shared_nodes, out_features=outcome_nodes)
#         #
#         # self.hidden4_mu = nn.Linear(in_features=outcome_nodes, out_features=outcome_nodes)
#         #
#         # self.hidden5_mu = nn.Linear(in_features=outcome_nodes, out_features=outcome_nodes)
#
#         self.out_mu = nn.Linear(in_features=outcome_nodes, out_features=1)
#
#     def forward(self, x, t):
#         if torch.cuda.is_available():
#             x = x.float().cuda()
#         else:
#             x = x.float()
#
#         x_t = torch.cat((x, t.float()), 1)
#
#         mu = F.elu(self.hidden1_mu(x_t))
#         mu = F.elu(self.hidden2_mu(mu))
#         # mu = F.elu(self.hidden3_mu(mu))
#         #
#         # mu = F.elu(self.hidden4_mu(mu))
#         # mu = F.elu(self.hidden5_mu(mu))
#         mu = torch.sigmoid(self.out_mu(mu))
#
#         return mu
