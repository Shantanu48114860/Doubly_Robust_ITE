import torch
import torch.nn as nn
import torch.nn.functional as F

from Constants import Constants


class Encoder_shared(nn.Module):
    def __init__(self, input_nodes=Constants.DRNET_INPUT_NODES,
                 shared_nodes=Constants.Encoder_shared_nodes):
        super(Encoder_shared, self).__init__()
        self.shared1 = nn.Linear(in_features=input_nodes, out_features=shared_nodes)

        self.shared2 = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

        self.shared3 = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

    def forward(self, x):
        x = F.relu((self.shared1(x)))
        x = F.relu((self.shared2(x)))
        x = F.relu((self.shared3(x)))
        return x


class Encoder_x(nn.Module):
    def __init__(self, shared_nodes=Constants.Encoder_shared_nodes,
                 out_nodes=Constants.Encoder_x_nodes):
        super(Encoder_x, self).__init__()
        self.out = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

        self.fc_mu = nn.Linear(in_features=shared_nodes, out_features=out_nodes)
        self.fc_log_var = nn.Linear(in_features=shared_nodes, out_features=out_nodes)

    def forward(self, x):
        x = F.relu((self.out(x)))
        x_mu = self.fc_mu(x)
        x_log_var = self.fc_log_var(x)
        return x_mu, x_log_var


class Encoder_t(nn.Module):
    def __init__(self, shared_nodes=Constants.Encoder_shared_nodes,
                 out_nodes=Constants.Encoder_t_nodes):
        super(Encoder_t, self).__init__()
        self.out = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

        self.fc_mu = nn.Linear(in_features=shared_nodes, out_features=out_nodes)
        self.fc_log_var = nn.Linear(in_features=shared_nodes, out_features=out_nodes)

    def forward(self, x):
        x = F.relu((self.out(x)))
        x_mu = self.fc_mu(x)
        x_log_var = self.fc_log_var(x)
        return x_mu, x_log_var


class Encoder_yf(nn.Module):
    def __init__(self, shared_nodes=Constants.Encoder_shared_nodes,
                 out_nodes=Constants.Encoder_yf_nodes):
        super(Encoder_yf, self).__init__()
        self.out = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

        self.fc_mu = nn.Linear(in_features=shared_nodes, out_features=out_nodes)
        self.fc_log_var = nn.Linear(in_features=shared_nodes, out_features=out_nodes)

    def forward(self, x):
        x = F.relu((self.out(x)))
        x_mu = self.fc_mu(x)
        x_log_var = self.fc_log_var(x)
        return x_mu, x_log_var


class Encoder_ycf(nn.Module):
    def __init__(self, shared_nodes=Constants.Encoder_shared_nodes,
                 out_nodes=Constants.Encoder_ycf_nodes):
        super(Encoder_ycf, self).__init__()
        self.out = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

        self.fc_mu = nn.Linear(in_features=shared_nodes, out_features=out_nodes)
        self.fc_log_var = nn.Linear(in_features=shared_nodes, out_features=out_nodes)

    def forward(self, x):
        x = F.relu((self.out(x)))
        x_mu = self.fc_mu(x)
        x_log_var = self.fc_log_var(x)
        return x_mu, x_log_var


class Decoder(nn.Module):
    def __init__(self, in_nodes=Constants.Decoder_in_nodes,
                 shared_nodes=Constants.Decoder_shared_nodes,
                 out_nodes=Constants.Decoder_out_nodes):
        super(Decoder, self).__init__()
        self.shared1 = nn.Linear(in_features=in_nodes, out_features=shared_nodes)

        self.shared2 = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

        self.shared3 = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

        self.out_nodes = nn.Linear(in_features=shared_nodes, out_features=out_nodes)

    def forward(self, x):
        x = F.relu((self.shared1(x)))
        x = F.relu((self.shared2(x)))
        x = F.relu((self.shared3(x)))
        x = self.out_nodes(x)
        return x


class Adversarial_VAE(nn.Module):
    def __init__(self, encoder_input_nodes=Constants.DRNET_INPUT_NODES,
                 encoder_shared_nodes=Constants.Encoder_shared_nodes,
                 encoder_x_out_nodes=Constants.Encoder_x_nodes,
                 encoder_t_out_nodes=Constants.Encoder_t_nodes,
                 encoder_yf_out_nodes=Constants.Encoder_yf_nodes,
                 encoder_ycf_out_nodes=Constants.Encoder_ycf_nodes,
                 decoder_in_nodes=Constants.Decoder_in_nodes,
                 decoder_shared_nodes=Constants.Decoder_shared_nodes,
                 decoder_out_nodes=Constants.Decoder_out_nodes):
        super(Adversarial_VAE, self).__init__()
        self.encoder_shared = Encoder_shared(input_nodes=encoder_input_nodes,
                                             shared_nodes=encoder_shared_nodes)
        self.encoder_x = Encoder_x(shared_nodes=encoder_shared_nodes,
                                   out_nodes=encoder_x_out_nodes)
        self.encoder_t = Encoder_t(shared_nodes=encoder_shared_nodes,
                                   out_nodes=encoder_t_out_nodes)
        self.encoder_yf = Encoder_yf(shared_nodes=encoder_shared_nodes,
                                     out_nodes=encoder_yf_out_nodes)
        self.encoder_ycf = Encoder_ycf(shared_nodes=encoder_shared_nodes,
                                       out_nodes=encoder_ycf_out_nodes)
        self.decoder = Decoder(in_nodes=decoder_in_nodes,
                               shared_nodes=decoder_shared_nodes,
                               out_nodes=decoder_out_nodes)

    def re_parametrize(self, mu, log_var):
        # the re-parameterization trick
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()

            return mu + eps * std

        else:
            return mu

    def forward(self, x):
        latent_mu_x, latent_log_var_x = self.encoder_x(self.encoder_shared(x))
        latent_mu_t, latent_log_var_t = self.encoder_t(self.encoder_shared(x))
        latent_mu_yf, latent_log_var_yf = self.encoder_yf(self.encoder_shared(x))
        latent_mu_ycf, latent_log_var_ycf = self.encoder_ycf(self.encoder_shared(x))

        latent_z_x = self.re_parametrize(latent_mu_x, latent_log_var_x)
        latent_z_t = self.re_parametrize(latent_mu_t, latent_log_var_t)
        latent_z_yf = self.re_parametrize(latent_mu_yf, latent_log_var_yf)
        latent_z_ycf = self.re_parametrize(latent_mu_ycf, latent_log_var_ycf)

        latent_z = torch.cat((latent_z_x, latent_z_t, latent_z_yf, latent_z_ycf), 1)
        x_hat = self.decoder(latent_z)

        return x_hat, latent_z, \
               latent_mu_x, latent_log_var_x, \
               latent_mu_t, latent_log_var_t, \
               latent_mu_yf, latent_log_var_yf, \
               latent_mu_ycf, latent_log_var_ycf


class Generator(nn.Module):
    def __init__(self, in_nodes=Constants.Info_GAN_Gen_in_nodes,
                 shared_nodes=Constants.Info_GAN_Gen_shared_nodes,
                 out_nodes=Constants.Info_GAN_Gen_out_nodes):
        super().__init__()
        self.shared1 = nn.Linear(in_features=in_nodes, out_features=shared_nodes)

        self.shared2 = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

        self.hidden_y0 = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

        self.out_y0 = nn.Linear(in_features=shared_nodes, out_features=out_nodes)

        self.hidden_y1 = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

        self.out_y1 = nn.Linear(in_features=shared_nodes, out_features=out_nodes)

        # self.bn1 = nn.BatchNorm1d(shared_nodes)

    def forward(self, y):
        y = F.relu(self.shared1(y))
        y = F.relu(self.shared2(y))

        y0 = F.relu(self.hidden_y0(y))
        y0 = self.out_y0(y0)

        y1 = F.relu(self.hidden_y1(y))
        y1 = self.out_y1(y1)

        return y0, y1


class Discriminator(nn.Module):
    def __init__(self, in_nodes=Constants.Info_GAN_Dis_in_nodes,
                 shared_nodes=Constants.Info_GAN_Dis_shared_nodes,
                 out_nodes=Constants.Info_GAN_Dis_out_nodes):
        super().__init__()
        self.hidden1 = nn.Linear(in_features=in_nodes, out_features=shared_nodes)

        self.hidden2 = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

        self.out = nn.Linear(in_features=shared_nodes, out_features=out_nodes)

        # self.bn1 = nn.BatchNorm1d(shared_nodes)

    def forward(self, x, y0, y1):
        x_t = torch.cat((x, y0, y1), 1)
        x_t = F.relu(self.hidden1(x_t))
        x_t = F.relu(self.hidden2(x_t))
        x_t = torch.sigmoid(self.out(x_t))

        return x_t


class QHead(nn.Module):
    def __init__(self, in_nodes=Constants.Info_GAN_Q_in_nodes,
                 shared_nodes=Constants.Info_GAN_Q_shared_nodes,
                 out_nodes=Constants.Info_GAN_Q_out_nodes):
        super().__init__()

        self.hidden1 = nn.Linear(in_features=in_nodes, out_features=shared_nodes)

        self.hidden2 = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

        # self.bn1 = nn.BatchNorm1d(shared_nodes)

        self.out_mu = nn.Linear(in_features=shared_nodes, out_features=out_nodes)
        self.out_var = nn.Linear(in_features=shared_nodes, out_features=out_nodes)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))

        mu = self.out_mu(x).squeeze()
        var = torch.exp(self.out_var(x).squeeze())

        return mu, var
