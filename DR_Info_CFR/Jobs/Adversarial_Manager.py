import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from Adversarial_model import Adversarial_VAE, Generator, Discriminator, QHead
from Constants import Constants
from Utils import Utils, NormalNLLLoss


class Adversarial_Manager:
    def __init__(self, encoder_input_nodes, encoder_shared_nodes,
                 encoder_x_out_nodes, encoder_t_out_nodes,
                 encoder_yf_out_nodes, encoder_ycf_out_nodes,
                 decoder_in_nodes, decoder_shared_nodes,
                 decoder_out_nodes,
                 gen_in_nodes, gen_shared_nodes, gen_out_nodes,
                 dis_in_nodes, dis_shared_nodes, dis_out_nodes,
                 Q_in_nodes, Q_shared_nodes, Q_out_nodes,
                 device):
        self.adversarial_vae = Adversarial_VAE(encoder_input_nodes=encoder_input_nodes,
                                               encoder_shared_nodes=encoder_shared_nodes,
                                               encoder_x_out_nodes=encoder_x_out_nodes,
                                               encoder_t_out_nodes=encoder_t_out_nodes,
                                               encoder_yf_out_nodes=encoder_yf_out_nodes,
                                               encoder_ycf_out_nodes=encoder_ycf_out_nodes,
                                               decoder_in_nodes=decoder_in_nodes,
                                               decoder_shared_nodes=decoder_shared_nodes,
                                               decoder_out_nodes=decoder_out_nodes).to(device)

        self.netG = Generator(in_nodes=gen_in_nodes,
                              shared_nodes=gen_shared_nodes,
                              out_nodes=gen_out_nodes).to(device)

        self.netD = Discriminator(in_nodes=dis_in_nodes,
                                  shared_nodes=dis_shared_nodes,
                                  out_nodes=dis_out_nodes).to(device)

        self.netQ = QHead(in_nodes=Q_in_nodes,
                          shared_nodes=Q_shared_nodes,
                          out_nodes=Q_out_nodes).to(device)

    def train_adversarial_model(self, train_parameters, device):
        epochs = train_parameters["epochs"]
        batch_size = train_parameters["batch_size"]
        vae_lr = train_parameters["vae_lr"]
        gan_G_lr = train_parameters["gan_G_lr"]
        gan_D_lr = train_parameters["gan_D_lr"]
        weight_decay = train_parameters["lambda"]
        shuffle = train_parameters["shuffle"]
        train_dataset = train_parameters["train_dataset"]
        VAE_BETA = train_parameters["VAE_BETA"]
        INFO_GAN_LAMBDA = train_parameters["INFO_GAN_LAMBDA"]
        INFO_GAN_ALPHA = train_parameters["INFO_GAN_ALPHA"]

        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=shuffle)

        adv_vae_optimizer = optim.Adam(params=self.adversarial_vae.parameters(),
                                       lr=vae_lr, weight_decay=weight_decay)
        D_optimizer = optim.Adam(params=self.netD.parameters(),
                                 lr=gan_D_lr, weight_decay=weight_decay,
                                 betas=(0.5, 0.999))
        G_optimizer = optim.Adam(
            [
                {'params': self.netG.parameters()},
                {'params': self.netQ.parameters()}
            ],
            lr=gan_G_lr,
            weight_decay=weight_decay,
            betas=(0.5, 0.999))

        train_loss_avg = []

        loss_VAE_recons_MSE = nn.MSELoss()
        loss_Q_con = NormalNLLLoss()
        loss_D = nn.BCELoss()
        loss_G_F = nn.MSELoss()

        for epoch in range(epochs):
            epoch += 1
            total_loss_train = 0
            with tqdm(total=len(train_data_loader)) as t:
                # VAE training
                self.adversarial_vae.train()
                for batch in train_data_loader:
                    covariates_X, T, e, y_f = batch
                    batch_n = covariates_X.size(0)
                    covariates_X = covariates_X.to(device)
                    T = T.to(device)
                    T_float = T.float().to(device)

                    # vae reconstruction
                    [covariates_X_hat, latent_z_code,
                     latent_mu_x, latent_log_var_x,
                     latent_mu_t, latent_log_var_t,
                     latent_mu_yf, latent_log_var_yf,
                     latent_mu_ycf, latent_log_var_ycf] = self.adversarial_vae(covariates_X)

                    # reconstruction error
                    if torch.cuda.is_available():
                        loss_recons = loss_VAE_recons_MSE(covariates_X_hat.float().cuda(),
                                                          covariates_X.float().cuda()).to(device)
                    else:
                        loss_recons = loss_VAE_recons_MSE(covariates_X_hat.float(),
                                                          covariates_X.float()).to(device)

                    loss_KL_X = Utils.vae_loss(latent_mu_x, latent_log_var_x)
                    loss_KL_T = Utils.vae_loss(latent_mu_t, latent_log_var_t)
                    loss_KL_yf = Utils.vae_loss(latent_mu_yf, latent_log_var_yf)
                    loss_KL_ycf = Utils.vae_loss(latent_mu_ycf, latent_log_var_ycf)

                    loss_VAE = loss_recons + VAE_BETA * (loss_KL_X + loss_KL_T + loss_KL_yf + loss_KL_ycf)

                    adv_vae_optimizer.zero_grad()
                    loss_VAE.backward()
                    adv_vae_optimizer.step()

                    # GAN training
                    self.netG.train()
                    self.netD.train()
                    self.netQ.train()
                    latent_z_code = latent_z_code.detach()
                    # sample from uniform(-1, 1)
                    noise_z_size = (Constants.Info_GAN_Gen_in_nodes - Constants.Decoder_in_nodes)
                    # noise_z = (-2) * torch.rand(batch_n, noise_z_size) + 1
                    noise_z = torch.randn(batch_n, noise_z_size)

                    noise_netG_input = torch.cat((latent_z_code, noise_z), dim=1)

                    y0, y1 = self.netG(noise_netG_input)
                    y_f_hat = T * y1 + (1 - T) * y0

                    y0_sigmoid = torch.sigmoid(y0)
                    y1_sigmoid = torch.sigmoid(y1)

                    netD_y0 = ((1 - T) * y_f + T * y0_sigmoid).type(torch.FloatTensor)  # if t = 0
                    netD_y1 = (T * y_f + (1 - T) * y1_sigmoid).type(torch.FloatTensor)  # if t = 1

                    # Discriminator loss and gradients

                    d_logit = self.netD(covariates_X,
                                        netD_y0.to(device).detach(),
                                        netD_y1.to(device).detach())
                    loss_Discriminator = loss_D(d_logit, T_float).to(device)
                    D_optimizer.zero_grad()
                    loss_Discriminator.backward(retain_graph=True)
                    D_optimizer.step()

                    if torch.cuda.is_available():
                        loss_Generator_F = loss_G_F(y_f_hat.float().cuda(),
                                                    y_f.float().cuda()).to(device)
                    else:
                        loss_Generator_F = loss_G_F(y_f_hat.float(),
                                                    y_f.float()).to(device)

                    # Generator Training
                    d_logit = self.netD(covariates_X,
                                        netD_y0.to(device),
                                        netD_y1.to(device))

                    # Q training
                    q_input = torch.cat((y0, y1), dim=1)
                    q_mu0, q_var0 = self.netQ(q_input)
                    con_loss0 = loss_Q_con(latent_z_code, q_mu0, q_var0) * 0.1

                    # Generator and Q losses and gradients
                    loss_Generator = -loss_D(d_logit, T_float).to(device)
                    loss_Info = INFO_GAN_LAMBDA * (con_loss0)
                    loss_Generator_total = loss_Generator + INFO_GAN_ALPHA * loss_Generator_F + loss_Info

                    G_optimizer.zero_grad()

                    loss_Generator_total.backward()
                    G_optimizer.step()

                    total_loss_train += loss_VAE.item() + loss_Discriminator.item() + loss_Generator_total.item()
                    t.set_postfix(epoch='{0}'.format(epoch),
                                  loss_train='{:05.3f}'.format(total_loss_train))
                    t.update()

        torch.save(self.adversarial_vae.state_dict(), "Models/adversarial_vae.pth")
        torch.save(self.netG.state_dict(), "Models/netG.pth")
        torch.save(self.netD.state_dict(), "Models/netD.pth")
        torch.save(self.netQ.state_dict(), "Models/netQ0.pth")

    def test_adversarial_model(self, train_parameters, device):
        eval_set = train_parameters["tensor_dataset"]
        _data_loader = torch.utils.data.DataLoader(eval_set,
                                                   shuffle=False)

        self.adversarial_vae.eval()
        self.netG.eval()
        self.netD.eval()
        self.netQ.eval()

        ycf_list = []
        for batch in _data_loader:
            covariates_X, T, y_f, _ = batch
            batch_n = covariates_X.size(0)
            covariates_X = covariates_X.to(device)
            # vae test
            [_, latent_z_code, _, _, _, _, _, _, _, _] = self.adversarial_vae(covariates_X)

            # GAN test
            latent_z_code = latent_z_code.detach()
            # sample from uniform(-1, 1)
            noise_z_size = (Constants.Info_GAN_Gen_in_nodes - Constants.Decoder_in_nodes)
            # noise_z = (-2) * torch.rand(batch_n, noise_z_size) + 1
            noise_z = torch.randn(batch_n, noise_z_size)
            noise_netG_input = torch.cat((latent_z_code, noise_z), dim=1)
            # noise_netG_input = latent_z_code

            y0, y1 = self.netG(noise_netG_input)
            y0_sigmoid = torch.sigmoid(y0)
            y1_sigmoid = torch.sigmoid(y1)
            y_cf = T * y0_sigmoid + (1 - T) * y1_sigmoid
            # y_cf = T * y0 + (1 - T) * y1
            ycf_list.append(y_cf.item())

        return Utils.convert_to_col_vector(np.array(ycf_list))
