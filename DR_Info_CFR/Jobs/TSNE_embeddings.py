import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import manifold

from Adversarial_model import Adversarial_VAE, Generator, Discriminator, QHead_y0, QHead_y1
from Constants import Constants
from Utils import Utils
from dataloader import DataLoader


def draw_digits_latent_space(z_mean_test, test_label, title, xlim, ylim):
    # colors = ['red', 'green', 'blue', 'purple']
    colors = ['#FF4000', '#0101DF']
    plt.xlim(xlim)
    plt.ylim(ylim)
    # plt.yticks(np.arange(-3, 3, 0.25))
    # plt.xticks(np.arange(-3, 3, 0.25))
    plt.scatter(z_mean_test[:, 0],
                z_mean_test[:, 1],
                c=test_label,
                cmap=matplotlib.colors.ListedColormap(colors))
    # cmap=plt.cm.get_cmap('jet', 2))
    # plt.title(title)
    plt.colorbar()
    plt.draw()
    plt.savefig("Plots/" + title, dpi=220)
    plt.clf()


device = Utils.get_device()
train_path = "Dataset/jobs_DW_bin.new.10.train.npz"
test_path = "Dataset/jobs_DW_bin.new.10.test.npz"
adversarial_vae = Adversarial_VAE(encoder_input_nodes=Constants.DRNET_INPUT_NODES,
                                  encoder_shared_nodes=Constants.Encoder_shared_nodes,
                                  encoder_x_out_nodes=Constants.Encoder_x_nodes,
                                  encoder_t_out_nodes=Constants.Encoder_t_nodes,
                                  encoder_yf_out_nodes=Constants.Encoder_yf_nodes,
                                  encoder_ycf_out_nodes=Constants.Encoder_ycf_nodes,
                                  decoder_in_nodes=Constants.Decoder_in_nodes,
                                  decoder_shared_nodes=Constants.Decoder_shared_nodes,
                                  decoder_out_nodes=Constants.Decoder_out_nodes).to(device)

netG = Generator(in_nodes=Constants.Info_GAN_Gen_in_nodes,
                 shared_nodes=Constants.Info_GAN_Gen_shared_nodes,
                 out_nodes=Constants.Info_GAN_Gen_out_nodes).to(device)

netD = Discriminator(in_nodes=Constants.Info_GAN_Dis_in_nodes,
                     shared_nodes=Constants.Info_GAN_Dis_shared_nodes,
                     out_nodes=Constants.Info_GAN_Dis_out_nodes).to(device)

netQ0 = QHead_y0(in_nodes=Constants.Info_GAN_Q_in_nodes,
                 shared_nodes=Constants.Info_GAN_Q_shared_nodes,
                 out_nodes=Constants.Info_GAN_Q_out_nodes).to(device)

netQ1 = QHead_y1(in_nodes=Constants.Info_GAN_Q_in_nodes,
                 shared_nodes=Constants.Info_GAN_Q_shared_nodes,
                 out_nodes=Constants.Info_GAN_Q_out_nodes).to(device)

adversarial_vae.load_state_dict(torch.load("Models/adversarial_vae.pth", map_location=device))
netG.load_state_dict(torch.load("Models/netG.pth", map_location=device))
netD.load_state_dict(torch.load("Models/netD.pth", map_location=device))
netQ0.load_state_dict(torch.load("Models/netQ0.pth", map_location=device))
netQ1.load_state_dict(torch.load("Models/netQ1.pth", map_location=device))

adversarial_vae.eval()
netG.eval()
netD.eval()
netQ0.eval()
netQ1.eval()

dL = DataLoader()
csv_path = "Dataset/Twin_data.csv"
np_train_X, np_train_T, np_train_e, np_train_yf, \
np_test_X, np_test_T, np_test_e, np_test_yf = \
    dL.load_train_test_jobs(train_path, test_path, 1)
codes_T = dict(mu_T=list(), logvar_T=list(), T=list())
tensor_test = Utils.convert_to_tensor(np_test_X, np_test_T, np_test_yf, np_test_e)
_data_loader = torch.utils.data.DataLoader(tensor_test,
                                           shuffle=False,
                                           batch_size=642)

means_X = np.empty((642, 5))
means_yf = np.empty((642, 1))
means_T = np.empty((642, 1))

labels_yf = np.empty((642, 1))
labels_T = np.empty((642, 1))
labels_ycf = np.empty((642, 1))
for batch in _data_loader:
    covariates_X, T, y_f, y_cf = batch
    batch_n = covariates_X.size(0)
    covariates_X = covariates_X.to(device)
    T = T.to(device)
    T_float = T.float().to(device)

    # vae reconstruction
    [covariates_X_hat, latent_z_code,
     latent_mu_x, latent_log_var_x,
     latent_mu_t, latent_log_var_t,
     latent_mu_yf, latent_log_var_yf,
     latent_mu_ycf, latent_log_var_ycf] = adversarial_vae(covariates_X)

    means_X = latent_mu_x.detach().numpy()
    means_yf = Utils.convert_to_col_vector(latent_mu_yf.detach().numpy())
    means_T = Utils.convert_to_col_vector(latent_mu_t.detach().numpy())
    labels_yf = y_f.numpy()
    labels_ycf = y_cf.numpy()
    labels_T = T.numpy()

print("------")
mean = np.concatenate((means_X, means_yf, means_T), axis=1)
# print(mean.shape)
# print(labels_yf)
# print(labels_ycf)

tsne = manifold.TSNE(n_components=2)
z_tsne = tsne.fit_transform(means_X)

plt.figure(figsize=(10, 6))
draw_digits_latent_space(z_tsne, labels_T, "After Training latent z vs T", xlim=[-5, 5], ylim=[-5, 5])
draw_digits_latent_space(z_tsne, labels_yf, "After Training latent z vs y_f", xlim=[-5, 5], ylim=[-5, 5])
draw_digits_latent_space(z_tsne, labels_ycf, "After Training latent z vs y_cf", xlim=[-5, 5], ylim=[-5, 5])
# draw_digits_latent_space(z_tsne, labels_T, "Before Training latent z vs T")


# plt.show()

# before training VAE
adversarial_vae = Adversarial_VAE(encoder_input_nodes=Constants.DRNET_INPUT_NODES,
                                  encoder_shared_nodes=Constants.Encoder_shared_nodes,
                                  encoder_x_out_nodes=Constants.Encoder_x_nodes,
                                  encoder_t_out_nodes=Constants.Encoder_t_nodes,
                                  encoder_yf_out_nodes=Constants.Encoder_yf_nodes,
                                  encoder_ycf_out_nodes=Constants.Encoder_ycf_nodes,
                                  decoder_in_nodes=Constants.Decoder_in_nodes,
                                  decoder_shared_nodes=Constants.Decoder_shared_nodes,
                                  decoder_out_nodes=Constants.Decoder_out_nodes).to(device)

netG = Generator(in_nodes=Constants.Info_GAN_Gen_in_nodes,
                 shared_nodes=Constants.Info_GAN_Gen_shared_nodes,
                 out_nodes=Constants.Info_GAN_Gen_out_nodes).to(device)

netD = Discriminator(in_nodes=Constants.Info_GAN_Dis_in_nodes,
                     shared_nodes=Constants.Info_GAN_Dis_shared_nodes,
                     out_nodes=Constants.Info_GAN_Dis_out_nodes).to(device)

netQ0 = QHead_y0(in_nodes=Constants.Info_GAN_Q_in_nodes,
                 shared_nodes=Constants.Info_GAN_Q_shared_nodes,
                 out_nodes=Constants.Info_GAN_Q_out_nodes).to(device)

netQ1 = QHead_y1(in_nodes=Constants.Info_GAN_Q_in_nodes,
                 shared_nodes=Constants.Info_GAN_Q_shared_nodes,
                 out_nodes=Constants.Info_GAN_Q_out_nodes).to(device)

adversarial_vae.eval()
netG.eval()
netD.eval()
netQ0.eval()
netQ1.eval()


means_X = np.empty((642, 5))
means_yf = np.empty((642, 1))
means_T = np.empty((642, 1))

labels_yf = np.empty((642, 1))
labels_T = np.empty((642, 1))
for batch in _data_loader:
    covariates_X, T, y_f, e = batch
    batch_n = covariates_X.size(0)
    covariates_X = covariates_X.to(device)
    T = T.to(device)
    T_float = T.float().to(device)

    # vae reconstruction
    [covariates_X_hat, latent_z_code,
     latent_mu_x, latent_log_var_x,
     latent_mu_t, latent_log_var_t,
     latent_mu_yf, latent_log_var_yf,
     latent_mu_ycf, latent_log_var_ycf] = adversarial_vae(covariates_X)

    means_X = latent_mu_x.detach().numpy()
    means_yf = Utils.convert_to_col_vector(latent_mu_yf.detach().numpy())
    means_T = Utils.convert_to_col_vector(latent_mu_t.detach().numpy())
    labels_yf = y_f.numpy()
    labels_T = T.numpy()

print("------")
mean = np.concatenate((means_X, means_yf, means_T), axis=1)
# print(mean.shape)
# print(labels_yf)
# print(labels_ycf)

tsne = manifold.TSNE(n_components=2)
z_tsne = tsne.fit_transform(means_X)

plt.figure(figsize=(10, 6))
draw_digits_latent_space(z_tsne, labels_T, "Before Training latent z vs T", xlim=[-30, 40], ylim=[-40, 40])
draw_digits_latent_space(z_tsne, labels_yf, "Before Training latent z vs y_f", xlim=[-30, 40], ylim=[-40, 40])
draw_digits_latent_space(z_tsne, labels_ycf, "Before Training latent z vs y_cf", xlim=[-30, 40], ylim=[-40, 40])
# draw_digits_latent_space(z_tsne, labels_T, "Before Training latent z vs T")


# plt.show()
