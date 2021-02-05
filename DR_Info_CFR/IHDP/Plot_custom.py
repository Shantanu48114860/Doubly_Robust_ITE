import matplotlib.pyplot as plt
import numpy as np


def plot_CEVAE_data_ATE():
    xs = np.array([3.0, 3.48, 3.7, 4.0, 4.48])
    y_LR1 = np.array([0.1, 0.103, 0.11, 0.108, 0.108])
    y_LR2 = np.array([0.09, 0.099, 0.108, 0.107, 0.108])

    y_TARNET = np.array([0.10, 0.099, 0.105, 0.103, 0.1028])
    y_CEVAE_cont = np.array([0.14, 0.049, 0.08, 0.075, 0.046])
    y_CEVAE = np.array([0.0401, 0.049, 0.045, 0.04, 0.041])

    y_dr_vidal = np.array([0.007, 0.025, 0.03, 0.012, 0.02])
    y_dr_vidal_wo_DR = np.array([0.16, 0.030, 0.05, 0.026, 0.07])

    plt.xticks(np.array([0.0, 3.0, 3.5, 4.0, 4.5]))
    plt.yticks(np.array([0.00, 0.04, 0.08, 0.12, 0.16]))
    # plt.ylim(0, 0.16)
    plt.grid(True)

    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.plot(xs, y_dr_vidal, marker='*', markersize=7, color='r',
             label='DR-VIDAL')
    plt.plot(xs, y_dr_vidal_wo_DR, marker='.', markersize=8, color='salmon',
             label='DR-VIDAL w/o DR')

    plt.plot(xs, y_LR1, marker='d', linestyle='--', color='grey',
             label='LR 1')
    plt.plot(xs, y_LR2, marker='d', linestyle='--', color='purple',
             label='LR 2')

    plt.plot(xs, y_TARNET, marker='o', color='orange',
             label='TARnet')

    plt.plot(xs, y_CEVAE, marker='v', markersize=7, color='deepskyblue',
             label='CEVAE')
    plt.plot(xs, y_CEVAE_cont, marker='^', markersize=7, color='darkorange',
             label='CEVAE cont')

    plt.xlabel('log(Nsamples)', fontsize=10)
    plt.ylabel('absolute ATE error', fontsize=10)
    plt.legend(loc='upper right')
    plt.draw()
    plt.savefig('Plots/Graphs/CEVAE_ATE_big', dpi=220)
    plt.clf()


def plot_CEVAE_data_PEHE():
    xs = np.array([3.0, 3.48, 3.7, 4.0, 4.48])
    y_GANITE = np.array([0.2, 0.135, 0.1, 0.1021, 0.0848])

    y_dr_vidal = np.array([0.18, 0.03, 0.038, 0.034, 0.033])
    y_dr_vidal_wo_DR = np.array([0.24, 0.05, 0.036, 0.038, 0.065])

    plt.xticks(np.array([0.0, 3.0, 3.5, 4.0, 4.5]))
    plt.yticks(np.array([0.00, 0.04, 0.08, 0.12, 0.18, 0.24, 0.65]))
    # plt.ylim(0, 0.16)
    plt.grid(True)

    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.plot(xs, y_dr_vidal, marker='d', markersize=7, color='r',
             label='DR-VIDAL')
    plt.plot(xs, y_dr_vidal_wo_DR, marker='*', markersize=7, color='darkorange',
             label='DR-VIDAL w/o DR')

    plt.plot(xs, y_GANITE, marker='^', markersize=7, color='deepskyblue',
             label='GANITE')

    plt.xlabel('log(Nsamples)', fontsize=10)
    plt.ylabel('PEHE error', fontsize=10)
    plt.legend(loc='upper right')
    plt.draw()
    plt.savefig('Plots/Graphs/CEVAE_PEHE_big', dpi=220)
    plt.clf()


def plot_DR_VIDAL_data_PEHE():
    xs = np.array([3.0, 3.48, 3.7, 4.0, 4.48])

    y_dr_vidal = np.array([0.90525604875, 0.86391737696, 0.9969492485, 0.778874472, 0.68214507637])
    y_dr_vidal_wo_DR = np.array([0.94987770404, 0.87157293555, 1.0203612826, 0.8122446968, 0.68304703824])
    y_GANITE = np.array([1.1840608204, 1.0189915947, 1.1878590003, 0.90735760813, 0.90606554476])

    plt.xticks(np.array([0.0, 3.0, 3.5, 4.0, 4.5]))
    plt.yticks(np.array([0.0, 0.6, 0.75, 0.90, 1.05, 1.18]))
    # plt.ylim(0, 0.16)
    plt.grid(True)

    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.plot(xs, y_dr_vidal, marker='d', markersize=7, color='r',
             label='DR-VIDAL')
    plt.plot(xs, y_dr_vidal_wo_DR, marker='*', markersize=7, color='darkorange',
             label='DR-VIDAL w/o DR')

    plt.plot(xs, y_GANITE, marker='^', markersize=7, color='deepskyblue',
             label='GANITE')

    plt.xlabel('log(Nsamples)', fontsize=10)
    plt.ylabel('log(PEHE error)', fontsize=10)
    plt.legend(loc='upper right')
    plt.draw()
    plt.savefig('Plots/Graphs/DR_VIDAL_PEHE_big', dpi=220)
    plt.clf()


plot_CEVAE_data_ATE()
plot_CEVAE_data_PEHE()
plot_DR_VIDAL_data_PEHE()
