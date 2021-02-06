import numpy as np
import matplotlib.pyplot as plt

csv_path = "MSE/DR_Results_Out_Final.csv"
data_X = np.loadtxt(csv_path, delimiter=",", skiprows=1)
print(data_X.shape)

T = (data_X[:, 1])

idx = (T == 1)
print("treated: ", T[idx].shape[0])
print("control: ", data_X.shape[0] - T[idx].shape[0])

diff_yf_DR_wo_DR = (data_X[:, -3])
diff_yf_DR = (data_X[:, -4])

diff_yf_DR_treated = diff_yf_DR[idx]
diff_yf_DR_control = diff_yf_DR[~idx]

diff_yf_DR_wo_DR_treated = diff_yf_DR_wo_DR[idx]
diff_yf_DR_wo_DR_control = diff_yf_DR_wo_DR[~idx]

DR_treated_best_arr = diff_yf_DR_treated[diff_yf_DR_treated <= diff_yf_DR_wo_DR_treated]
DR_control_best_arr = diff_yf_DR_control[diff_yf_DR_control <= diff_yf_DR_wo_DR_control]

wo_DR_treated_best_arr = diff_yf_DR_wo_DR_treated[diff_yf_DR_wo_DR_treated <= diff_yf_DR_treated]
wo_DR_control_best_arr = diff_yf_DR_wo_DR_control[diff_yf_DR_wo_DR_control <= diff_yf_DR_control]

print(DR_treated_best_arr.shape)
print(DR_control_best_arr.shape)

print(wo_DR_treated_best_arr.shape)
print(wo_DR_control_best_arr.shape)

treated_DR_prob = np.round(DR_treated_best_arr.shape[0] / T[idx].shape[0] * 100)
wo_DR_treated_DR_prob = np.round(wo_DR_treated_best_arr.shape[0] / T[idx].shape[0] * 100)
print(treated_DR_prob)
print(wo_DR_treated_DR_prob)

control_DR_prob = np.round(DR_control_best_arr.shape[0] / (data_X.shape[0] - T[idx].shape[0]) * 100)
wo_DR_control_DR_prob = np.round(wo_DR_control_best_arr.shape[0] / (data_X.shape[0] - T[idx].shape[0]) * 100)
print(control_DR_prob)
print(wo_DR_control_DR_prob)

treated_np = np.array(treated_DR_prob, wo_DR_treated_DR_prob)
control_np = np.array(wo_DR_control_DR_prob, control_DR_prob)

DR_Probs_np = np.array([treated_DR_prob, wo_DR_control_DR_prob])
wo_DR_Probs_np = np.array([wo_DR_treated_DR_prob, control_DR_prob])

N = 2

ind = np.arange(N)
width = 0.35

p1 = plt.bar(ind, DR_Probs_np, width)
p2 = plt.bar(ind, wo_DR_Probs_np, width,
             bottom=DR_Probs_np)

plt.ylabel('Percentage(%)')
plt.title('IHDP')
plt.xticks(ind, ('T=1', 'T=0'))
plt.yticks(np.arange(0, 100, 20))
plt.legend((p1[0], p2[0]), ('DR', 'W/o DR'))

# plt.show()
plt.draw()
plt.savefig("Plots/IHDP_BoxPlots_y_f.jpeg", dpi=220)
plt.clf()