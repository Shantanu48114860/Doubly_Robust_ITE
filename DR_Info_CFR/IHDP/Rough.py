# import matplotlib.pyplot as pyplot
# import numpy as np
#
# csv_path = "MSE/DR_Results_Out_Final.csv"
# data_X = np.loadtxt(csv_path, delimiter=",", skiprows=1)
# print(data_X.shape)
# N = 50
# diff_yf_DR_Prob_wo_DR = data_X[:, -1]
# diff_yf_DR_Prob = data_X[:, -2]
#
# print(diff_yf_DR_Prob[0])
# print(diff_yf_DR_Prob_wo_DR[0])
#
# max = max(np.max(diff_yf_DR_Prob_wo_DR),np.max(diff_yf_DR_Prob_wo_DR))
# print(max)
#
# bins1 = np.linspace(0, max, 20)
# pyplot.hist(diff_yf_DR_Prob, bins1, alpha=0.5, label="With DR", color='#B60E0E',
#             histtype="bar",
#             edgecolor='r')
# pyplot.hist(diff_yf_DR_Prob_wo_DR, bins1, alpha=0.5, label="Without DR", color='g',
#             histtype="bar",
#             edgecolor='g')
# pyplot.xlabel('Difference', fontsize=12)
# pyplot.ylabel('Frequency', fontsize=12)
# pyplot.title("IHDP")
# # pyplot.ylim(0, 100)
# pyplot.xticks(fontsize=7)
# pyplot.yticks(fontsize=7)
# pyplot.legend(loc='upper right')
# pyplot.draw()
# pyplot.savefig("Plots/IHDP_Hist_y_f.jpeg", dpi=220)
# pyplot.clf()


import matplotlib.pyplot as plt
import numpy as np

csv_path = "MSE/DR_Results_Out_Final.csv"
data_X = np.loadtxt(csv_path, delimiter=",", skiprows=1)
print(data_X.shape)
N = 50
grades_range = np.linspace(0, data_X.shape[0], data_X.shape[0])

diff_yf_DR_Prob_wo_DR = (data_X[:, -1])
diff_yf_DR_Prob = (data_X[:, -2])

colors = np.random.rand(data_X.shape[0])
# area = (100 * np.random.rand(data_X.shape[0])) ** 1100  # 0 to 15 point radii

# plt.xlim(0, 0.002, 10)
plt.xlabel('With DR', fontsize=7)
plt.ylabel('W/O DR', fontsize=7)
# plt.grid(True)
plt.scatter(diff_yf_DR_Prob, diff_yf_DR_Prob_wo_DR, colors, alpha=1)
# plt.scatter(grades_range, diff_yf_DR_Prob_wo_DR, c='red', alpha=0.5)
# plt.show()
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.draw()
plt.savefig("Plots/IHDP_scatter_y_f.jpeg", dpi=220)
plt.clf()


