import matplotlib.pyplot as plt
import numpy as np

csv_path = "MSE/DR_Results_Out_Final_1.csv"
data_X = np.loadtxt(csv_path, delimiter=",", skiprows=1)
print(data_X.shape)

grades_range = np.linspace(0, data_X.shape[0], data_X.shape[0])

diff_yf_DR_Prob_wo_DR = (data_X[:, -4])
diff_yf_DR_Prob = (data_X[:, -5])
T = (data_X[:, 1])

idx = (T == 1)

# diff_yf_DR_Prob_treated = diff_yf_DR_Prob[idx]
# diff_yf_DR_Prob_control = diff_yf_DR_Prob[~idx]
#
# diff_yf_DR_Prob_wo_DR_treated = diff_yf_DR_Prob_wo_DR[idx]
# diff_yf_DR_Prob_wo_DR_control = diff_yf_DR_Prob_wo_DR[~idx]

colors = np.random.rand(data_X.shape[0])
# area = (100 * np.random.rand(data_X.shape[0])) ** 1100  # 0 to 15 point radii

# plt.xlim(0, 0.002, 10)
plt.xlabel('With DR', fontsize=7)
plt.ylabel('W/O DR', fontsize=7)
# plt.grid(True)
# arr1 = plt.scatter(diff_yf_DR_Prob_treated, diff_yf_DR_Prob_wo_DR_treated, c='#CCEB0A', alpha=1)
arr2 = plt.scatter(diff_yf_DR_Prob, diff_yf_DR_Prob_wo_DR, c='#F0B27A', alpha=1)
# plt.scatter(grades_range, diff_yf_DR_Prob_wo_DR, c='red', alpha=0.5)
# plt.show()


plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
# plt.legend(loc='upper right')
# plt.legend([arr1, arr2], ['treated', 'control'])
plt.draw()
plt.savefig("Plots/Twins_scatter_y_f.jpeg", dpi=220)
plt.clf()

import matplotlib.pyplot as pyplot
import numpy as np


def plot_hist():
    csv_path = "MSE/DR_Results_Out_Final_1.csv"
    data_X = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    print(data_X.shape)
    diff_yf_DR_Prob_wo_DR = data_X[:, -1]
    diff_yf_DR_Prob = data_X[:, -2]

    _max = max(np.max(diff_yf_DR_Prob_wo_DR), np.max(diff_yf_DR_Prob_wo_DR))
    print(_max)

    print(diff_yf_DR_Prob[0])
    print(diff_yf_DR_Prob_wo_DR[0])

    bins1 = np.linspace(0, _max, 50)
    diff_yf_DR_Prob_wo_DR = (data_X[:, -4])
    diff_yf_DR_Prob = (data_X[:, -5])
    T = (data_X[:, 1])

    idx = (T == 1)

    diff_yf_DR_Prob_treated = diff_yf_DR_Prob[idx]
    diff_yf_DR_Prob_control = diff_yf_DR_Prob[~idx]

    diff_yf_DR_Prob_wo_DR_treated = diff_yf_DR_Prob_wo_DR[idx]
    diff_yf_DR_Prob_wo_DR_control = diff_yf_DR_Prob_wo_DR[~idx]

    # pyplot.hist(diff_yf_DR_Prob_treated, bins1, alpha=0.5, label="With DR", color='#B60E0E',
    #             histtype="bar",
    #             edgecolor='r')
    # pyplot.hist(diff_yf_DR_Prob_wo_DR_treated, bins1, alpha=0.5, label="Without DR", color='g',
    #             histtype="bar",
    #             edgecolor='g')

    pyplot.hist(diff_yf_DR_Prob, bins1, alpha=0.5, label="With DR", color='#B60E0E',
                histtype="bar",
                edgecolor='r')
    pyplot.hist(diff_yf_DR_Prob_wo_DR, bins1, alpha=0.5, label="Without DR", color='g',
                histtype="bar",
                edgecolor='g')

    pyplot.xlabel('Difference', fontsize=12)
    pyplot.ylabel('Frequency', fontsize=12)
    pyplot.title("Twins")
    # pyplot.ylim(0, 100)
    pyplot.xticks(fontsize=7)
    pyplot.yticks(fontsize=7)
    pyplot.legend(loc='upper right')
    pyplot.draw()
    pyplot.savefig("Plots/Twins_Hist_y_f.jpeg", dpi=220)
    pyplot.clf()


plot_hist()

# import matplotlib.pyplot as plt
# girls_grades = [89, 90, 70, 89, 100, 80, 90, 100, 80, 34]
# boys_grades = [30, 29, 49, 48, 100, 48, 38, 45, 20, 30]
# grades_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# fig=plt.figure()
# ax=fig.add_axes([0,0,1,1])
# ax.scatter(grades_range, girls_grades, color='r')
# ax.scatter(grades_range, boys_grades, color='b')
# ax.set_xlabel('Grades Range')
# ax.set_ylabel('Grades Scored')
# ax.set_title('scatter plot')
# plt.show()
