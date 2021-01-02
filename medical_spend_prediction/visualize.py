import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_data(charges, box_num, name):
    a = charges
    a_min = int(np.min(a))
    a_max = int(np.max(a))
    bin_len = (a_max - a_min) // box_num
    plt.hist(a, bins=range(a_min, a_max+1, bin_len))
    plt.title(name)
    plt.show()


def div_by(class_name, options):
    class1 = []
    class2 = []
    for i in range(train_set.shape[0]):
        if train_set[class_name][i] == options[0]:
            class1.append(train_set['charges'][i])
        else:
            class2.append(train_set['charges'][i])
    return class1, class2


train_set = pd.read_csv('./data/train.csv')
plot_data(train_set['charges'], 100, 'total')

plt.scatter(train_set['bmi'], train_set['charges'])
plt.plot([18, 18], [0, 80000])
plt.plot([20.5, 20.5], [0, 80000])
plt.plot([30, 30], [0, 80000])
plt.plot([38.5, 38.5], [0, 80000])
plt.plot([43, 43], [0, 80000])
plt.title('bmi')
plt.show()

plt.scatter(train_set['children'], train_set['charges'])
plt.title('children')
plt.show()

plt.scatter(train_set['age'], train_set['charges'])
# plt.plot([0, 80], [7000, 33000])
# plt.plot([0, 80], [35000, 55000])

# plt.plot([0, 80], [11897.478016666675, 11897.478016666675])
# plt.plot([0, 80], [29905.052745454548, 29905.052745454548])
# plt.plot([0, 80], [54711.54118166668, 54711.54118166668])

# plt.plot([40, 40], [0, 80000])

weight = [11897.478016666675, 29905.052745454548, 54911.54118166668]
k = [335.15583882, 331.21211242, 360.11455144]
b = [-1248.07877273, 17207.9611405, 40475.01258136]
plt.plot([0, 80], [b[0], 80 * k[0] + b[0]])
plt.plot([0, 80], [b[1], 80 * k[1] + b[1]])
plt.plot([0, 80], [b[2], 80 * k[2] + b[2]])

plt.title('age')
plt.show()
print(np.min(train_set['age']))
print(np.max(train_set['age']))
plt.scatter(train_set['region'], train_set['charges'])
plt.title('region')
plt.show()

no_smoke, smoke = div_by('smoker', ['no', 'yes'])
plot_data(no_smoke, 100, 'no smoke')
plot_data(smoke, 100, 'smoke')
male, female = div_by('sex', ['male', 'female'])
#plot_data(male, 100, 'male')
#plot_data(female, 100, 'female')

line1 = 20000
line2 = 40000


def cal_e0():
    # entropy for no splitting
    total_num = train_set.shape[0]
    b1_num = 0
    b2_num = 0
    b3_num = 0
    for i in range(total_num):
        if train_set['charges'][i] < line1:
            b1_num += 1
        elif train_set['charges'][i] > line2:
            b3_num += 1
        else:
            b2_num += 1
    p1 = b1_num / total_num
    p2 = b2_num / total_num
    p3 = b3_num / total_num
    return -(p1 * np.log(p1) + p2 * np.log(p2) + p3 * np.log(p3))


def cal_e_sex():
    total_num = train_set.shape[0]
    b1_num = 0
    b2_num = 0
    b3_num = 0
    b4_num = 0  # female
    b5_num = 0
    b6_num = 0
    for i in range(total_num):
        if train_set['charges'][i] < line1 and train_set['sex'][i] == 'male':
            b1_num += 1
        elif train_set['charges'][i] > line2 and train_set['sex'][i] == 'male':
            b3_num += 1
        elif train_set['sex'][i] == 'male':
            b2_num += 1
        elif train_set['charges'][i] < line1 and train_set['sex'][i] == 'female':
            b4_num += 1
        elif train_set['charges'][i] > line2 and train_set['sex'][i] == 'female':
            b5_num += 1
        elif train_set['sex'][i] == 'female':
            b6_num += 1
    p1 = b1_num / total_num
    p2 = b2_num / total_num
    p3 = b3_num / total_num
    p4 = b4_num / total_num
    p5 = b5_num / total_num
    p6 = b6_num / total_num
    return -(p1 * np.log(p1) + p2 * np.log(p2) + p3 * np.log(p3) + p4 * np.log(p4) + p5 * np.log(p5) + p6 * np.log(p6))


print('E0: ' + str(cal_e0()))
print('ESEX: ' + str(cal_e_sex()))
