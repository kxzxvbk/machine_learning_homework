'''
通过smoke, sex, region, bmi, children, 猜测在123哪一档 p
然后根据age 再猜属于哪一档 p
最后根据age推导当前年龄，每一档的均值k
最终结果：\sum kp
'''
import pandas as pd
import numpy as np

bmi_range = [18, 20.5, 30, 38.5, 43]
age_range = range(18, 65)

class_1 = []
age_1 = []
class_2 = []
age_2 = []
class_3 = []
age_3 = []
W1 = []
W2 = []
W3 = []
Y1 = []
Y2 = []
Y3 = []
Charges = []
count = 0


def line1(x):
    return 7000 + x * (26000 / 80)


def line2(x):
    return 35000 + x * (20000 / 80)


def linear_regression():
    from sklearn import linear_model
    import numpy as np
    global Charges
    global Y1
    global Y2
    global Y3
    global W1
    global W2
    global W3

    W1 = np.array(W1).reshape(-1, 1)
    W2 = np.array(W2).reshape(-1, 1)
    W3 = np.array(W3).reshape(-1, 1)

    Y1 = np.array(Y1).reshape(-1, 1)
    Y2 = np.array(Y2).reshape(-1, 1)
    Y3 = np.array(Y3).reshape(-1, 1)

    model = linear_model.LinearRegression(fit_intercept=True)
    model.fit(W3, Y3)
    print(model.coef_)
    print(model.intercept_)


no_smoke_p = np.array([0, 0, 0], dtype=float)
smoke_p = np.array([0, 0, 0], dtype=float)

male_p = np.array([0, 0, 0], dtype=float)
female_p = np.array([0, 0, 0], dtype=float)

region_p = np.array([
    [0, 0, 0],    # southeast
    [0, 0, 0],    # northeast
    [0, 0, 0],    # northwest
    [0, 0, 0]     # southwest
], dtype=float)

bmi_p = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
], dtype=float)

children_p = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
], dtype=float)

age_p = np.zeros([len(age_range), 3], dtype=float)


def train():
    global smoke_p
    global no_smoke_p
    global female_p
    global male_p
    global age_p
    global children_p
    global bmi_p
    global region_p

    avg0 = 0
    avg1 = 0
    avg2 = 0
    num0 = 0
    num1 = 0
    num2 = 0
    train_set = pd.read_csv('./data/train.csv')
    for i in range(train_set.shape[0]):
        age = train_set['age'][i]
        is_smoke = False if train_set['smoker'][i] == 'no' else True
        is_male = False if train_set['sex'][i] == 'female' else True
        charge_value = train_set['charges'][i]
        Charges.append(charge_value)

        if train_set['region'][i] == 'southeast':
            region_num = 0
        elif train_set['region'][i] == 'northeast':
            region_num = 1
        elif train_set['region'][i] == 'northwest':
            region_num = 2
        else:
            region_num = 3

        bmi_num = -1
        bmi_value = train_set['bmi'][i]
        for ind in range(len(bmi_range)):
            if bmi_range[ind] > bmi_value:
                bmi_num = ind
                break
        if bmi_num == -1:
            bmi_num = len(bmi_range)

        children_num = train_set['children'][i]

        split1 = line1(age)
        split2 = line2(age)

        if charge_value < split1:
            block_num = 0
            avg0 += charge_value
            num0 += 1
            class_1.append(charge_value)
            age_1.append(age)
        elif charge_value > split2:
            block_num = 2
            avg2 += charge_value
            num2 += 1
            class_3.append(charge_value)
            age_3.append(age)
        else:
            block_num = 1
            avg1 += charge_value
            num1 += 1
            class_2.append(charge_value)
            age_2.append(age)

        if is_smoke:
            smoke_p[block_num] += 1
        else:
            no_smoke_p[block_num] += 1

        if is_male:
            male_p[block_num] += 1
        else:
            female_p[block_num] += 1

        age_num = train_set['age'][i]
        age_p[age_num - age_range[0]][block_num] += 1

        region_p[region_num][block_num] += 1

        bmi_p[bmi_num][block_num] += 1
        children_p[children_num][block_num] += 1

    smoke_p /= np.sum(smoke_p)
    # print(smoke_p)
    no_smoke_p /= np.sum(no_smoke_p)
    # print(no_smoke_p)
    male_p /= np.sum(male_p)
    # print(male_p)
    female_p /= np.sum(female_p)
    # print(female_p)
    for r in region_p:
        r /= np.sum(r)
    print(region_p)
    for b in bmi_p:
        b /= np.sum(b)
    # print(bmi_p)
    for c in children_p:
        c /= np.sum(c)
    print(children_p)
    for a in age_p:
        a /= np.sum(a)
    # print(age_p)
    print(avg0 / num0)
    print(avg1 / num1)
    print(avg2 / num2)


def evaluate(is_male, is_smoker, region_num, bmi_num, children_num, age, charge_value=None):
    p = np.array([1., 1., 1.])
    weight = [11897.478016666675, 29905.052745454548, 54911.54118166668]
    k = [335.15583882, 331.21211242, 360.11455144]
    b = [-1248.07877273, 17207.9611405, 40475.01258136]
    bias = 1478.23318144
    weights = [0.98405754, -0.03936257,  0.03111483]
    weight_modify = [k[0] * age + b[0], k[1] * age + b[1], k[2] * age + b[2]]
    # if is_male:
    #    p *= male_p
    # else:
    #     p *= female_p

    if is_smoker:
        p *= smoke_p
    else:
        p *= no_smoke_p

    # p *= region_p[region_num]
    p *= bmi_p[bmi_num]
    # p *= children_p[children_num]
    # p *= age_p[age - age_range[0]]

    choice = np.argmax(p)
    #if choice == 0:
    #    W1.append(weight_modify[choice])
    #    Y1.append(charge_value)
    #elif choice == 1:
    #    W2.append(weight_modify[choice])
    #    Y2.append(charge_value)
    #elif choice == 2:
    #    W3.append(weight_modify[choice])
    #    Y3.append(charge_value)

    index = np.argmax(p)
    if index == 0:
        ans = 0.98088812 * weight_modify[index] + 1971.19251334
    elif index == 1:
        ans = 0.94379909 * weight_modify[index] + 821.41480916
    else:
        ans = 0.98832207 * weight_modify[index] + 454.12037589
    return ans


def eval_train(path):
    train_set = pd.read_csv(path)
    total_mean = np.mean(train_set['charges'])
    total_var = np.sum((train_set['charges'] - total_mean) ** 2)

    total_preds = []

    for i in range(train_set.shape[0]):
        age = train_set['age'][i]
        charge_val = train_set['charges'][i]
        is_smoke = False if train_set['smoker'][i] == 'no' else True
        is_male = False if train_set['sex'][i] == 'female' else True

        if train_set['region'][i] == 'southeast':
            region_num = 0
        elif train_set['region'][i] == 'northeast':
            region_num = 1
        elif train_set['region'][i] == 'northwest':
            region_num = 2
        else:
            region_num = 3

        bmi_num = -1
        bmi_value = train_set['bmi'][i]
        for ind in range(len(bmi_range)):
            if bmi_range[ind] > bmi_value:
                bmi_num = ind
                break
        if bmi_num == -1:
            bmi_num = len(bmi_range)

        children_num = train_set['children'][i]

        total_preds.append(evaluate(is_male, is_smoke, region_num, bmi_num, children_num, age, charge_val))

    total_preds = np.array(total_preds)
    total_loss = np.sum((total_preds - train_set['charges']) ** 2)
    print("ToTal R:  " + str(1 - total_loss / total_var))


def submit_test(path):
    train_set = pd.read_csv(path)

    total_preds = []

    for i in range(train_set.shape[0]):
        age = train_set['age'][i]
        is_smoke = False if train_set['smoker'][i] == 'no' else True
        is_male = False if train_set['sex'][i] == 'female' else True

        if train_set['region'][i] == 'southeast':
            region_num = 0
        elif train_set['region'][i] == 'northeast':
            region_num = 1
        elif train_set['region'][i] == 'northwest':
            region_num = 2
        else:
            region_num = 3

        bmi_num = -1
        bmi_value = train_set['bmi'][i]
        for ind in range(len(bmi_range)):
            if bmi_range[ind] > bmi_value:
                bmi_num = ind
                break
        if bmi_num == -1:
            bmi_num = len(bmi_range)

        children_num = train_set['children'][i]
        total_preds.append(evaluate(is_male, is_smoke, region_num, bmi_num, children_num, age))
    train_set['charges'] = np.array(total_preds)
    train_set.to_csv('submission.csv', index=False)


train()
eval_train(path='./data/train.csv')
# linear_regression()
submit_test(path='./data/test_sample.csv')

