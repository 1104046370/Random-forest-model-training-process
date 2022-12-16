from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import time
import os
import values
import pandas as pd


################################################################################
#
# 以下函数作为基础通用函数，适用领域为 机器学习 & 视频编码
#
################################################################################
def yuv_import(filename, dims, num_frm):
    """输入图像"""
    num_frm = num_frm + 1  # 帧数从0帧开始计数
    fp = open(filename, 'rb')

    u = dims[0] // 2
    v = dims[1] // 2
    yt = np.zeros((num_frm, dims[0], dims[1]), np.uint8, 'C')

    for i in range(num_frm):
        for m in range(dims[0]):
            for n in range(dims[1]):
                yt[i, m, n] = ord(fp.read(1))
        fp.seek(u * v * 2, 1)

    fp.close()
    return yt


def get_pixel(data, frame, x, y, weight, height):
    """获取对应图像区域的像素值"""
    pixel_need_encode = data[frame][y:y + height, x:x + weight]

    return pixel_need_encode


def make_data_four(n_values, label, data):
    """制作数据集"""
    n = label.shape[0]
    xns, xqt, xmt = np.zeros((n, n_values)), np.zeros((n, n_values)), np.zeros((n, n_values))  # 初始化数据集

    yns, yqt, yhv, y23 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    k_qt, k_mt, k_ns, k = 0, 0, 0, 0  # 初始化指针

    for i in range(n):
        # k = progress_bar(i, n, k)  # 进度条

        cor = label[i]
        v, xns[k_ns] = v, xmt[k_mt] = v, xqt[k_qt] = get_data(cor, data, n_values)

        if v:  # 错误情况删除该条数据
            ll = 0
        else:
            ll = cor[6]
        if ll > 0:
            yns[k_ns] = 1
            if ll == 1:
                yqt[k_qt] = 1
                xmt = np.delete(xmt, -1, axis=0)
                yhv = np.delete(yhv, -1, axis=0)
                y23 = np.delete(y23, -1, axis=0)
                k_mt = k_mt - 1
            else:
                if ll == 2 or ll == 4:
                    yhv[k_mt] = 1
                    if ll == 2:
                        y23[k_mt] = 1
                else:
                    if ll == 3:
                        y23[k_mt] = 1
        else:
            xqt = np.delete(xqt, -1, axis=0)
            xmt = np.delete(xmt, -1, axis=0)
            yqt = np.delete(yqt, -1, axis=0)
            yhv = np.delete(yhv, -1, axis=0)
            y23 = np.delete(y23, -1, axis=0)
            k_qt = k_qt - 1
            k_mt = k_mt - 1

            if v:
                xns = np.delete(xns, -1, axis=0)
                yns = np.delete(yns, -1, axis=0)
                k_ns = k_ns - 1

        k_qt = k_qt + 1
        k_mt = k_mt + 1
        k_ns = k_ns + 1  # 指针前进

    print()  # 输出一个空行

    return xns, xqt, xmt, yns, yqt, yhv, y23


def make_data_one(n_v, cu, label, data):
    """制作数据集"""
    n = label.shape[0]
    xns = np.zeros((n, n_v))  # 初始化数据集
    yns = np.zeros(n)

    k_ns, k = 0, 0

    for i in range(n):
        k = progress_bar(i, n, k)  # 进度条

        cor = label[i]
        v, xns[k_ns] = get_data(cor, data, cu, n_v)

        if v:
            ll = 0  # 错误数据
        else:
            ll = cor[6]

        yns[k_ns] = ll

        if v:
            xns = np.delete(xns, -1, axis=0)
            yns = np.delete(yns, -1, axis=0)
            k_ns = k_ns - 1

        k_ns = k_ns + 1  # 指针前进

    print()  # 输出一个空行

    return xns, yns


def get_proportion(array):
    """计算标签集合中的1/0的比例数"""
    if len(array) > 0:
        result = sum(array) / len(array)
        return result, sum(array), len(array) - sum(array)
    else:
        return 0, 0, 0


def train_random_forest(x, y, tree=10):
    """训练随机森林"""
    model = RandomForestClassifier(n_estimators=tree, min_samples_split=3)
    if x.shape[0] <= 2:
        print('no model train')
        return 0
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        # # get_model_score(y_pred, y_test)
    return model


def evaluate_forest(m0, m1, m2, m3, x, y, mode=0):
    """评估训练好的模型在整体数据集上的表现"""
    if mode == 0:
        y0 = m0.predict(x)
        y1 = m1.predict(x)
        y2 = m2.predict(x)
        y3 = m3.predict(x)
        i = 0
        y_test = []
        ly = len(y0)
        while i < ly:
            if y0[i]:
                if y1[i]:
                    y_test.append(1)
                else:
                    if y3[i]:
                        if y2[i]:
                            y_test.append(2)
                        else:
                            y_test.append(3)
                    else:
                        if y2[i]:
                            y_test.append(4)
                        else:
                            y_test.append(5)
            else:
                y_test.append(0)
            i += 1
        model_score(y, y_test)


def model_score(y, y_test):
    """准确率"""
    ly = y.shape[0]
    i = 0
    score = [0, 0, 0, 0, 0, 0]
    fail = [0, 0, 0, 0, 0, 0]
    while i < ly:
        if y_test[i] == y[i]:
            score[int(y[i])] += 1
        else:
            fail[int(y[i])] += 1
        i += 1
    # print('6种类别的分布情况如下：')
    # print('True:', score)
    # print('False:', fail)
    # print('Total:',
    #       [score[0] + fail[0], score[1] + fail[1], score[2] + fail[2], score[3] + fail[3], score[4] + fail[4],
    #        score[5] + fail[5]])

    # print('6种类别的准确率为：', end=' ')
    # for i in range(6):
    #     if score[i] + fail[i] == 0:
    #         print('none,', end=' ')
    #     else:
    #         result = score[i] / (score[i] + fail[i])
    #         print('%.3f,' % result, end='')
    result = sum(score) / (sum(score) + sum(fail))
    # print('综合准确率为：%.3f' % result)
    print('%.3f' % result, end=' ')
    return result


def get_model_score(y_score, y_test):
    """准确率"""
    a = np.logical_xor(y_score, y_test)
    k_z = (y_test.shape[0] - a.sum(axis=0)) / y_test.shape[0] * 100
    """灵敏度"""
    b = np.logical_and(y_score, y_test)
    de = y_test.sum(axis=0)
    if de == 0:
        de = de + 1
    k_l = (b.sum(axis=0)) / de * 100
    """特异度"""
    c = np.logical_or(y_score, y_test)
    z = y_test.shape[0]
    df = z - y_test.sum(axis=0)
    if df == 0:
        df = df + 1
    k_t = (z - c.sum(axis=0)) / df * 100
    print("该模型的预测准确、灵敏、特异为：%.3f%%、%.3f%%、%.3f%%" % (k_z, k_l, k_t))


def print_pro(x1, x2, x3, y1, y2, y3, y4):
    """计算训练数据的比例"""
    print(x1.shape, y1.shape,
          x2.shape, y2.shape,
          x3.shape, y3.shape,
          x3.shape, y4.shape)
    print(get_proportion(y1),
          get_proportion(y2),
          get_proportion(y3),
          get_proportion(y4))


def train_four(x1, x2, x3, y1, y2, y3, y4, tree):
    """训练模型"""
    model_ns = train_random_forest(x1, y1, tree)
    # print('========================')
    model_qt = train_random_forest(x2, y2, tree)
    # print('========================')
    model_hv = train_random_forest(x3, y3, tree)
    # print('========================')
    model_23 = train_random_forest(x3, y4, tree)
    return [model_ns, model_qt, model_hv, model_23]


def save_model(list_model, list_name, name):
    y = time.localtime(time.time())
    path = "./test_model/" + name + "_%d_%d_%d" % (y.tm_year, y.tm_mon, y.tm_mday)
    if not (os.path.isdir(path)):
        os.makedirs(path)
    for i in range(4):
        joblib.dump(list_model[i], path + "/" + list_name[i])


def save_model_single(model, model_name, name):
    y = time.localtime(time.time())
    path = "./test_model/" + name + "_%d_%d_%d" % (y.tm_year, y.tm_mon, y.tm_mday)
    if not (os.path.isdir(path)):
        os.makedirs(path)
    joblib.dump(model, path + "/" + model_name)


def progress_bar(i, length, k):
    print('\r' + '[' + '=' * k + ' ' * (30 - k) + ']' + "%.1f %%" % (i / length * 100), end='')
    # if (length // 30 == 0):
    #     length += 1
    if (i + 1) % (length // 30) == 0:
        k = k + 1
    return k


def get_pixel_diff(data, frame, x, y, weight, height):
    pixel01 = get_pixel(data, frame, x, y, weight, height)
    pixel02 = get_pixel(data, frame - 1, x, y, weight, height)
    pixel = np.where(pixel01 > pixel02, pixel01 - pixel02, pixel02 - pixel01)
    return pixel


def get_data(cor, d, n_v):
    """标准数据集获取函数"""
    tmp = d[(d[0] == cor[1]) & (d[1] == cor[2]) & (d[2] == cor[3]) & (d[3] == cor[4]) & (d[4] == cor[5])]

    if tmp.size == 0:
        result = np.zeros(n_v)
        return True, result
    else:
        result = tmp.values[0][3:]

        return False, result


def get_train_x_p(pixel):
    """标准训练集数据"""
    pixel_avg = values.get_train_x_avg(pixel, 16)  # 均值化处理
    a1 = values.get_variance(pixel)  # 方差
    a2 = values.get_neighbor_diff(pixel_avg)  # 邻差
    a3 = values.get_max_diff_min(pixel_avg)  # 极值
    result = np.concatenate((a1, a2, a3))
    return result


def get_label_data(df, cu):
    """对标签进行筛选,df为标签集，cu为提取宽"""
    filtrate = df[((df['weight'] == cu[0]) & (df['height'] == cu[1])) &
                  (df['label'] > -1) &
                  (df['frame'] > 0)]
    return filtrate


# def count_pre(i, l_name, t, function=get_label_data):
#     """标准预处理函数，用于提取对应cu的数据以及标签，可对标签筛选进行修正"""
#     table = {(16, 16): 0, (32, 32): 1, (64, 64): 2, (128, 128): 3,
#              (32, 64): 4, (64, 32): 5, (64, 128): 6, (128, 64): 7}  # 不同CU对应的编号
#     order = table[i]
#
#     label_f = function(pd.read_csv(l_name), i).values
#     data_all = [pd.read_csv('./train_csv/' + t[order], names=range(i[1] * i[0] + 5))]  # 读取对应的文件
#
#     data_pre = data_all[0].drop_duplicates(subset=[0, 1, 2, 3, 4])  # 去除重复的数据项
#
#     return label_f, data_pre

def get_filter(df, cu, frame):
    """对标签进行筛选,df为标签集，cuw为提取宽"""
    filtrate = df[((df['weight'] == cu[0]) & (df['height'] == cu[1])) &
                  (df['label'] > -1) &
                  (df['frame'] > 0)]
    return filtrate


def count_pre(i, l_name, t, cu=None, frame=None, function=get_filter):
    """标准预处理函数，用于提取对应cu的数据以及标签，可对标签筛选进行修正"""

    label = function(pd.read_csv(l_name), cu, frame).values
    data_all = [pd.read_csv('./test/' + t, names=range(i + 3))]  # 读取对应的文件

    data = data_all[0].drop_duplicates(subset=[0, 1, 2, 3, 4])  # 去除重复的数据项
    if cu:
        data = data[(data[0] > 0) & (data[3] == cu[0]) & (data[4] == cu[1])]

    return label, data
