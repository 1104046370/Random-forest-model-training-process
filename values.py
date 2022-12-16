import numpy as np
import cv2


################################################################################
#
# 以下函数作为数据处理函数，适用领域为 np数组数据处理
#
################################################################################


def get_train_x_avg(pixel, side):
    """对输入数据进行均值化处理"""
    step0 = pixel.shape[0] // side  # 均值化的步长
    step1 = pixel.shape[1] // side
    result = np.zeros((side, side))
    for i in range(side):
        for j in range(side):
            average = pixel[i * step0:(i + 1) * step0, j * step1:(j + 1) * step1].sum() / step0 / step1
            result[i, j] = average
    return result


def get_variance_old_old(a):
    """获取6种不同分割模式下的方差大小"""
    side = a.shape[0]
    avg = a.sum() / side / side  # 平均值

    # 计算四叉树情况下的方差
    qt = np.zeros((2, 2))
    k_qt = side // 2
    qt[0, 0] = a[0:k_qt, 0:k_qt].sum() / k_qt / k_qt
    qt[0, 1] = a[0:k_qt, k_qt:2 * k_qt].sum() / k_qt / k_qt
    qt[1, 0] = a[k_qt:2 * k_qt, 0:k_qt].sum() / k_qt / k_qt
    qt[1, 1] = a[k_qt:2 * k_qt, k_qt:2 * k_qt].sum() / k_qt / k_qt
    variance_qt = ((qt - avg) * (qt - avg)).sum() / 4 / 4

    # 计算横向二叉树情况下的方差
    tmp0 = (qt[0, 0] + qt[0, 1]) / 2
    tmp1 = (qt[1, 0] + qt[1, 1]) / 2
    variance_bth = ((tmp0 - avg) * (tmp0 - avg) + (tmp1 - avg) * (tmp1 - avg)) / 2 / 2

    # 计算纵向二叉树情况下的方差
    tmp0 = (qt[0, 0] + qt[1, 0]) / 2
    tmp1 = (qt[0, 1] + qt[1, 1]) / 2
    variance_btv = ((tmp0 - avg) * (tmp0 - avg) + (tmp1 - avg) * (tmp1 - avg)) / 2 / 2

    # 计算横向三叉树情况下的方差
    k_tt = side // 4
    tt = np.zeros(3)
    tt[0] = a[0:k_tt, :].sum() / k_tt / side
    tt[1] = a[k_tt:3 * k_tt, :].sum() / k_tt / side / 2
    tt[2] = a[3 * k_tt:4 * k_tt, :].sum() / k_tt / side
    variance_tth = ((tt - avg) * (tt - avg)).sum() / 3 / 3

    # 计算纵向三叉树情况下的方差
    tt[0] = a[:, 0:k_tt].sum() / k_tt / side
    tt[1] = a[:, k_tt:3 * k_tt].sum() / k_tt / side / 2
    tt[2] = a[:, 3 * k_tt:4 * k_tt].sum() / k_tt / side
    variance_ttv = ((tt - avg) * (tt - avg)).sum() / 3 / 3

    result = np.array([avg, variance_qt, variance_bth, variance_btv, variance_tth, variance_ttv])
    return result


def get_variance_old(a):
    ns = __get_variance(a)

    qt = (__get_variance(a[0:a.shape[0] // 2, 0:a.shape[1] // 2]) +
          __get_variance(a[0:a.shape[0] // 2, a.shape[1] // 2:a.shape[1]]) +
          __get_variance(a[a.shape[0] // 2:a.shape[0], 0:a.shape[1] // 2]) +
          __get_variance(a[a.shape[0] // 2:a.shape[0], a.shape[1] // 2:a.shape[1]])) / 16

    bth = (__get_variance(a[0:a.shape[0] // 2, :]) +
           __get_variance(a[a.shape[0] // 2:a.shape[0], :])) / 4
    btv = (__get_variance(a[:, 0:a.shape[0] // 2]) +
           __get_variance(a[:, a.shape[0] // 2: a.shape[0]])) / 4

    tth = (__get_variance(a[0:a.shape[0] // 4, :]) +
           __get_variance(a[a.shape[0] // 4:(a.shape[0] // 4) * 3, :]) +
           __get_variance(a[(a.shape[0] // 4) * 3: a.shape[0], :])) / 9
    ttv = (__get_variance(a[:, 0:a.shape[0] // 4]) +
           __get_variance(a[:, a.shape[0] // 4:(a.shape[0] // 4) * 3]) +
           __get_variance(a[:, (a.shape[0] // 4) * 3: a.shape[0]])) / 9

    return ns, qt, bth, btv, tth, ttv


def get_variance(a):
    ns = __get_variance(a)

    qt_0 = __get_variance(a[0:a.shape[0] // 2, 0:a.shape[1] // 2])
    qt_1 = __get_variance(a[0:a.shape[0] // 2, a.shape[1] // 2:a.shape[1]])
    qt_2 = __get_variance(a[a.shape[0] // 2:a.shape[0], 0:a.shape[1] // 2])
    qt_3 = __get_variance(a[a.shape[0] // 2:a.shape[0], a.shape[1] // 2:a.shape[1]])

    bth_0 = __get_variance(a[0:a.shape[0] // 2, :])
    bth_1 = __get_variance(a[a.shape[0] // 2:a.shape[0], :])

    btv_0 = __get_variance(a[:, 0:a.shape[1] // 2])
    btv_1 = __get_variance(a[:, a.shape[1] // 2: a.shape[1]])

    tth_0 = __get_variance(a[0:a.shape[0] // 4, :])
    tth_1 = __get_variance(a[a.shape[0] // 4:(a.shape[0] // 4) * 3, :])
    tth_2 = __get_variance(a[(a.shape[0] // 4) * 3: a.shape[0], :])
    ttv_0 = __get_variance(a[:, 0:a.shape[1] // 4])
    ttv_1 = __get_variance(a[:, a.shape[1] // 4:(a.shape[1] // 4) * 3])
    ttv_2 = __get_variance(a[:, (a.shape[1] // 4) * 3: a.shape[1]])

    return ns, qt_0, qt_1, qt_2, qt_3, bth_0, bth_1, btv_0, btv_1, tth_0, tth_1, tth_2, ttv_0, ttv_1, ttv_2


def __get_variance(a):
    side01 = a.shape[0]
    side02 = a.shape[1]
    avg = a.sum() / side01 / side02  # 平均值
    variance = ((a - avg) * (a - avg)).sum() / side01 / side02

    return variance


def get_pixel_gradient(array):
    """获取对应区域图像的梯度值"""
    sobel_x = cv2.Sobel(array, cv2.CV_64F, dx=1, dy=0)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.Sobel(array, cv2.CV_64F, dx=0, dy=1)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    result = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    return result


def get_num_color(array, x=(0.0, 1.0), y=(0.0, 1.0)):
    """计算对应图像区域的颜色数"""
    a1 = int(x[0] * array.shape[0])
    a2 = int(x[1] * array.shape[0])
    b1 = int(y[0] * array.shape[1])
    b2 = int(y[1] * array.shape[1])
    color_list = np.unique(array[a1:a2, b1:b2])
    return len(color_list)


def get_mean(a):
    ns = __get_mean(a)

    qt_0 = __get_mean(a[0:a.shape[0] // 2, 0:a.shape[1] // 2])
    qt_1 = __get_mean(a[0:a.shape[0] // 2, a.shape[1] // 2:a.shape[1]])
    qt_2 = __get_mean(a[a.shape[0] // 2:a.shape[0], 0:a.shape[1] // 2])
    qt_3 = __get_mean(a[a.shape[0] // 2:a.shape[0], a.shape[1] // 2:a.shape[1]])

    bth_0 = __get_mean(a[0:a.shape[0] // 2, :])
    bth_1 = __get_mean(a[a.shape[0] // 2:a.shape[0], :])

    btv_0 = __get_mean(a[:, 0:a.shape[0] // 2])
    btv_1 = __get_mean(a[:, a.shape[0] // 2: a.shape[0]])

    tth_0 = __get_mean(a[0:a.shape[0] // 4, :])
    tth_1 = __get_mean(a[a.shape[0] // 4:(a.shape[0] // 4) * 3, :])
    tth_2 = __get_mean(a[(a.shape[0] // 4) * 3: a.shape[0], :])
    ttv_0 = __get_mean(a[:, 0:a.shape[0] // 4])
    ttv_1 = __get_mean(a[:, a.shape[0] // 4:(a.shape[0] // 4) * 3])
    ttv_2 = __get_mean(a[:, (a.shape[0] // 4) * 3: a.shape[0]])

    return ns, qt_0, qt_1, qt_2, qt_3, bth_0, bth_1, btv_0, btv_1, tth_0, tth_1, tth_2, ttv_0, ttv_1, ttv_2


def __get_mean(a):
    side01 = a.shape[0]
    side02 = a.shape[1]
    avg = a.sum() / side01 / side02  # 平均值
    return avg


def get_num_color_diff(pixel):
    num_all = get_num_color(pixel)
    diff_qt = (get_num_color(pixel, x=(0, 0.5), y=(0, 0.5)) +
               get_num_color(pixel, x=(0, 0.5), y=(0, 0.5)) +
               get_num_color(pixel, x=(0, 0.5), y=(0, 0.5)) +
               get_num_color(pixel, x=(0, 0.5), y=(0, 0.5)) - 4 * num_all) / 4
    diff_bth = (get_num_color(pixel, x=(0, 0.5)) + get_num_color(pixel, x=(0.5, 1)) - 2 * num_all) / 2
    diff_btv = (get_num_color(pixel, y=(0, 0.5)) + get_num_color(pixel, y=(0.5, 1)) - 2 * num_all) / 2
    diff_tth = (get_num_color(pixel, x=(0, 0.25)) +
                get_num_color(pixel, x=(0.25, 0.75)) +
                get_num_color(pixel, x=(0.75, 1)) - 3 * num_all) / 3
    diff_ttv = (get_num_color(pixel, y=(0, 0.25)) +
                get_num_color(pixel, y=(0.25, 0.75)) +
                get_num_color(pixel, y=(0.75, 1)) - 3 * num_all) / 3
    return diff_qt, diff_bth, diff_btv, diff_tth, diff_ttv, num_all


def get_entropy_different(array):
    """计算划分前后的熵的变化情况"""
    entropy_all = get_entropy(array)
    entropy_qt = (get_entropy(array, (0, 0.5), (0, 0.5)) +
                  get_entropy(array, (0.5, 1), (0, 0.5)) +
                  get_entropy(array, (0, 0.5), (0.5, 1)) +
                  get_entropy(array, (0.5, 1), (0.5, 1))) / 4
    diff_qt = entropy_qt - entropy_all

    return entropy_all, entropy_qt, diff_qt


def get_entropy(array, x=(0.0, 1.0), y=(0.0, 1.0)):
    """获取对应图像区域的熵"""
    tmp = np.zeros(256)
    k = 0
    res = 0
    for i in range(int(array.shape[0] * x[0]), int(array.shape[0] * x[1])):
        for j in range(int(array.shape[1] * y[0]), int(array.shape[1] * y[1])):
            val = array[i][j]
            tmp[val] = tmp[val] + 1
            k = k + 1
    for i in range(256):
        tmp[i] = float(tmp[i] / k)
    for i in range(256):
        res = res + tmp[i] * tmp[i]

    res = 1 - res
    return res


def get_neighbor_diff(a):
    side01 = a.shape[0]
    side02 = a.shape[1]
    su_h = 0
    su_v = 0
    for i in range(side01):
        for j in range(side02 - 1):
            su_h += abs(a[i][j] - a[i][j + 1])

    for i in range(side02 - 1):
        for j in range(side01):
            su_v += abs(a[i][j] - a[i + 1][j])

    su_h = su_h / side01 / side02
    su_v = su_v / side01 / side02

    su_left = 0
    su_right = 0
    for i in range(side01 - 1):
        su_left += abs(a[i][i] - a[i + 1][i + 1])
        su_right += abs(a[i][side01 - i - 1] - a[i + 1][side01 - i - 2])
    su_left /= side01
    su_right /= side01
    return su_h, su_v, su_left, su_right


def get_max_diff_min(a):
    k = a.ravel()
    return max(k), min(k), max(k) - min(k), sum(k) / len(k)
