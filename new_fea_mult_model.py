import basis
import os
import pickle
import numpy as np


################################################################################
#
# 多个二分类模型
#
################################################################################


def get_filter_label(df, cuw):
    """对标签进行筛选,df为标签集，cuw为提取宽"""
    filtrate = df[((df['weight'] == cuw) & (df['height'] == cuw)) &
                  (df['label'] > -1) &
                  (df['frame'] > 0)]
    return filtrate


def model_pred(list, x):
    result = np.zeros(x.shape[0])
    k = 0
    for i in x:
        y_0 = list[0].predict(i.reshape(1, -1))
        if y_0 == 1:
            y_1 = list[1].predict(i.reshape(1, -1))
            if y_1 == 0:
                y_2 = list[2].predict(i.reshape(1, -1))
                y_3 = list[3].predict(i.reshape(1, -1))
                if y_3 == 1:
                    if y_2 == 1:
                        result[k] = 2
                    elif y_2 == 0:
                        result[k] = 3
                elif y_3 == 0:
                    if y_2 == 1:
                        result[k] = 4
                    elif y_2 == 0:
                        result[k] = 5
            elif y_1 == 1:
                result[k] = 1
        k = k + 1
    return result

def make_data_one(n_v, label, data):
    """制作数据集"""
    n = label.shape[0]
    xns = np.zeros((n, n_v))  # 初始化数据集
    yns = np.zeros(n)

    k_ns, k = 0, 0

    for i in range(n):
        # k = basis.progress_bar(i, n, k)  # 进度条

        cor = label[i]
        v, xns[k_ns] = basis.get_data(cor, data, n_v)

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

    # print()  # 输出一个空行

    return xns, yns


if __name__ == '__main__':
    """
    总计需要4个分类器,每个分类器均为2分类，最终达成6分类效果,分类器分类过程与编码器内部结构一致。
    """
    # ########################基础设置区域################################ #
    qp_list = [22, 27, 32, 37]  # 视频的QP值
    frame_need = 20  # 提取CU标签的帧数，用于保存模型
    video_list = [['Campfire', 'ParkRunning3', 'Tango2', 'DaylightRoad2'],  # A-0-30
                  ['BasketballDrive', 'BQTerrace', 'Cactus', 'MarketPlace', 'RitualDance'],  # B-1-50
                  ['RaceHorsesC', 'BQMall', 'PartyScene', 'BasketballDrill'],  # C-2-50
                  ['RaceHorses', 'BQSquare', 'BlowingBubbles', 'BasketballPass'],  # D-3-50
                  ['FourPeople', 'Johnny', 'KristenAndSara']]  # E-4-50
    number_frame = 20
    RF_tree_number = 10  # 树的数量
    y = [0, 0, 0, 0]
    for index_video in [0]:  # 序列编号
        maker_name = video_list[3][index_video]  # 视频名称

        # ########################高级设置区域################################ #
        list_cu = [(32, 32), (64, 64), (128, 128), (32, 64), (64, 32), (64, 128), (128, 64)]
        n_split = 16  # 分区数目
        number_values = 2 + 1 + 3 + 1 + 22 * 3 + 2 + 3 + 1  # 选择的特征值个数

        # ########################数据读取区域################################ #
        print("当前视频为" + maker_name)
        result = np.ones((16, 7))  # 结果记录

        for qp in qp_list:
            print(str(qp) + '==================================')
            train_csv = maker_name + '/cu_' + str(qp) + '.csv'
            name_label = "./label_csv/" + maker_name + "_inter_ra_f" + str(number_frame) + "_" + str(qp) + ".csv"
            while not os.path.isfile(name_label):  # 找label文件
                number_frame = number_frame + 10
                name_label = "./label_csv/" + maker_name + "_inter_ra_f" + str(number_frame) + "_" + str(qp) + ".csv"

            # ########################数据读取区域################################ #
            count = 0
            for k in list_cu:
                label_f, data_pre = basis.count_pre(i=number_values, l_name=name_label, t=train_csv, cu=k,
                                                    frame=frame_need)

                """生成数据"""
                x_all, y_all = make_data_one(number_values, label_f, data_pre)
                x_ns, x_qt, x_mtt, y_ns, y_qt, y_hv, y_23 = basis.make_data_four(number_values, label_f, data_pre)
                list_x = [x_ns, x_qt, x_mtt, x_mtt]
                list_y = [y_ns, y_qt, y_hv, y_23]

                # """输出测试"""
                # basis.print_pro(x_ns, x_qt, x_mtt, y_ns, y_qt, y_hv, y_23)

                """训练模型"""
                for tree in [RF_tree_number]:
                    list_model = basis.train_four(x_ns, x_qt, x_mtt, y_ns, y_qt, y_hv, y_23, tree)
                    #     for i in range(4):
                    #         print("=============  " + str(i))
                    #         '''模型评价'''
                    #         if list_model[i]:
                    #             y_test = list_model[i].predict(list_x[i])
                    #             sorce = basis.model_score(list_y[i], y_test)
                    #             if qp == 22:
                    #                 result[i][count] = sorce
                    #             elif qp == 27:
                    #                 result[i + 4][count] = sorce
                    #             elif qp == 32:
                    #                 result[i + 8][count] = sorce
                    #             elif qp == 37:
                    #                 result[i + 12][count] = sorce
                    #
                    #             """模型保存"""
                    #             model_name = str(k[0]) + '_' + str(k[1]) + '_' + str(i) + '.pkl'
                    #             # save for test
                    #             path = './test_model/new-' + maker_name + '-t10-mult/' + str(qp)
                    #             if not (os.path.isdir(path)):
                    #                 os.makedirs(path)
                    #             with open(path + '/' + model_name, "wb") as f:
                    #                 pickle.dump(list_model[i], f)
                    #         else:
                    #             print("当前模型不存在")
                    # count = count + 1
                    for i in range(4):
                        if list_model[i]:
                            y[i] = list_model[i].predict(x_all)
                    for j in range(len(y[0])):
                        if y[0][j] == 1:
                            if y[1][j] == 0:
                                if y[3][j] == 1:
                                    if y[2][j] == 1:
                                        y[0][j] = 2
                                    elif y[2][j] == 0:
                                        y[0][j] = 3
                                    else:
                                        print("error")
                                elif y[3][j] == 0:
                                    if y[2][j] == 1:
                                        y[0][j] = 4
                                    elif y[2][j] == 0:
                                        y[0][j] = 5
                                    else:
                                        print("error")
                    # print(y[0])
                    # print("==>")
                    if len(y[0])>0:
                        basis.model_score(y_all, y[0])

        # """写入数据"""
        # path_data = "./txt/" + maker_name + ".txt"
        # np.savetxt(path_data, result, fmt='%.3f')
