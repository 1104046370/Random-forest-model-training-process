import basis
import os
import pickle
import pandas as pd
import numpy as np
import values
from sklearn import tree


# def get_data(cor, d, n_v):
#     """标准数据集获取函数"""
#     tmp = d[(d[0] == cor[1]) & (d[1] == cor[2]) & (d[2] == cor[3]) & (d[3] == cor[4]) & (d[4] == cor[5])]
#
#     if tmp.size == 0:
#         result = np.zeros(n_v)
#         return True, result
#     else:
#         result = tmp.values[0][3:]
#
#         return False, result


def get_filter(df, cu, frame):
    """对标签进行筛选,df为标签集，cuw为提取宽"""
    filtrate = df[(df['label'] > -1) & (df['frame'] > 0) & (df['frame'] < frame)]
    return filtrate


def get_filter_2(df, cu, frame):
    """对标签进行筛选,df为标签集，cuw为提取宽"""
    filtrate = df[((df['weight'] == cu[0]) & (df['height'] == cu[1])) &
                  (df['label'] > -1) &
                  (df['frame'] > 0)]
    return filtrate


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
    总计需要1个分类器,该分类器为6分类,分类器使用编码器中的过程数据进行分类。
    """
    # ########################基础设置区域################################ #
    qp_list = [22, 27, 32, 37]  # 视频的QP值
    frame_need = 20  # 提取CU标签的帧数，用于保存模型
    video_list = [['Campfire', 'ParkRunning3', 'Tango2', 'DaylightRoad2'],  # A-0-30
                  ['BasketballDrive', 'BQTerrace', 'Cactus', 'MarketPlace', 'RitualDance'],  # B-1
                  ['RaceHorsesC', 'BQMall', 'PartyScene', 'BasketballDrill'],  # C-2
                  ['RaceHorses', 'BQSquare', 'BlowingBubbles', 'BasketballPass'],  # D-3
                  ['FourPeople', 'Johnny', 'KristenAndSara']]  # E-4
    maker_name = video_list[1][0]  # 视频名称

    # ########################高级设置区域################################ #
    list_cu = [(32, 32), (64, 64), (128, 128), (32, 64), (64, 32), (64, 128), (128, 64)]
    n_split = 16  # 分区数目
    number_values = 2 + 1 + 3 + 1 + 22 * 3 + 2 + 3 + 1  # 选择的特征值个数
    number_frame = 20

    # ########################数据读取区域################################ #
    print("当前视频为" + maker_name)
    for qp in qp_list:
        train_csv = maker_name + '/cu_' + str(qp) + '.csv'
        name_label = "./label_csv/" + maker_name + "_inter_ra_f" + str(number_frame) + "_" + str(qp) + ".csv"
        while not os.path.isfile(name_label):
            number_frame = number_frame + 10
            name_label = "./label_csv/" + maker_name + "_inter_ra_f" + str(number_frame) + "_" + str(qp) + ".csv"
        for k in list_cu:
            label_f, data_pre = basis.count_pre(i=number_values, l_name=name_label, t=train_csv, cu=k, frame=frame_need,
                                                function=get_filter_2)  # 可修改提取条件，增加提取函数即可

            """生成数据集"""
            x_ns, y_ns = make_data_one(number_values, label_f, data_pre)

            # # ######################## new add ################
            clf_tree = tree.DecisionTreeClassifier()
            if x_ns.shape[0] == 0:
                print("no model")
                continue
            clf_tree.fit(x_ns, y_ns)
            for fea in clf_tree.feature_importances_:
                print(fea,end=',')
            x_ns = x_ns * clf_tree.feature_importances_
            print()

        #     for i_tree in [10]:
        #         """训练模型"""
        #         m = basis.train_random_forest(x_ns, y_ns, tree=i_tree)
        #         if m:
        #             '''模型评价'''
        #             y_test = m.predict(x_ns)
        #             basis.model_score(y_ns, y_test)
        #
        #             """模型保存"""
        #
        #             model_name = str(k[0]) + '_' + str(k[1]) + '.pkl'
        #
        #             # save for test
        #             path = './test_model/new-' + maker_name + '-t10/' + str(qp)
        #             if not (os.path.isdir(path)):
        #                 os.makedirs(path)
        #             with open(path + '/' + model_name, "wb") as f:
        #                 pickle.dump(m, f)
        # print('===' + str(qp))
