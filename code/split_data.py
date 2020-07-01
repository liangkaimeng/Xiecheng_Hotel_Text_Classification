#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : lkm

import Xiecheng_Hotel_Text_Classification.code.PrepareData as prd
import Xiecheng_Hotel_Text_Classification.code.balanced_Sampling as bs
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # tf-idf转换器
import numpy as np
from sklearn.feature_selection import f_classif, SelectKBest

data_file = r"D:\LKM\Deep_Learning\Xiecheng_Hotel_Text_Classification\data\ChnSentiCorp_htl_all.csv"

def split_data_sample():
    Class = prd.ClearData_Class()
    data = Class.GetData(data_file)
    data = Class.Remove_Impurity(data, 'review')
    data = bs.get_balanced_words(6000, data, 'label')
    x_train, x_test, y_train, y_test = train_test_split(data['review'], data['label'],
                                                        random_state=1, shuffle=True)
    tf_idf = TfidfVectorizer()
    x_trian_vec = tf_idf.fit_transform(x_train)  # 将训练集文本转换为向量
    x_test_vec = tf_idf.transform(x_test)  # 将测试集文本转换为向量
    x_trian_vec = x_trian_vec.astype(np.float64)  # 转换为numpy的float类型，便于sklearn处理
    x_test_vec = x_test_vec.astype(np.float64)

    selector = SelectKBest(f_classif, k=min(500, x_trian_vec.shape[1]))  # 选择器
    selector.fit(x_trian_vec, y_train)  # 训练
    selected_x_trian = selector.transform(x_trian_vec)  # 转换x_train
    selected_x_test = selector.transform(x_test_vec)  # 转换x_test

    return selected_x_trian, y_train, selected_x_test, y_test

# if __name__ == '__main__':
#     split_data_sample()
