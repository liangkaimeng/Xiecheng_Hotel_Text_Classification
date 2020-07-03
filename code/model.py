#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : lkm

from sklearn.svm import SVC
from joblib import dump, load
from Xiecheng_Hotel_Text_Classification.code import split_data

# 定义模型
model = SVC(C=2.0, kernel='rbf')
# 获取数据
selected_x_trian, y_train, selected_x_test, y_test = split_data.split_data_sample()
# 模型拟合
model.fit(selected_x_trian, y_train)
# 模型持久化
dump(model, 'SVC_MODEL.joblib')

