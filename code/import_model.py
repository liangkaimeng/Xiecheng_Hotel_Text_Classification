#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : lkm

from joblib import load
from sklearn.metrics import classification_report, roc_auc_score
from Xiecheng_Hotel_Text_Classification.code import split_data

selected_x_trian, y_train, selected_x_test, y_test = split_data.split_data_sample()
# 导入模型
model = load(r"SVC_MODEL.joblib")
y_svc = model.predict(selected_x_test)
print('svc测试集: ', classification_report(y_test, y_svc))
print('svc测试集auc：', roc_auc_score(y_test, y_svc))

