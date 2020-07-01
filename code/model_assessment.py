#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : lkm

import warnings
from Xiecheng_Hotel_Text_Classification.code import split_data
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')

class ModelAssessment:
    def __init__(self):
        selected_x_trian, y_train, selected_x_test, y_test = split_data.split_data_sample()
        self.x_train = selected_x_trian
        self.y_train = y_train
        self.x_test = selected_x_test
        self.y_test = y_test

        # self.Conventional_Assess_Model()
        self.Integration_Assess_Model()

    def Conventional_Assess_Model(self):
        models = {}
        models['LR'] = LogisticRegression()
        models['KNN'] = KNeighborsClassifier()
        models['DTC'] = DecisionTreeClassifier()
        models['SVM'] = SVC()

        # 算法比较
        results = []
        for key in models:
            kfold = KFold(n_splits=10, random_state=7)
            cv_results = cross_val_score(models[key], self.x_train, self.y_train,
                                         cv=kfold, scoring='accuracy')
            results.append(cv_results)
            print('%s : %f (%f)' % (key, cv_results.mean(), cv_results.std()))

        # 绘制箱线图--数据的分布情况
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(models.keys())
        plt.show()

    def Integration_Assess_Model(self):
        models = {}
        models['AB'] = AdaBoostClassifier()
        models['GBM'] = GradientBoostingClassifier()
        models['RF'] = RandomForestClassifier()
        models['ETR'] = ExtraTreesClassifier()

        # 算法比较
        results = []
        for key in models:
            kfold = KFold(n_splits=10, random_state=7)
            cv_results = cross_val_score(models[key], self.x_train, self.y_train,
                                         cv=kfold, scoring='accuracy')
            results.append(cv_results)
            print('%s : %f (%f)' % (key, cv_results.mean(), cv_results.std()))

        # 绘制箱线图--数据的分布情况
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(models.keys())
        plt.show()


if __name__ == '__main__':
    ModelAssessment()