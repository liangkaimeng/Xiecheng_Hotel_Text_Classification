#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : lkm

import _thread
import warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from Xiecheng_Hotel_Text_Classification.code import split_data

warnings.filterwarnings('ignore')

class AdjustParams:
    def __init__(self):
        selected_x_trian, y_train, selected_x_test, y_test = split_data.split_data_sample()
        self.x_train = selected_x_trian
        self.y_train = y_train
        self.x_test = selected_x_test
        self.y_test = y_test

        self.Svm_Model()
        # self.RandomForest_Model()
        # self.Extratree_Model()



    def Svm_Model(self):
        param_grid = {}
        param_grid['C'] = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
        param_grid['kernel'] = ['linear', 'poly', 'sigmoid', 'rbf']
        model = SVC()
        kfold = KFold(n_splits=10, random_state=7)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
        grid_result = grid.fit(X=self.x_train, y=self.y_train)
        print('最优：%s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))
        cv_results = zip(grid_result.cv_results_['mean_test_score'],
                         grid_result.cv_results_['std_test_score'],
                         grid_result.cv_results_['params'])
        for mean, std, param in cv_results:
            print('%f (%f) with %r' % (mean, std, param))

    def RandomForest_Model(self):
        param_grid = {}
        param_grid['n_estimators'] = range(10, 71, 10)
        param_grid['max_depth'] = range(3, 14, 2)
        param_grid['criterion'] = ['gini', 'entropy']
        param_grid['min_samples_split'] = range(50, 191, 20)
        model = RandomForestClassifier()
        kfold = KFold(n_splits=10, random_state=7)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
        grid_result = grid.fit(X=self.x_train, y=self.y_train)
        print('最优：%s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))
        cv_results = zip(grid_result.cv_results_['mean_test_score'],
                         grid_result.cv_results_['std_test_score'],
                         grid_result.cv_results_['params'])
        for mean, std, param in cv_results:
            print('%f (%f) with %r' % (mean, std, param))

    def Extratree_Model(self):
        param_grid = {}
        param_grid['n_estimators'] = range(10, 71, 10)
        param_grid['max_depth'] = range(3, 14, 2)
        param_grid['criterion'] = ['gini', 'entropy']
        param_grid['min_samples_split'] = range(50, 191, 20)
        model = ExtraTreesClassifier()
        kfold = KFold(n_splits=10, random_state=7)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
        grid_result = grid.fit(X=self.x_train, y=self.y_train)
        print('最优：%s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))
        cv_results = zip(grid_result.cv_results_['mean_test_score'],
                         grid_result.cv_results_['std_test_score'],
                         grid_result.cv_results_['params'])
        for mean, std, param in cv_results:
            print('%f (%f) with %r' % (mean, std, param))

if __name__ == '__main__':
    AdjustParams()