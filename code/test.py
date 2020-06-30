#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : lkm

from sklearn.feature_extraction.text import TfidfVectorizer  # tf-idf转换器
import tensorflow as tf  # 导入tensorflow
import matplotlib.pyplot as plt
import gensim  # 导入gensim
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from xgboost import XGBClassifier  # xgboost
import tensorflow.keras as keras
import jieba  # 导入jieba分词库
import re

plt.rcParams['font.sans-serif'] = ['SimHei']

""" Read Data """
file = r"D:\LKM\Deep_Learning\Xiecheng_Hotel_Text_Classification\data\ChnSentiCorp_htl_all.csv"
stop_file = r"D:\LKM\Deep_Learning\Xiecheng_Hotel_Text_Classification\data\baidu_stopwords.txt"
data = pd.read_csv(file)
# print(data.head(10))

# # 数据的形状
# print(data.shape)
# # 查看重复数据
# print(data.duplicated().sum())

# 将数据格式转换为字符串方便处理
data['review'] = data['review'].astype('str')

# 去除标点符号和数字
patten = r"[!\"#$%&'()*+,-./:;<=>?@[\\\]^_`{|}~—！，。？·￥、《》···【】：" "''\s0-9]+"  # 标点符号和数字
re_obj = re.compile(patten)
def clear(text):
    return re_obj.sub('', text)  #去除标点符号和数字
data['review'] = data['review'].apply(clear)

# 分词处理
def cut_words(words):
    return jieba.lcut(words)  # 使用lcut分词
data['review'] = data['review'].apply(cut_words)  #apply函数对series的每一行数据进行处理

# 停用词处理
stop_list = [
    i.strip() for i in open(file, encoding='utf-8').readlines()]  #读取停用词列表

def remove_stop(words):  #移除停用词函数
    texts = []
    for word in words:  # 遍历词列表里的每一个词
        if word not in stop_list:  # 若不在停用词列表中就将结果追加至texts列表中
            texts.append(word)
    return texts

data['review'] = data['review'].apply(remove_stop)


def list_str(content):
    return ' '.join(content)  #将列表元素转换为字符串

data['review'] = data['review'].apply(list_str)

def get_balanced_words(size,
                       positive_comment=data[data['label'] == 1],
                       negtive_comment=data[data['label'] == 0]):
    word_size = size // 2
    #获取正负评论数
    num_pos = positive_comment.shape[0]
    num_neg = negtive_comment.shape[0]
    #     当 正(负)品论数中<采样数量/2 时，进行上采样，否则都是下采样；
    #     其中pandas的sample方法里的repalce参数代表是否进行上采样，默认不进行
    balanced_words = pd.concat([
        positive_comment.sample(word_size,
                                replace=num_pos < word_size,
                                random_state=0),
        negtive_comment.sample(word_size,
                               replace=num_neg < word_size,
                               random_state=0)
    ])
    #     打印样本个数
    print('样本总数：', balanced_words.shape[0])
    print('正样本数：', balanced_words[data['label'] == 1].shape[0])
    print('负样本数：', balanced_words[data['label'] == 0].shape[0])
    print('')
    return balanced_words

data_4000 = get_balanced_words(4000)
data_6000 = get_balanced_words(6000)

x_train, x_test, y_train, y_test = train_test_split(data_4000['review'],
                                                    data_4000['label'],
                                                    random_state=1,
                                                    shuffle=True)

tf_idf = TfidfVectorizer()
x_trian_vec = tf_idf.fit_transform(x_train)  #将训练集文本转换为向量
x_test_vec = tf_idf.transform(x_test)  #将测试集文本转换为向量

from sklearn.feature_selection import f_classif
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

x_trian_vec = x_trian_vec.astype(np.float64)  #转换为numpy的float类型，便于sklearn处理
x_test_vec = x_test_vec.astype(np.float64)

selector = SelectKBest(f_classif, k=min(500, x_trian_vec.shape[1]))  #选择器
selector.fit(x_trian_vec, y_train)  #训练
selected_x_trian = selector.transform(x_trian_vec)  # 转换x_train
selected_x_test = selector.transform(x_test_vec) # 转换x_test

lr = LogisticRegression()
lr.fit(selected_x_trian, y_train)
y_lr = lr.predict(selected_x_test)
#打印训练集和测试集的f1值、recall等
print('lr测试集：', classification_report(y_test, y_lr))
print('lr测试集auc：', roc_auc_score(y_test, y_lr))
print()
print('lr训练集：', classification_report(y_train, lr.predict(selected_x_trian)))
print('lr训练集auc：', roc_auc_score(y_train, lr.predict(selected_x_trian)))


