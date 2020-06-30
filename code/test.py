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