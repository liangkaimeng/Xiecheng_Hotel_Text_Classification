#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : lkm

import re
import jieba
import pandas as pd

# data_file = r"D:\LKM\Deep_Learning\Xiecheng_Hotel_Text_Classification\data\ChnSentiCorp_htl_all.csv"
stop_word_file = r"D:\LKM\Deep_Learning\Xiecheng_Hotel_Text_Classification\data\baidu_stopwords.txt"

class ClearData_Class:
    def __init__(self):
        pass

    """ 处理标点符号和数字 """
    def Remove_Impurity(self, data, field):
        data[field] = data[field].astype(str) # Convert data format to object type
        patten = r"[!\"#$%&'()*+,-./:;<=>?@[\\\]^_`{|}~—！，。？·￥、《》···【】：" "''\s0-9]+"
        self.re_obj = re.compile(patten) # Remove puntuation mark and number
        data[field] = data[field].apply(self.remove)
        data[field] = data[field].apply(self.cut_words)
        data[field] = data[field].apply(self.remove_stop)
        data[field] = data[field].apply(self.list_str)
        return data

    """ 将列表元素转换为字符串 """
    def list_str(self, content):
        return ' '.join(content)

    """ 移除停用词 """
    def remove_stop(self, words):
        stop_list = [i.strip() for i in open(stop_word_file, encoding='utf-8').readlines()]
        texts = []
        for word in words:
            if word not in stop_list:
                texts.append(word)
        return texts

    """ 中文分词 """
    def cut_words(self, words):
        return jieba.lcut(words)

    """ 去除标点符号和数字 """
    def remove(self, text):
        return self.re_obj.sub('', text)

    """ 获取数据 """
    def GetData(self, data_file):
        return pd.read_csv(data_file)

if __name__ == '__main__':
    ClearData_Class()











