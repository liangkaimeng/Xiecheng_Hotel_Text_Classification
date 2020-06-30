#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : lkm

import re
import pandas as pd

data_file = r"D:\LKM\Deep_Learning\Xiecheng_Hotel_Text_Classification\data\ChnSentiCorp_htl_all.csv"
stop_word_file = r"D:\LKM\Deep_Learning\Xiecheng_Hotel_Text_Classification\data\baidu_stopwords.txt"

class ClearData_Class:
    def __init__(self):
        pass
        # self.Remove_Impurity(self.GetData(), 'review')

    def Remove_Impurity(self, data, field):
        data[field] = data[field].astype(str) # Convert data format to object type
        patten = r"[!\"#$%&'()*+,-./:;<=>?@[\\\]^_`{|}~—！，。？·￥、《》···【】：" "''\s0-9]+"
        self.re_obj = re.compile(patten) # Remove puntuation mark and number
        data[field] = data[field].apply(self.remove)




    def remove(self, text):
        return self.re_obj.sub('', text)

    def GetData(self):
        return pd.read_csv(data_file)



if __name__ == '__main__':
    ClearData_Class()











