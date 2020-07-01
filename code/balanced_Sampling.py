#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : lkm

import pandas as pd

def get_balanced_words(size, data, label):
    positive_comment = data[data[label] == 1]
    negtive_comment = data[data[label] == 0]
    word_size = size // 2
    # 获取正负评论数
    num_pos = positive_comment.shape[0]
    num_neg = negtive_comment.shape[0]
    #     当 正(负)品论数中<采样数量/2 时，进行上采样，否则都是下采样；
    #     其中pandas的sample方法里的repalce参数代表是否进行上采样，默认不进行
    balanced_words = pd.concat([positive_comment.sample(word_size, replace=num_pos < word_size,
                                                        random_state=0),
                                negtive_comment.sample(word_size, replace=num_neg < word_size,
                                                       random_state=0)])
    return balanced_words