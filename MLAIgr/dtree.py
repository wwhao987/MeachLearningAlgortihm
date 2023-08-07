#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 14:39
# @Author  : 朱紫宇
# @File    : dtree.py
# @Software: PyCharm
import numpy as np

class DecisionTree(object):
    def __init__(self,lr=0.001,interation=1000):
        self.lr = lr
        self.iteration = iteration
        self.theta = None
        self.entropy =None
        self._hd=None

    def calc_entropy(self,x):
        """

        :param x: training data
        :return:
        """
        label = x.iloc[:,-1]
        # 确定标签的类别
        lb = label.value_counts()
        for k in lb.keys():
            h_d = lb[k]/len(label)
            self.entropy = np.sum(-h_d*np.log(h_d))
        return self.entropy
    def cac_information_gain(self,x,k):
        """

        :param x: training data
        :param k: count of class
        :return:
        """
        entropy = self.calc_entropy(x)
        f_class = x[k].value_counts()
        for i in f_class.keys():
            h_c_d = f_class[i]/len(f_class.shape[0])
            en_i = self.calc_entropy(x[k]==i)
            gain +=h_c_d*en_i

        return entropy-gain

    # 获取标签最多的那一类
    def get_most_label(self,data):
        data_label = data.iloc[:, -1]
        label_sort = data_label.value_counts(sort=True)
        return label_sort.keys()[0]

    # 挑选最优特征，即信息增益最大的特征
    def get_best_feature(self,data):
        features = data.columns[:-1]
        res = {}
        for a in features:
            temp = self.calc_information_gain(data, a)
            res[a] = temp
        res = sorted(res.items(), key=lambda x: x[1], reverse=True)
        return res[0][0]

    ##将数据转化为（属性值：数据）的元组形式返回，并删除之前的特征列
    def drop_exist_feature(self,data, best_feature):
        attr = pd.unique(data[best_feature])
        new_data = [(nd, data[data[best_feature] == nd]) for nd in attr]
        new_data = [(n[0], n[1].drop([best_feature], axis=1)) for n in new_data]
        return new_data


class get_a_tree:
    def __init__(self,root=None,left=None,right=None,threshold=None,feature = None,label=None):
        self.root = root
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature =feature
        self.label = label
