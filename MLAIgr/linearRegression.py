#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/24 01:36
# @Author  : 朱紫宇
# @File    : linearRegression.py
# @Software: PyCharm
import numpy as np


# def dataSet():
#     """
#
#     :param data: 数据集
#     :return:
#     """

class LinearRegression:
    def __init__(self,,theta=None,bias=None,iteration=10000):
        self.iteration=iteration
        self.theta = theta
        self.bias = bias

    def fit(self,theta,x,y,lr):
        for _ in range(self.iteration):
            y_hat = self.lossFuncation(x,y)
            error = y-y_hat
            gd = self.gradient(theta,x,y,lr)

    def lossFuncation(self,x,y):
        return (1/2)*np.sum(np.sqrt(np.dot(x.t*self.theta)-y))

    def gradient(self,theta,x,y,lr):
        theta = theta+lr*(1/len(x))*np.sum(np.sqrt(np.dot(x.T,theta)*x))
        self.theta=theta
        return self.theta