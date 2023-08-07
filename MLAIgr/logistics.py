#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 14:48
# @Author  : 朱紫宇
# @File    : logistics.py
# @Software: PyCharm
import numpy as np
from tqdm import tqdm
import time
class LogisticRegression(object):
    def __init__(self,lr=0.001,tol=1e-5,iter=10000,verbose=True):
        self.lr=lr
        self.theta = None
        self.cost = None
        self.iter = iter
        self.tol = tol
        self.verbose = verbose
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def cost_func(self,x,y,theta):
        self.m = len(y)
        h_hat =  self.sigmoid(np.dot(x,theta))
        cost = (-1 / self.m) * np.sum(y * np.log(h_hat) + (1 - y) * np.log(1 - h_hat))
        return cost

    def optimizer(self,x,y):
        self.m = len(y)
        self.theta = np.zeros(x.shape[1])
        self.costs = []
        tq = tqdm(range(self.iter))
        for epoch in tq:
            h_hat = self.sigmoid(np.dot(x,self.theta))
            gd = np.dot(x.T,(h_hat-y))/self.m
            self.theta -= self.lr*gd
            cost=self.cost_func(x,y,self.theta)
            self.costs.append(cost)
            if epoch % 50 == 0 and self.verbose:
                time.sleep(0.3)
                print(f"Epoch{epoch}:cost={cost}")
            if epoch > 0 and abs(self.costs[epoch- 1] - cost) < self.tol or self.costs[epoch-1]==cost:
                print("---cost is not changing or the train processing is not subject the condition of preview---")
                break

    def train(self,x,y):
        ones = np.ones((x.shape[0], 1))
        X = np.concatenate((ones, x), axis=1)*
        self.optimizer(x, y)

    def predict(self, X):
        # 模型预测
        ones = np.ones((X.shape[0],1))
        X = np.concatenate((ones, X),axis=1)
        predictions = self.sigmoid(np.dot(X,self.theta))
        predictions = np.where(predictions >= self.threshold, 1, 0)
        return predictions
    def evaluate(self,x,y,threshold=0.5):
        pre = self.predict(x)
        pre[pre >= threshold] = 1
        pre[pre < threshold] = 0
        self.acc = np.sum(pre==y)/len(y)
        true_positive = np.sum((predictions == 1) & (y == 1))
        false_positive = np.sum((predictions == 1) & (y == 0))
        true_negative = np.sum((predictions == 0) & (y == 0))
        false_negative = np.sum((predictions == 0) & (y == 1))
        self.precision = true_positive / (true_positive + false_positive)
        self.recall = true_positive / (true_positive + false_negative)
        self.f1_score = 2 * (precision * recall) / (precision + recall)

        return {
            'accuracy': self.acc,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score
        }


if __name__ == '__main__':
    X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
    y = np.array([0, 0, 1, 1])
    # 创建逻辑回归模型对象
    model = LogisticRegression(verbose=True)
    # 模型训练
    model.train(X, y)
    # 打印训练结果
    print("Final Theta:", model.theta)
    print("Final Cost:", model.costs[-1])
    # 预测新样本
    new_sample = np.array([[1,6]])
    prediction = model.predict(new_sample)
    print("Prediction for new sample:", prediction)
    # 模型评估
    evaluation = model.evaluate(X, y)
    print("Evaluation:", evaluation)