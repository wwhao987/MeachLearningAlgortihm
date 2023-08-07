#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/8 17:07
# @Author  : 朱紫宇
# @File    : regressionModel.py
# @Software: PyCharm
import numpy as np
from tqdm import tqdm
import time
class LinearRegression:
    def __init__(self, lr=0.01, epochs=10000,threshold=1e-10,regularization=None, lambda_=0):
        self.lr = lr
        self.epochs = epochs
        self.regularization = regularization
        self.threshold = threshold
        self.lambda_ = lambda_
        self.w = None  # 权重
        self.b = None  # 偏置

    def fit(self, X, y):
        # 初始化权重和偏置
        self.w = np.zeros(X.shape[1])
        self.b = 0
        n = tqdm(range(self.epochs))
        prev_loss = float('inf')
        # 梯度下降算法
        for epoch in n:
            # 计算预测值
            y_pred = np.dot(X, self.w) + self.b

            # 计算梯度
            dw = -(2 / len(X)) * np.dot(X.T, y - y_pred)
            db = -(2 / len(X)) * np.sum(y - y_pred)

            # 添加正则化项
            if self.regularization == 'l2':
                dw += (2 / len(X)) * self.lambda_ * self.w
            elif self.regularization == 'l1':
                dw += (2 / len(X)) * self.lambda_ * np.sign(self.w)

            # 更新权重和偏置
            self.w -= self.lr * dw
            self.b -= self.lr * db
            # 计算损失函数的变化
            loss = np.mean((y_pred - y) ** 2)

            # cacl acc
            predictions = np.round(y_pred)
            accuracy = np.mean(predictions == y)

            # std_error and mean_error
            std = np.std(y_pred - y)
            mea = np.mean(np.abs(y_pred - y))

            # print error
            if epoch % 10 == 0:
                time.sleep(0.3)
                print(f"Epoch{epoch}:Loss={loss},Accuracy={accuracy},\
                            Std={std},MEA={mea}")
            if epoch > 0:
                loss_diff = np.abs(loss - prev_loss)

                # 判断是否收敛
                if loss_diff < self.threshold:
                    print(f"Converged after {epoch} iterations")
                    break

            prev_loss = loss
    def predict(self, X):
        # 根据学习得到的权重和偏置进行预测
        return np.dot(X, self.w) + self.b


X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])  # 输入特征
y = np.array([2, 3, 4, 5])  # 输出值
X_new = np.array([[1, 5], [1, 6]])
lr = LinearRegression()
lr.fit(X,y)
print(lr.predict(X_new))