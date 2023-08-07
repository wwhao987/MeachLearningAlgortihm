#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 16:05
# @Author  : 朱紫宇
# @File    : ll.py
# @Software: PyCharm
import numpy as np


class LogisticRegression:
    def __init__(self, alpha=0.01, num_iterations=1000, tol=1e-4, verbose=True):
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.tol = tol
        self.verbose = verbose
        self.theta = None
        self.costs = None

    def sigmoid(self, z):
        # Sigmoid函数
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y, theta):
        # 代价函数
        m = len(y)
        h = self.sigmoid(np.dot(X, theta))
        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost

    def gradient_descent(self, X, y):
        # 梯度下降算法
        m = len(y)
        self.theta = np.zeros(X.shape[1])
        self.costs = []

        for i in range(self.num_iterations):
            h = self.sigmoid(np.dot(X, self.theta))
            gradient = np.dot(X.T, (h - y)) / m
            self.theta -= self.alpha * gradient
            cost = self.cost_function(X, y, self.theta)
            self.costs.append(cost)

            if self.verbose:
                print("Iteration:", i + 1, "Cost:", cost)

            if i > 0 and abs(self.costs[i - 1] - cost) < self.tol:
                break

    def fit(self, X, y):
        # 模型训练
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        self.gradient_descent(X, y)

    def predict(self, X, threshold=0.5):
        # 模型预测
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        predictions = self.sigmoid(np.dot(X, self.theta))
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0
        return predictions

    def evaluate(self, X, y, threshold=0.5):
        # 模型评估
        predictions = self.predict(X, threshold)
        accuracy = np.sum(predictions == y) / len(y)

        true_positive = np.sum((predictions == 1) & (y == 1))
        false_positive = np.sum((predictions == 1) & (y == 0))
        true_negative = np.sum((predictions == 0) & (y == 0))
        false_negative = np.sum((predictions == 0) & (y == 1))

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_score = 2 * (precision * recall) / (precision + recall)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }


# 示例数据集
X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
y = np.array([0, 0, 1, 1])

# 创建逻辑回归模型对象
model = LogisticRegression(verbose=True)

# 模型训练
model.fit(X, y)

# 打印训练结果
print("Final Theta:", model.theta)
print("Final Cost:", model.costs[-1])

# 预测新样本
new_sample = np.array([[1, 6]])
prediction = model.predict(new_sample)
print("Prediction for new sample:", prediction)

# 模型评估
evaluation = model.evaluate(X, y)
print("Evaluation:", evaluation)
