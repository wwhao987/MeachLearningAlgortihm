#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/6 19:46
# @Author  : 朱紫宇(朱飞)
# @File    : knnmodel.py
# @Software: PyCharm

"""
K:
data
距离排序
"""

import numpy as np

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
class KNNClassifier:
    def __init__(self, k):
        self.k = k
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # 投票法
    def predict(self, X_test):
        y_pred = []
        for x_test in X_test:
            # 计算测试样本与所有训练样本的距离
            distances = [self.euclidean_distance(x_test, x_train) for x_train in self.X_train]
            # print(f"distance:{distances}")
            # 根据距离排序，获取最近的K个邻居的索引
            k_indices = np.argsort(distances)[:self.k]
            # print(f"index:{k_indices}")
            # 对K个邻居的标签进行统计
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            # print("=======")
            # print(k_nearest_labels)
            # 统计标签出现的次数
            label_counts = {}
            for label in k_nearest_labels:
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
            # 多数表决，选择出现次数最多的标签作为预测结果
            y_pred.append(max(label_counts, key=label_counts.get))
        return y_pred
    # 加权平均
    def means(self,X_test):
        """

        :param X_test:
        :return:
        """
        y_pred = []
        for x_test in X_test:
            # 计算测试样本与所有训练样本的距离
            distances = [self.euclidean_distance(x_test, x_train) for x_train in self.X_train]
            # print(f"distance:{distances}")
            #权重
            weights = [1/(i+1e-5)for i in distances]
            weights /=sum(weights)
            # 根据距离排序，获取最近的K个邻居的索引
            k_indices = np.argsort(weights)[:self.k]
            # k_nearest_labels = [self.y_train[i] for i in k_indices]
            y_pred.append(k_indices)
        return y_pred


    def confusion_matrix(self, y_true, y_pred):
        unique_labels = np.unique(y_true)
        num_labels = len(unique_labels)
        matrix = np.zeros((num_labels, num_labels), dtype=int)
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        for true_label, pred_label in zip(y_true, y_pred):
            true_index = label_to_index[true_label]
            pred_index = label_to_index[pred_label]
            matrix[true_index][pred_index] += 1
        return matrix

    def accuracy(self, y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        return correct / total

    def precision(self, y_true, y_pred):
        unique_labels = np.unique(y_true)
        precision_scores = {}
        for label in unique_labels:
            true_positives = np.sum((y_true == label) & (y_pred == label))
            predicted_positives = np.sum(y_pred == label)
            if predicted_positives == 0:
                precision_scores[label] = 0
            else:
                precision_scores[label] = true_positives / predicted_positives
        return precision_scores

    def recall(self, y_true, y_pred):
        unique_labels = np.unique(y_true)
        recall_scores = {}
        for label in unique_labels:
            true_positives = np.sum((y_true == label) & (y_pred == label))
            actual_positives = np.sum(y_true == label)
            if actual_positives == 0:
                recall_scores[label] = 0
            else:
                recall_scores[label] = true_positives / actual_positives
        return recall_scores

    def f1_score(self, y_true, y_pred):
        unique_labels = np.unique(y_true)
        f1_scores = {}
        for label in unique_labels:
            precision = self.precision(y_true, y_pred)[label]
            recall = self.recall(y_true, y_pred)[label]
            if precision + recall == 0:
                f1_scores[label] = 0
            else:
                f1_scores[label] = 2 * (precision * recall) / (precision + recall)
        return f1_scores



# 电影数据集
X_train = np.array([
    [120, 0, 0, 1],  # 电影A
    [90, 1, 0, 2],  # 电影B
    [105, 0, 1, 3],  # 电影C
    [95, 1, 0, 2],  # 电影D
    [130, 0, 0, 1],  # 电影E
    [110, 0, 1, 3]  # 电影F
    ])

y_train = np.array(['剧情', '喜剧', '动作', '喜剧', '剧情', '动作'])

# 测试样本
X_test = np.array([
    [10, 3],  # 未知电影X
    [90, 30],# 未知电影Y
])

# 创建KNN分类器对象，选择K=3
knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)

# 预测未知电影类型
y_pred1 = knn.predict(X_test)
y_pred = knn.means(X_test)
print(f"加权分类结果:\n{y_pred}")
# print(f"加权混淆矩阵是:\n{knn.confusion_matrix(y_train,y_pred)}")
print("--------"*30)
print(f"投票法分类结果:\n{y_pred1}")
print(f"投票法混淆矩阵是:\n{knn.confusion_matrix(y_train,y_pred1)}")