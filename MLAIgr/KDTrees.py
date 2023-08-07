#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/7 16:50
# @Author  : 朱紫宇
# @File    : KDTrees.py
# @Software: PyCharm

# import numpy as np
# a = [(2, 3), (5, 4),(3,1), (9, 6), (4, 7), (8, 1), (7, 2)]
# data = sorted(a,key=lambda x:x[0]) # [(2, 3), (4, 7), (5, 4), (7, 2), (8, 1), (9, 6)]
# # 获取中位数的索引
# median_index = len(data) // 2
# print(median_index)
#
# # 获取较大的中位数的点
# median_point = data[median_index]
#
# print(median_point)

import numpy as np

class KDTree:
    def __init__(self, points):
        self.points = points
        self.k = len(points[0])
        self.root = np.any(self.build_kdtree(points))

    class Node:
        def __init__(self, point, left, right):
            """
            跟左右
            :param point:
            :param left:
            :param right:
            """
            self.point = point
            self.left = left
            self.right = right

    def build_kdtree(self, points, depth=0):
        """
        构建KD树的递归方法
        :param points: 数据点集
        :param depth: 当前递归深度
        :return: KD树的根节点
        """
        if not np.any(points):
            return None

        # 选择当前深度对应的坐标轴进行划分先
        axis = depth % self.k # 0 表示x轴，1表示y轴
        points.sort(key=lambda point: point[axis]) # 点按照轴排序，取中位数
        median = len(points) // 2
        # 递归完成树的遍历，前序遍历（遍历顺序）：根左右
        return self.Node(
            point=points[median],
            left=self.build_kdtree(points[:median], depth + 1),
            right=self.build_kdtree(points[median + 1:], depth + 1)
        )

    def query(self, target, k):
        """
        执行最近邻搜索
        :param target: 目标点
        :param k: 需要搜索的最近邻点的个数
        :return: 最近邻点集合
        """
        self.best_points = []
        self.best_distances = []
        self.knn_search(self.root, target, k, depth=0)
        return self.best_points

    def knn_search(self, node, target, k, depth):
        """
        KD树最近邻搜索的递归方法
        :param node: 当前节点
        :param target: 目标点
        :param k: 需要搜索的最近邻点的个数
        :param depth: 当前递归深度
        """
        if node is None:
            return

        axis = depth % self.k
        curr_point = node.point
        # 计算欧式距离（2——norm)
        distance = np.linalg.norm(np.array(curr_point) - np.array(target))

        # 如果当前最近邻点集不满，或者当前点距离小于最远距离，则插入当前点
        if len(self.best_points) < k or distance < max(self.best_distances):
            self.insert_point(curr_point, distance, k)

        # 根据目标点的位置选择子树进行搜索
        if target[axis] < curr_point[axis]:
            self.knn_search(node.left, target, k, depth + 1)
        else:
            self.knn_search(node.right, target, k, depth + 1)

        # 检查目标点与当前划分轴的距离，决定是否需要搜索另一子树
        if abs(target[axis] - curr_point[axis]) < max(self.best_distances):
            if target[axis] < curr_point[axis]:
                self.knn_search(node.right, target, k, depth + 1)
            else:
                self.knn_search(node.left, target, k, depth + 1)

    def insert_point(self, point, distance, k):
        """
        维护当前最近邻点集的方法
        :param point: 待插入的点
        :param distance: 待插入点与目标点的距离
        :param k: 最近邻点的个数
        """
        if len(self.best_points) < k:
            self.best_points.append(point)
            self.best_distances.append(distance)
        else:
            max_distance_idx = np.argmax(self.best_distances)
            self.best_points[max_distance_idx] = point
            self.best_distances[max_distance_idx] = distance


