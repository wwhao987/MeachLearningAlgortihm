#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/7 17:56
# @Author  : 朱紫宇
# @File    : kdtest.py
# @Software: PyCharm
import numpy as np
from KDTrees import KDTree
if __name__ == '__main__':

    # 电影数据集
    movies = [
        [0.1, 0.8, 2010],   # 电影1的特征向量
        [0.5, 0.4, 1998],   # 电影2的特征向量
        [0.3, 0.6, 2005],   # 电影3的特征向量
        [0.9, 0.2, 2001],   # 电影4的特征向量
    ]

    # 将电影数据集转换为NumPy数组
    movies = np.array(movies)

    # 构建KDTree
    kdtree = KDTree(movies)

    # 测试目标电影
    target_movie = [0.4, 0.5, 2003]  # 目标电影的特征向量

    # 执行最近邻搜索
    k = 2  # 搜索最近的两个电影
    nearest_movies = kdtree.query(target_movie, k)

    # 输出结果
    print("目标电影的特征向量:", target_movie)
    print("最近的两个电影的特征向量:")
    for movie in nearest_movies:
        print(movie)