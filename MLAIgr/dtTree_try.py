#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/23 22:34
# @Author  : 朱紫宇
# @File    : dtTree_try.py
# @Software: PyCharm
from math import log
import operator
import treeplotter
import matplotlib.pyplot as plt

def createDataSet():
    dataSet = [[0, 0, 0, 0, 'N'],
               [0, 0, 0, 1, 'N'],
               [1, 0, 0, 0, 'Y'],
               [2, 1, 0, 0, 'Y'],
               [2, 2, 1, 0, 'Y'],
               [2, 2, 1, 1, 'N'],
               [1, 2, 1, 1, 'Y']]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #7
    labelCounts = {}
    for featVec in dataSet:#1.[0, 0, 0, 0, 'N']
        currentLabel = featVec[-1] # 1.'N',2.n,3.y,4.y,5.n,6.y,7y
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1      # 数每一类各多少个， {'Y': 4, 'N': 3}
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def majorityCnt(classList):
    """
    输入：分类类别列表
    输出：子节点的分类
    描述：数据集已经处理了所有属性，但是类标签依然不是唯一的，
          采用多数判决的方法决定该子节点的分类
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reversed=True)
    return sortedClassCount[0][0]
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                 #feature个数,4
    baseEntropy = calcShannonEnt(dataSet)             #整个dataset的熵
    bestInfoGainRatio = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  #每个feature的list
        uniqueVals = set(featList)                      #每个list的唯一值集合
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  #每个唯一值对应的剩余feature的组成子集
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            splitInfo += -prob * log(prob, 2)
        infoGain = baseEntropy - newEntropy              #这个feature的infoGain
        if (splitInfo == 0): # fix the overflow bug
            continue
        infoGainRatio = infoGain / splitInfo             #这个feature的infoGainRatio
        if (infoGainRatio > bestInfoGainRatio):          #选择最大的gain ratio
            bestInfoGainRatio = infoGainRatio
            bestFeature = i                              #选择最大的gain ratio对应的feature
    return bestFeature


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:                      #只看当第i列的值＝value时的item
            reduceFeatVec = featVec[:axis]              #featVec的第i列给除去
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]         # ['N', 'N', 'Y', 'Y', 'Y', 'N', 'Y']
    if classList.count(classList[0]) == len(classList): #'N'
        # classList所有元素都相等，即类别完全相同，停止划分
        return classList[0]                                  #splitDataSet(dataSet, 0, 0)此时全是N，返回N
    if len(dataSet[0]) == 1:       #[0, 0, 0, 0, 'N']                          #[0, 0, 0, 0, 'N']
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)             #0－> 2
        # 选择最大的gain ratio对应的feature
    bestFeatLabel = labels[bestFeat]                         #outlook -> windy
    myTree = {bestFeatLabel:{}}
        #多重字典构建树{'outlook': {0: 'N'
    del(labels[bestFeat])                                    #['temperature', 'humidity', 'windy'] -> ['temperature', 'humidity']
    featValues = [example[bestFeat] for example in dataSet]  #[0, 0, 1, 2, 2, 2, 1]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]                                #['temperature', 'humidity', 'windy']
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
            # 划分数据，为下一层计算准备
    return myTree
def classify(inputTree, featLabels, testVec):
    """
    输入：决策树，分类标签，测试数据
    输出：决策结果
    描述：跑决策树
    """

    firstStr = list(inputTree.keys())[0]                       # ['outlook'], outlook
    secondDict = inputTree[firstStr]                           # {0: 'N', 1: 'Y', 2: {'windy': {0: 'Y', 1: 'N'}}}
    featIndex = featLabels.index(firstStr)                     # outlook所在的列序号0
    for key in secondDict.keys():                              # secondDict.keys()＝[0, 1, 2]
        if testVec[featIndex] == key:                          # secondDict[key]＝N
            # test向量的当前feature是哪个值，就走哪个树杈
            if type(secondDict[key]).__name__ == 'dict':       # type(secondDict[key]).__name__＝str
                # 如果secondDict[key]仍然是字典，则继续向下层走
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                # 如果secondDict[key]已经只是分类标签了，则返回这个类别标签
                classLabel = secondDict[key]
    return classLabel

# Create Test Set
def createTestSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    testSet = [[0, 1, 0, 0],
               [0, 2, 1, 0],
               [2, 1, 1, 0],
               [0, 1, 1, 1],
               [1, 1, 0, 1],
               [1, 0, 1, 0],
               [2, 1, 0, 1]]
    return testSet

def classifyAll(inputTree, featLabels, testDataSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testDataSet:               #逐个item进行分类判断
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    labels_tmp = labels[:]
    desicionTree = createTree(dataSet, labels_tmp)
    treeplotter.createPlot(desicionTree)
    inputTree = desicionTree
    featLabels = ['outlook', 'temperature', 'humidity', 'windy']
    testVec = [0, 1, 0, 0]
    classify(inputTree, featLabels, testVec)
    # 多分类
    print("##"*10)
    testSet = createTestSet()
    print('classifyResult:\n', classifyAll(desicionTree, labels, testSet))



np.arr