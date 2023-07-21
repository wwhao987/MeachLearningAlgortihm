# k-nearst neighnors

##### 1.task

- 分类
  - major voter
- 回归
  - mean value

##### 2.类型

- 有监督
  - 有X
  - 有Y

##### 3.原理

- 训练过程：存储数据
- 从training set中获取K个代测样本到待测样本距离最近的数据
- 由k个样本数据来预测当前待样本的label属性
- 距离度量：欧式距离



##### 4. K-value choice

- K-infulence with model
  - k值小，模型容易overfitting，模型复杂，sapce小，范围压缩，train error小
  - k值大，模型简单，模型容易underfitting
  - 极限bound：K=1or K=all data set
  - k对噪声数据敏感，有噪声时可能分不对
  - k值选取：交叉验证，网格搜索，长尾可视化

##### 5.预测分类规则

- 多数表决法
- 加权多数表决法：权重和距离成反比

##### 6.回归

- 平均值法

  ```python	
  ：：math:
    step1:target=6,x_train样本算距离
    step2:选取K
    step3得出结果：mean=sum(distance)/k
  ```

- 加权平均值法

  ```python	
  precetion = (W_1*y_1+w_2*y_2+...+w_n*y_n)/sum(w_i)
  加权：weight = 1/distance(target-x_i)
  ```

  

##### 7.knn存在问题

- 线性扫描，复杂度高
- 异常值敏感
- 内存消耗大
- 预测速度慢
- K值不好选取
- 对imbalance数据不友好
- 维度灾难