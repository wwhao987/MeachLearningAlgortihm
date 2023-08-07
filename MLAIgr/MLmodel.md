### 概念

K最近邻（K-Nearest Neighbors，KNN）是一种常用的分类和回归算法，它基于实例之间的相似性进行预测。KNN算法的推导过程可以分为以下几个步骤：

1. 数据准备：首先，我们需要准备带有标签的训练数据集。每个数据样本都有一组特征和对应的标签。这些特征可以表示为向量，标签可以是分类的类别或者连续的数值。
2. 计算距离：KNN算法使用距离来衡量实例之间的相似性。常用的距离度量方法有欧氏距离、曼哈顿距离、余弦相似度等。对于两个特征向量x和y，距离度量可以表示为d(x, y)。
3. 选择K值：K值是KNN算法的一个超参数，用于确定预测时考虑的最近邻居数量。我们需要根据具体的问题和数据集来选择合适的K值。
4. 预测过程：对于一个新的测试样本，KNN算法的预测过程包括以下几个步骤：
   * 计算测试样本与训练集中每个样本之间的距离。
   * 根据距离的大小，选择距离最近的K个样本。
   * 统计这K个样本中各个类别的数量。
   * 根据多数表决原则，将测试样本归类为数量最多的类别。
   * 如果是回归问题，可以计算K个样本标签的平均值作为预测结果。
5. 模型评估：使用测试集对KNN模型进行评估，常见的评估指标包括准确率、精确率、召回率等。

需要注意的是，KNN算法并没有显示的训练过程，它属于一种懒惰学习（lazy learning）方法，只在预测时进行计算。

总结起来，KNN算法的推导过程包括数据准备、距离计算、选择K值、预测过程和模型评估。通过计算样本之间的距离和多数表决原则，KNN算法能够根据相似性进行分类或回归预测。

`````file
训练过程进行数据获取,伪训练学习的过程，算计距离：在预测部分进行，投票进行
模型存储：存储__init__()&训练集

```
````
`````

### 公式

KNN算法中的距离度量通常使用欧氏距离（Euclidean distance）或曼哈顿距离（Manhattan distance）。下面给出这两种距离的具体公式：

1. 欧氏距离（Euclidean distance）： 欧氏距离是最常用的距离度量方法，它计算两个样本向量之间的直线距离。
   对于两个样本向量x和y，欧氏距离可以表示为： d(x, y) = sqrt(∑(xi - yi)²)
   其中，xi和yi分别表示向量x和y的第i个特征，∑表示求和运算，sqrt表示平方根运算。
2. 曼哈顿距离（Manhattan distance）： 曼哈顿距离是另一种常用的距离度量方法，它计算两个样本向量之间的城市街区距离（即沿坐标轴的距离总和）。
   对于两个样本向量x和y，曼哈顿距离可以表示为： d(x, y) = ∑|xi - yi|
   其中，xi和yi分别表示向量x和y的第i个特征，∑表示求和运算，| |表示绝对值运算。

在KNN算法中，我们可以使用这些距离度量方法来计算样本之间的相似性，并根据距离的大小选择最近的K个邻居。然后，通过多数表决原则将测试样本分类为数量最多的类别，或者计算K个样本标签的平均值作为回归问题的预测结果。

## 线性回归

假设我们有一个训练数据集，其中包含m个样本。每个样本由输入特征向量x和对应的输出值y组成。我们的目标是找到一个线性模型，使得输入特征x能够预测输出值y。

1. 定义模型： 我们假设线性模型为： y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b 其中，w₁, w₂, ..., wₙ是权重（也称为系数），x₁, x₂, ..., xₙ是输入特征，b是偏置（也称为截距）。
2. 定义损失函数： 我们使用平方损失函数（Mean Squared Error，MSE）作为衡量模型预测与真实值之间差异的指标： Loss = (1/m) \* ∑[i=1 to m] (yᵢ - (w₁x₁ᵢ + w₂x₂ᵢ + ... + wₙxₙᵢ + b))² 其中，yᵢ是第i个样本的真实输出值，x₁ᵢ, x₂ᵢ, ..., xₙᵢ是第i个样本的输入特征。
3. 最小化损失函数： 我们的目标是找到一组最优的权重w和偏置b，使得损失函数达到最小值。为了最小化损失函数，我们可以使用梯度下降法等优化算法。通过不断迭代更新权重和偏置，直到损失函数收敛或达到停止条件。
4. 梯度下降更新规则： 在梯度下降算法中，我们根据损失函数的梯度来更新权重和偏置。梯度表示了损失函数相对于权重和偏置的变化率。更新规则如下： wⱼ = wⱼ - α \* (∂Loss/∂wⱼ) b = b - α \* (∂Loss/∂b) 其中，α是学习率，控制每次迭代的步长。
5. 计算梯度： 我们需要计算损失函数对权重和偏置的偏导数。根据链式法则，可以推导出： ∂Loss/∂wⱼ = (2/m) \* ∑[i=1 to m] (yᵢ - (w₁x₁ᵢ + w₂x₂ᵢ + ... + wₙxₙᵢ + b)) \* (-xⱼᵢ), ∂Loss/∂b = (2/m) \* ∑[i=1 to m] (yᵢ - (w₁x₁ᵢ + w₂x₂ᵢ + ... + wₙxₙᵢ + b))
6. ## 梯度更新模版
7. ```python
   #
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
   ```

### 线性回归公式

#### objective

$J({\theta})$=$\frac{1}{2}\sum_{i=1}^m(h_{\theta}(x^i)-y^i)^2$

$\theta$=$(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$

#### L1 Norm

$J({\theta})$=$\frac{1}{2m}\sum_{i=0}^m(h_{\theta}(x^i)-y^i)^2+\lambda\sum_{j=0}^n|{\theta_j}|$

#### L2 Norm

$J({\theta})$=$\frac{1}{2m}\sum_{i=0}^m(h_{\theta}(x^i)-y^i)+\lambda\sum_{j=1}^n|\theta_j|^2$

#### ElasitcNet

$J({\theta}$)=$\frac{1}{2m}\sum_{i=0}^m(h_\theta(x^i)-y^i)+\lambda(p\sum_{j=1}^n|\theta_j|+(1-p)\sum_{j=1}^n|{\theta_j}|^2$

#### 交叉墒

$$
J(\mathbf{θ})=-\frac{1}{m}∑_{i=1}^{m}y_ilogh_θ(x_i)+(1−y_i)log(1−h_θ(x_i))
$$
