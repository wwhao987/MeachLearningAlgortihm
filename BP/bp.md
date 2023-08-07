假设我们有一个三层的前馈神经网络，包括一个输入层、一个隐藏层和一个输出层。

1. 定义符号：

- 输入数据：我们有n个训练样本，每个样本有m个特征。我们用X表示输入数据矩阵，大小为n×m。
- 隐藏层：我们有h个隐藏单元。我们用H表示隐藏层的权重矩阵，大小为m×h。每个隐藏单元都有一个偏置项，我们用b1表示隐藏层的偏置向量，大小为h。
- 输出层：我们有一个输出单元。我们用O表示输出层的权重向量，大小为h×1。输出层也有一个偏置项，我们用b2表示输出层的偏置。
- 激活函数：我们使用sigmoid作为隐藏层和输出层的激活函数。

2. 前向传播：

- 隐藏层输入：Z1 = X · H + b1
- 隐藏层输出：A1 = sigmoid(Z1)
- 输出层输入：Z2 = A1 · O + b2
- 输出层输出：A2 = sigmoid(Z2)

3. 损失函数：

我们使用平方损失函数（mean squared error）来度量预测值与真实值之间的差异：

- 损失函数：L = 1/2 * (Y - A2)^2

其中，Y表示真实的输出值。

4. 反向传播：

反向传播的目标是计算损失函数对权重和偏置的梯度，以便通过梯度下降来更新它们。

- 损失函数对输出层输入的导数：dL_dZ2 = A2 - Y
- 输出层权重的梯度：dL_dO = A1.T · dL_dZ2
- 输出层偏置的梯度：dL_db2 = 求和(dL_dZ2)
- 隐藏层输出的导数：dL_dA1 = dL_dZ2 · O.T
- 隐藏层输入的导数：dL_dZ1 = dL_dA1 * sigmoid_derivative(Z1)
- 隐藏层权重的梯度：dL_dH = X.T · dL_dZ1
- 隐藏层偏置的梯度：dL_db1 = 求和(dL_dZ1)

其中，sigmoid_derivative(x) = sigmoid(x) · (1 - sigmoid(x)) 是sigmoid函数的导数。

5. 参数更新：

使用梯度下降法来更新权重和偏置：

- 更新隐藏层权重：H = H - learning_rate * dL_dH
- 更新隐藏层偏置：b1 = b1 - learning_rate * dL_db1
- 更新输出层权重：O = O - learning_rate * dL_dO
- 更新输出层偏置：

b2 = b2 - learning_rate * dL_db2

其中，learning_rate是学习率，控制每次更新的步长。

这就是一个基本的BP神经网络的定义和公式推导过程。通过重复进行前向传播、反向传播和参数更新，可以逐渐优化神经网络以更好地拟合训练数据。

```
self.H -= learning_rate * dL_dH
        self.b1 -= learning_rate * dL_db1
        self.O -= learning_rate * dL_dO
        self.b2 -= learning_rate * dL_db2
```
