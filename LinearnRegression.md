# LiearnRegression

##### LiearnRegressionNotion

- Superviesed Algorithm used for predicting continuous outcome variable
- Find the minimize the sum fo squared residuals in datasets
- Is a iteration algorithm
- X determined Y
- **$X$ and $Y$** have a relation of  Affine
-  hope the all dataset on training uniform distribution at the side of line
- **Best model**：The distance from the data from all training sets to the straight line is minimal

##### primary components of LiearnRegression

- **Dependent Variable (Y):** The variable we want to predict or forecast.
- **Independent Variables (X):** One or more variables that are used to determine Y.
- **Coefficient (B):** The weight or importance given to the independent variables.
- **Intercept (A):** The base value of Y when all independent variables are zero.
- **Error Term (E):** The difference between the actual and predicted values of Y.

##### formula

- $Y$=$\theta^{T}*X+\Sigma_{i=1}^m{\varepsilon_{i}}$ （1）
  - **Note**:$\varepsilon_{i}$ subject to norm distribution,$\mu=0,\sigma$
  - **norm distribution**:$y=\frac{1}{\sigma*\sqrt{2*\pi}}*\exp^{\frac{\Sigma_{i=1}^m\varepsilon_{i}}{2\sigma^2}}$（2）

##### object

- Minimize the error between $y_{pred}$ and $y_{true}$
- $J(\theta)=\frac{1}{2m}\Sigma_{i=1}^m(h_{\theta(x^I)}-y^i)^2$  :最小二乘法 等价极大似然估计 -$\ln\theta$

- 中心极限定理：$N$足够大时，数据服从正态分布，变量间相互独立
- 将（1）带入（2），然后去对数似然函函数，即可得到目标函数

##### $\theta$求解过程

- 解析式法
- $J(\theta)=\frac{1}{2m}\Sigma_{i=1}^m(h_{\theta(x^I)}-y^i)^2$ =$\frac{1}{2}(X\theta-Y)^T(X\theta-Y)\rightarrow min_{\theta}J(\theta)$
- $$\nabla_{\theta}J(\theta)=\nabla_{\theta}(\frac{1}{2}(X\theta-Y)^T(X\theta-Y))\\=\nabla_{\theta}(\frac{1}{2}(\theta^TX^T-Y^T)(X\theta-Y))\\=\nabla_\theta(\frac{1}{2}(\theta^TX^TX\theta-\theta^TX^TY-Y^TX\theta-Y^TY)\\=\frac{1}{2}(2X^TX\theta-x^TY-(Y^TX)^T)\\=X^TX\theta-X^TY$$
- $$\theta=(X^TX)^{-1}X^TY$$
- Problem:$X^TX$一定要可逆，否则无法求解
- solution：加入其他数据，平滑因子$\theta=(XTX+\lambda{I})^{-1}Xy$
- 矩阵难求逆



##### costFunction

![costFunction](/Users/afei/PycharmProjects/deepBlueAILecture/python基础/NLP/image/目标函数.png)

##### overfitting 

- $L_{1norm},L_{2norm}$
- $L_{1nomr}:\\J(\theta)=\frac{1}{2}\Sigma_{1}^m(\hat{y-y})\\\Sigma_{i=1}^m|\theta_{i}|\le{t}$
- $L_{2nomr}:\\J(\theta)=\frac{1}{2}\Sigma_{1}^m(\hat{y-y})\\\Sigma_{i=1}^m|\theta_{i}^2|\le{t}^2$
- ![image](/Users/afei/PycharmProjects/deepBlueAILecture/python基础/NLP/image/L1`&L2.png)

##### L_1 and L_2

- L1求解更快，会产生更多的稀疏矩阵，鲁棒性略差
- L2求解略微慢，但是不会产生稀疏解，鲁棒性更好
- 两者都可以降低模型的复杂度，其中$\lambda$的大小可以控制模型的复杂度，$\lambda$大意味着会降低更多的参数对特征X的影响，模型则简单，容易underfitting，其中：L2需要手动设置threshold来进行特征过滤

- ElasitcNet
  - ![image](/Users/afei/PycharmProjects/deepBlueAILecture/python基础/NLP/image/ElasitcNet.png)

- $\lambda$是模型的超参数，可以手动给定或则使用CV,网格搜索

##### 调参数

- ![iamge](/Users/afei/PycharmProjects/deepBlueAILecture/python基础/NLP/image/corssvalidation.png)

##### summary

- ![image](/Users/afei/PycharmProjects/deepBlueAILecture/python基础/NLP/image/summary.png)