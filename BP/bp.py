# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/6 12:36
# @Author  : 朱紫宇(朱飞）
# @File    : bp.py BP手写神经网络
# @Software: PyCharm
import time
import numpy as np
from tqdm import tqdm


class Module(object):
    def __init__(self,input_size,hidden_size,output_size):
        #shape(n,m)
        self.input_size = input_size
        # hiddenlayer weights and bias,shape(m,h)
        self.hidden_size = hidden_size
        # outputlayer weights and bias,shape(h,)
        self.output_size = output_size
        # random initial weights of hidden_layer and output_layer
        # hidden_layer weights
        self.w_h = np.random.rand(self.input_size,self.hidden_size)
        self.b_h = np.random.rand(self.hidden_size)
        # output_layer weights
        self.w_o = np.random.rand(self.hidden_size,self.output_size)
        self.b_o = np.random.rand(self.output_size)

    def forward(self,X):
        """
        forward_processing
        :return:
        """
        self.z1 = np.dot(X,self.w_h)+self.b_h # 4*4
        self.a1 = self.sigmoid(self.z1) # 4*4
        self.z2 = np.dot(self.a1,self.w_o)+self.b_o # 4*1
        self.a2 = self.sigmoid(self.z2) # 4*1
        return self.a2

    def backward(self,X,y,lr):
        x_back = X.shape[0]
        # error of gredient
        self.loss = self.a2-y  # self.a2:4*1,y:2*4,self.loss:4*4
        # output_weights gredient
        self.g_0 =np.dot(self.a1.T,self.loss) # 4*4
        self.g_ob = np.sum(self.loss)
        # hidden_weights gredient
        self.g_oh = np.dot(self.loss.T,self.w_o) # with output self.w_o.T:1*4
        self.g_ih = self.g_oh*self.sigmoid_back(self.z1)
        self.g_Wh =np.dot(X.T,self.g_ih)
        self.g_hb = np.sum(self.g_ih)
        # update weights
        self.w_h -=lr*self.g_Wh
        self.g_hb -=lr*self.g_hb
        self.w_o -=lr*self.g_0.T
        self.b_o -=lr*self.g_ob

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_back(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def train(self,X, y, epochs, lr):
        tq = tqdm(range(epochs))
        prev_loss = float('inf')
        threshold=1e-10
        best_loss = float('inf')  # best loss
        best_weights = None  # best weights
        for epoch in tq:
            # forward
            output = self.forward(X)
            # backward
            self.backward(X, y, lr)
            # get loos
            loss = np.mean(0.5 * (output - y) ** 2)
            # cacl acc
            predictions = np.round(output)
            accuracy = np.mean(predictions == y)

            # std_error and mean_error
            std = np.std(output - y)
            mea = np.mean(np.abs(output - y))

            # print error
            if epoch % 50 == 0:
                time.sleep(0.3)
                print(f"Epoch{epoch}:Loss={loss},Accuracy={accuracy},\
                 Std={std},MEA={mea}")

            # get best_weights
            if loss < best_loss:
                best_loss = loss
                best_weights = {
                    'w_h': self.w_h,
                    'b_h': self.b_h,
                    'w_o': self.w_o,
                    'b_o': self.b_o
                }
            self.w_h = best_weights['w_h']
            self.b_h = best_weights['b_h']
            self.w_o = best_weights['w_o']
            self.b_o = best_weights['b_o']
            # condition of stop training
            if abs(loss - prev_loss) < threshold:
                print("Training stopped due to convergence.")
                break
            prev_loss = loss

    def predict(self, X):
        # prediction
        output = self.forward(X)
        return np.round(output)


if __name__ == '__main__':
    # initial object
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # 4*2
    y = np.array([[0,1,3,4],[4,5,6,7]]) # 2*4
    # get neuralnetwork,B
    nn = Module(input_size=2, hidden_size=4, output_size=1)

    # training model
    nn.train(X, y, epochs=10000, lr=0.001)
    # prediction
    predictions = nn.predict(X)
    print(f"Predictions:{predictions}")
