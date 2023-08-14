# -*- coding:utf-8 -*-
"""
作者：cqw
日期：2023年08月11日
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
print("================================")
#导入数据
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

#打乱数据
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

#分割训练集与测试集
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

#转换x的数据类型
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

#from_tensor_slices函数使输入特征和标签值一一对应
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#生成神经网络的参数
w1 = tf.Variable(tf.random.normal(([4, 3]),stddev=0.1,seed=1))
b1 = tf.Variable(tf.random.normal([3],stddev=0.1,seed=1))

lr = 0.1
train_loss_results = []
test_acc = []
epoch = 500
loss_all = 0

#训练部分
for epoch in range(epoch):
    for step,(x_train,y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ -y))
            loss_all += loss.numpy()
        grads = tape.gradient(loss, [w1 , b1])

        #实现梯度更新
        w1.assign_sub(grads[0]*lr)
        b1.assign_sub(grads[1]*lr)

    print("Epoch{},loss:{}".format(epoch,loss/4))
    train_loss_results.append(loss_all/4)
    loss_all = 0