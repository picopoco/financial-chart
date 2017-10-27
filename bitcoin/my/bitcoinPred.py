'''
This script shows how to predict stock prices using a basic RNN
'''
import os

import matplotlib
import numpy as np
import tensorflow as tf
import pandas as pd
from dplython import DplyFrame, X, select, sift  # dplython 이라는 모듈 추가
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


# def MinMaxScaler(data):
#     min1 = np.min(data)
#     min2 = np.min(data, 1)
#    # min2 = np.min(data, 2)
#   #  min3 = np.min(data, -1)
#
#     min = np.min(data, 0)
#     max = np.max(data, 0)
#     numerator = data - min
#     denominator = max - min
#     # noise term prevents the zero division
#     ed = 1e-7
#     res1 = numerator / (denominator )
#     res = numerator / (denominator + ed)
#     return res
# def MinMaxScaler(data):
#     numerator = data - np.min(data, 0)  #  최소값과의 차이
#
#     denominator = np.max(data, 0) - np.min(data, 0)  #  최대값 - 최소값  = 최대 진폭 분모 나는느 값
#
#     return numerator / (denominator + 1e-7) # 최소값차이 - 최대 진폭


# train Parameters
#seq_length = 60
seq_length = 5
data_dim = 5
hidden_dim = 10
output_dim = 1
learing_rate = 0.01
iterations = 2000




# Open, High, Low, Volume, Close
data = DplyFrame(pd.read_csv('../kaggle_bitcoin_from_2017_04_01_1.csv', delimiter=','))
data = data >> select(X.Close,X.Open,X.High,X.Low,X.Volume__Currency_)

data = data[0:1000]
print(len(data))
data = data.dropna(axis=0)

#df.dropna(thresh=3)
#df["COLUMN_A"].fillna(df["COLUMN_A".mean(), inplace=True)

#xy = data[:1:-1]
data = np.asarray(data)

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

#data = MinMaxScaler(data)
print(data)

x = data
#y = xy[:, [-1]]  # Close as label
#y = xy[0:,-1]# Close as label
y = data[:, [0]]   # 배열로 입력 [0] 값과 없는 것을 제대로 구별해야 한다.  shaper가 달라진다.
print('y',y,np.shape(y))


# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(  outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learing_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

print(len(trainX) ,len(trainY))
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse))

    # Plot trainX
   # plt.plot(scaler.inverse_transform(data))
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()

    # plt.plot(testY)
    # plt.plot(test_predict)
    # plt.xlabel("Time Period")
    # plt.ylabel("Stock Price")
    # plt.show()
