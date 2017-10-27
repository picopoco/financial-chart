import pandas as pd
import numpy as np
import tensorflow as tf
from dplython import DplyFrame, X, select, sift  # dplython 이라는 모듈 추가
import matplotlib.pyplot as plt

tf.set_random_seed(4444)   # tf random_seed 넣음.

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)  #  최소값과의 차이
    denominator = np.max(data, 0) - np.min(data, 0)  #  최대값 - 최소값  = 최대 진폭 분모 나는느 값

    return numerator / (denominator + 1e-7) # 최소값차이 - 최대 진폭

# train Parameters
seq_length = 1
data_dim = 8
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500

# last, diff_24h, diff_per_24h, bid, ask, low, high, volume
data = DplyFrame(pd.read_csv('../bitcoin_ticker.csv', delimiter=','))
#data = data >> sift(X.rpt_key == 'btc_krw') >> select(X.last, X.diff_24h, X.diff_per_24h, X.bid, X.ask, X.low, X.high, X.volume)
#data = data >> sift(X.rpt_key == 'btc_krw') >> select(X.last, X.diff_24h, X.diff_per_24h, X.bid, X.ask, X.low, X.high, X.volume)
data = data >> sift(X.rpt_key == 'btc_krw') >> select(X.last, X.diff_24h, X.diff_per_24h, X.bid, X.ask, X.low, X.high, X.volume)
print("(X.Open)",data)

data = np.asarray(data)
data = MinMaxScaler(data)
x = data
y = data[:, [0]]  # last as label
#y = data[:100, [-1]]
print('y',y,np.shape(y))

# build a dataset
dataX = []
dataY = []

print("ylen:",len(y))
#for i in range(0,100):
for i in range(0, len(y) - seq_length):  # 총길이 - 60만큼 돌린다.
    _x = x[i:i + seq_length]  # 60만큼
    _y = y[i + seq_length]  # Next close price  ㅁ
    print(i,_x, "->", _y) # ㅇ
    dataX.append(_x)
    dataY.append(_y)

#np.savetxt('/home/yeolpyeong/bitcoin_ticker_dataX.csv', dataX, delimiter=',')
#np.savetxt('/home/yeolpyeong/bitcoin_ticker_dataY.csv', dataY, delimiter=',')

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
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
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
plt.show()