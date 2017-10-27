'''
#This script shows how to predict stock prices using a basic RNN
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

# parameters to be set
look_back = 1
epochs = 30
batch_size = 1

# fix random seed for reproducibility
np.random.seed(7)
# read all prices using panda
dataframe =  pd.read_csv('USDJPY_Candlestick_10_m_BID_24.04.2017-25.04.2017.csv',   header=0)
dataframe.head()
dataset = dataframe['Close']
# reshape to column vector
#close_prices = close.values.reshape(len(close), 1)
dataset = dataset.values
dataset = dataset.astype('float32')
close_prices = dataset.reshape((-1,1))
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices = scaler.fit_transform(close_prices)
# split data into training set and test set
train_size = int(len(close_prices) * 0.67)
test_size = len(close_prices) - train_size
train, test = close_prices[0:train_size,:], close_prices[train_size:len(close_prices),:]

print('Split data into training set and test set... Number of training samples/ test samples:', len(train), len(test))
# convert an array of values into a time series dataset
# in form
#                     X                     Y
# t-look_back+1, t-look_back+2, ..., t     t+1

def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# convert Apple's stock price data into time series dataset
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
(trainX.shape, trainY.shape, testX.shape, testY.shape)
# reshape input of the LSTM to be format [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
(trainX.shape, testX.shape)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size,verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions and targets to unscaled
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.5f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.5f RMSE' % (testScore))

# shift predictions of training data for plotting
trainPredictPlot = np.empty_like(close_prices)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift predictions of test data for plotting
testPredictPlot = np.empty_like(close_prices)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(close_prices)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(close_prices))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()