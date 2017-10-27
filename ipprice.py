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
#dataframe =  pd.read_csv('USDJPY_Candlestick_10_m_BID_24.04.2017-25.04.2017.csv',   header=0)
dataframe =  pd.read_csv('data/DAT_ASCII_EURUSD_M1_2016.csv',   header=0)

dataframe.select()

#dataframe.head()
#dataset = dataframe['Close']
# reshape to column vector
#close_prices = close.values.reshape(len(close), 1)
# dataset = dataset.values
# dataset = dataset.astype('float32')
# close_prices = dataset.reshape((-1,1))
# # normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# close_prices = scaler.fit_transform(close_prices)
# # split data into training set and test set
# train_size = int(len(close_prices) * 0.67)
# test_size = len(close_prices) - train_size

# plot baseline and predictions
#lt.plot(scaler.inverse_transform(close_prices))
#plt.plot(dataframe)
plt.plot(dataframe[:,:])
plt.show()