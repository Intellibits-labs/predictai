#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

from time_series_to_supervised import *

path = os.path.dirname(os.path.abspath(__file__))
file_loc = os.path.join(path, 'dataset.xls')
df = pd.read_excel(file_loc, nrows=840)

n_sample = 588

values = df.values
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = time_series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[6,7,8,9,10]], axis=1, inplace=True)
values = reframed.values
train = values[:n_sample, :]
test = values[n_sample:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


# design network
model = Sequential()
model.add(LSTM(32, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=300, batch_size=32, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

yhat_copy = np.repeat(yhat,6,axis=-1)

y_pred_future = scaler.inverse_transform(yhat_copy)[:,5]
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,5]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,5]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, y_pred_future))
print('Test RMSE: %.3f' % rmse)


x_tick_values = [1, 61, 121, 181, 241]
x_tick_labels = ['Day 596', 'Day 556', 'Day 616', 'Day 676', 'Day 736']
pyplot.plot(inv_y, label='Actual')
pyplot.plot(y_pred_future, label='Predicted')
pyplot.xticks(x_tick_values, x_tick_labels, rotation='vertical')
pyplot.xlabel("Day")
pyplot.ylabel("Price (in Rupees)")
pyplot.title("TATA MOTORS - Actual vs Predicted")
pyplot.show()