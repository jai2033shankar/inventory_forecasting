#!/usr/bin/env python
# coding: utf-8

# In[7]:


import scipy

print('scipy: %s' % scipy.__version__)
# numpy
import numpy

print('numpy: %s' % numpy.__version__)
# matplotlib
import matplotlib

print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas

print('pandas: %s' % pandas.__version__)
# statsmodels
import statsmodels

print('statsmodels: %s' % statsmodels.__version__)
# scikit-learn
import sklearn

print('sklearn: %s' % sklearn.__version__)

# In[13]:


from pandas import read_csv

series = read_csv('inv.csv', header=0, index_col=0, parse_dates=True,
                  squeeze=True)
split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', header=False)
validation.to_csv('validation.csv', header=False)

# In[14]:


# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]

# In[16]:
#
#
#
# from pandas import read_csv
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# # load data
# series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# # prepare data
# X = series.values
# X = X.astype('float32')
# train_size = int(len(X) * 0.50)
# train, test = X[0:train_size], X[train_size:]
# # walk-forward validation
# history = [x for x in train]
# predictions = list()
# for i in range(len(test)):
# 	# predict
# 	yhat = history[-1]
# 	predictions.append(yhat)
# 	# observation
# 	obs = test[i]
# 	history.append(obs)
# 	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# # report performance
# mse = mean_squared_error(test, predictions)
# rmse = sqrt(mse)
# print('RMSE: %.3f' % rmse)
#
#
# # In[17]:
#
#
#
# from pandas import read_csv
# series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# print(series.describe())
#
#
# # In[18]:
#
#
#
# from pandas import read_csv
# from matplotlib import pyplot
# series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# series.plot()
# pyplot.show()
#
#
# # In[19]:
#
#
#
# from pandas import read_csv
# from pandas import DataFrame
# from pandas import Grouper
# from matplotlib import pyplot
# series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# groups = series['1964':'1970'].groupby(Grouper(freq='A'))
# years = DataFrame()
# pyplot.figure()
# i = 1
# n_groups = len(groups)
# for name, group in groups:
# 	pyplot.subplot((n_groups*100) + 10 + i)
# 	i += 1
# 	pyplot.plot(group)
# pyplot.show()
#
#
# # In[20]:
#
#
#
# from pandas import read_csv
# from matplotlib import pyplot
# series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# pyplot.figure(1)
# pyplot.subplot(211)
# series.hist()
# pyplot.subplot(212)
# series.plot(kind='kde')
# pyplot.show()
#
#
# # In[21]:
#
#
#
# from pandas import read_csv
# from pandas import DataFrame
# from pandas import Grouper
# from matplotlib import pyplot
# series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# groups = series['1964':'1970'].groupby(Grouper(freq='A'))
# years = DataFrame()
# for name, group in groups:
# 	years[name.year] = group.values
# years.boxplot()
# pyplot.show()
#
#
# # In[22]:
#
#
#
# from pandas import read_csv
# from pandas import Series
# from statsmodels.tsa.stattools import adfuller
# from matplotlib import pyplot
#
# # create a differenced series
# def difference(dataset, interval=1):
# 	diff = list()
# 	for i in range(interval, len(dataset)):
# 		value = dataset[i] - dataset[i - interval]
# 		diff.append(value)
# 	return Series(diff)
#
# series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# X = series.values
# X = X.astype('float32')
# # difference data
# months_in_year = 12
# stationary = difference(X, months_in_year)
# stationary.index = series.index[months_in_year:]
# # check if stationary
# result = adfuller(stationary)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
# 	print('\t%s: %.3f' % (key, value))
# # save
# stationary.to_csv('stationary.csv', header=False)
# # plot
# stationary.plot()
# pyplot.show()
#
#
# # In[23]:
#
#
#
# # invert differenced value
# def inverse_difference(history, yhat, interval=1):
# 	return yhat + history[-interval]
#
#
# # In[24]:
#
#
# from pandas import read_csv
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.graphics.tsaplots import plot_pacf
# from matplotlib import pyplot
# series = read_csv('stationary.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# pyplot.figure()
# pyplot.subplot(211)
# plot_acf(series, ax=pyplot.gca())
# pyplot.subplot(212)
# plot_pacf(series, ax=pyplot.gca())
# pyplot.show()
#
#
# # In[25]:
#
#
# from pandas import read_csv
# from sklearn.metrics import mean_squared_error
# from statsmodels.tsa.arima.model import ARIMA
# from math import sqrt
#
# # create a differenced series
# def difference(dataset, interval=1):
# 	diff = list()
# 	for i in range(interval, len(dataset)):
# 		value = dataset[i] - dataset[i - interval]
# 		diff.append(value)
# 	return diff
#
# # invert differenced value
# def inverse_difference(history, yhat, interval=1):
# 	return yhat + history[-interval]
#
# # load data
# series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# # prepare data
# X = series.values
# X = X.astype('float32')
# train_size = int(len(X) * 0.50)
# train, test = X[0:train_size], X[train_size:]
# # walk-forward validation
# history = [x for x in train]
# predictions = list()
# for i in range(len(test)):
# 	# difference data
# 	months_in_year = 12
# 	diff = difference(history, months_in_year)
# 	# predict
# 	model = ARIMA(diff, order=(1,1,1))
# 	model_fit = model.fit()
# 	yhat = model_fit.forecast()[0]
# 	yhat = inverse_difference(history, yhat, months_in_year)
# 	predictions.append(yhat)
# 	# observation
# 	obs = test[i]
# 	history.append(obs)
# 	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# # report performance
# rmse = sqrt(mean_squared_error(test, predictions))
# print('RMSE: %.3f' % rmse)
#
#
# # In[26]:
#
#
#
# # grid search ARIMA parameters for time series
# import warnings
# from pandas import read_csv
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# import numpy
#
# # create a differenced series
# def difference(dataset, interval=1):
# 	diff = list()
# 	for i in range(interval, len(dataset)):
# 		value = dataset[i] - dataset[i - interval]
# 		diff.append(value)
# 	return numpy.array(diff)
#
# # invert differenced value
# def inverse_difference(history, yhat, interval=1):
# 	return yhat + history[-interval]
#
# # evaluate an ARIMA model for a given order (p,d,q) and return RMSE
# def evaluate_arima_model(X, arima_order):
# 	# prepare training dataset
# 	X = X.astype('float32')
# 	train_size = int(len(X) * 0.50)
# 	train, test = X[0:train_size], X[train_size:]
# 	history = [x for x in train]
# 	# make predictions
# 	predictions = list()
# 	for t in range(len(test)):
# 		# difference data
# 		months_in_year = 12
# 		diff = difference(history, months_in_year)
# 		model = ARIMA(diff, order=arima_order)
# 		model_fit = model.fit()
# 		yhat = model_fit.forecast()[0]
# 		yhat = inverse_difference(history, yhat, months_in_year)
# 		predictions.append(yhat)
# 		history.append(test[t])
# 	# calculate out of sample error
# 	rmse = sqrt(mean_squared_error(test, predictions))
# 	return rmse
#
# # evaluate combinations of p, d and q values for an ARIMA model
# def evaluate_models(dataset, p_values, d_values, q_values):
# 	dataset = dataset.astype('float32')
# 	best_score, best_cfg = float("inf"), None
# 	for p in p_values:
# 		for d in d_values:
# 			for q in q_values:
# 				order = (p,d,q)
# 				try:
# 					rmse = evaluate_arima_model(dataset, order)
# 					if rmse < best_score:
# 						best_score, best_cfg = rmse, order
# 					print('ARIMA%s RMSE=%.3f' % (order,rmse))
# 				except:
# 					continue
# 	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
#
# # load dataset
# series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# # evaluate parameters
# p_values = range(0, 7)
# d_values = range(0, 3)
# q_values = range(0, 7)
# warnings.filterwarnings("ignore")
# evaluate_models(series.values, p_values, d_values, q_values)
#
#
# # In[27]:
#
#
# # summarize ARIMA forecast residuals
# from pandas import read_csv
# from pandas import DataFrame
# from statsmodels.tsa.arima.model import ARIMA
# from matplotlib import pyplot
#
# # create a differenced series
# def difference(dataset, interval=1):
# 	diff = list()
# 	for i in range(interval, len(dataset)):
# 		value = dataset[i] - dataset[i - interval]
# 		diff.append(value)
# 	return diff
#
# # invert differenced value
# def inverse_difference(history, yhat, interval=1):
# 	return yhat + history[-interval]
#
# # load data
# series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# # prepare data
# X = series.values
# X = X.astype('float32')
# train_size = int(len(X) * 0.50)
# train, test = X[0:train_size], X[train_size:]
# # walk-forward validation
# history = [x for x in train]
# predictions = list()
# for i in range(len(test)):
# 	# difference data
# 	months_in_year = 12
# 	diff = difference(history, months_in_year)
# 	# predict
# 	model = ARIMA(diff, order=(0,0,1))
# 	model_fit = model.fit()
# 	yhat = model_fit.forecast()[0]
# 	yhat = inverse_difference(history, yhat, months_in_year)
# 	predictions.append(yhat)
# 	# observation
# 	obs = test[i]
# 	history.append(obs)
# # errors
# residuals = [test[i]-predictions[i] for i in range(len(test))]
# residuals = DataFrame(residuals)
# print(residuals.describe())
# # plot
# pyplot.figure()
# pyplot.subplot(211)
# residuals.hist(ax=pyplot.gca())
# pyplot.subplot(212)
# residuals.plot(kind='kde', ax=pyplot.gca())
# pyplot.show()
#
#
# # In[28]:
#
#
#
# # plots of residual errors of bias corrected forecasts
# from pandas import read_csv
# from pandas import DataFrame
# from statsmodels.tsa.arima.model import ARIMA
# from matplotlib import pyplot
# from sklearn.metrics import mean_squared_error
# from math import sqrt
#
# # create a differenced series
# def difference(dataset, interval=1):
# 	diff = list()
# 	for i in range(interval, len(dataset)):
# 		value = dataset[i] - dataset[i - interval]
# 		diff.append(value)
# 	return diff
#
# # invert differenced value
# def inverse_difference(history, yhat, interval=1):
# 	return yhat + history[-interval]
#
# # load data
# series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# # prepare data
# X = series.values
# X = X.astype('float32')
# train_size = int(len(X) * 0.50)
# train, test = X[0:train_size], X[train_size:]
# # walk-forward validation
# history = [x for x in train]
# predictions = list()
# bias = 165.904728
# for i in range(len(test)):
# 	# difference data
# 	months_in_year = 12
# 	diff = difference(history, months_in_year)
# 	# predict
# 	model = ARIMA(diff, order=(0,0,1))
# 	model_fit = model.fit()
# 	yhat = model_fit.forecast()[0]
# 	yhat = bias + inverse_difference(history, yhat, months_in_year)
# 	predictions.append(yhat)
# 	# observation
# 	obs = test[i]
# 	history.append(obs)
# # report performance
# rmse = sqrt(mean_squared_error(test, predictions))
# print('RMSE: %.3f' % rmse)
# # errors
# residuals = [test[i]-predictions[i] for i in range(len(test))]
# residuals = DataFrame(residuals)
# print(residuals.describe())
# # plot
# pyplot.figure()
# pyplot.subplot(211)
# residuals.hist(ax=pyplot.gca())
# pyplot.subplot(212)
# residuals.plot(kind='kde', ax=pyplot.gca())
# pyplot.show()
#
#
# # In[29]:
#
#
#
# # ACF and PACF plots of residual errors of bias corrected forecasts
# from pandas import read_csv
# from pandas import DataFrame
# from statsmodels.tsa.arima.model import ARIMA
# from matplotlib import pyplot
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.graphics.tsaplots import plot_pacf
#
# # create a differenced series
# def difference(dataset, interval=1):
# 	diff = list()
# 	for i in range(interval, len(dataset)):
# 		value = dataset[i] - dataset[i - interval]
# 		diff.append(value)
# 	return diff
#
# # invert differenced value
# def inverse_difference(history, yhat, interval=1):
# 	return yhat + history[-interval]
#
# # load data
# series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# # prepare data
# X = series.values
# X = X.astype('float32')
# train_size = int(len(X) * 0.50)
# train, test = X[0:train_size], X[train_size:]
# # walk-forward validation
# history = [x for x in train]
# predictions = list()
# for i in range(len(test)):
# 	# difference data
# 	months_in_year = 12
# 	diff = difference(history, months_in_year)
# 	# predict
# 	model = ARIMA(diff, order=(0,0,1))
# 	model_fit = model.fit()
# 	yhat = model_fit.forecast()[0]
# 	yhat = inverse_difference(history, yhat, months_in_year)
# 	predictions.append(yhat)
# 	# observation
# 	obs = test[i]
# 	history.append(obs)
# # errors
# residuals = [test[i]-predictions[i] for i in range(len(test))]
# residuals = DataFrame(residuals)
# print(residuals.describe())
# # plot
# pyplot.figure()
# pyplot.subplot(211)
# plot_acf(residuals, ax=pyplot.gca())
# pyplot.subplot(212)
# plot_pacf(residuals, ax=pyplot.gca())
# pyplot.show()
#
#
# # In[30]:
#
#
#
# # save finalized model
# from pandas import read_csv
# from statsmodels.tsa.arima.model import ARIMA
# import numpy
#
# # create a differenced series
# def difference(dataset, interval=1):
# 	diff = list()
# 	for i in range(interval, len(dataset)):
# 		value = dataset[i] - dataset[i - interval]
# 		diff.append(value)
# 	return diff
#
# # load data
# series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# # prepare data
# X = series.values
# X = X.astype('float32')
# # difference data
# months_in_year = 12
# diff = difference(X, months_in_year)
# # fit model
# model = ARIMA(diff, order=(0,0,1))
# model_fit = model.fit()
# # bias constant, could be calculated from in-sample mean residual
# bias = 165.904728
# # save model
# model_fit.save('model.pkl')
# numpy.save('model_bias.npy', [bias])
#
#
# # In[31]:
#
#
#
# # load finalized model and make a prediction
# from pandas import read_csv
# from statsmodels.tsa.arima.model import ARIMAResults
# import numpy
#
# # invert differenced value
# def inverse_difference(history, yhat, interval=1):
# 	return yhat + history[-interval]
#
# series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# months_in_year = 12
# model_fit = ARIMAResults.load('model.pkl')
# bias = numpy.load('model_bias.npy')
# yhat = float(model_fit.forecast()[0])
# yhat = bias + inverse_difference(series.values, yhat, months_in_year)
# print('Predicted: %.3f' % yhat)


# In[34]:


# load and evaluate the finalized model on the validation dataset
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def trythis():
    # load and prepare datasets
    dataset = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
    X = dataset.values.astype('float32')
    history = [x for x in X]
    months_in_year = 12
    validation = read_csv('validation.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
    y = validation.values.astype('float32')
    # load model
    model_fit = ARIMAResults.load('model.pkl')
    bias = numpy.load('model_bias.npy')
    # make first prediction
    predictions = list()
    sendist = list()
    yhat = float(model_fit.forecast()[0])
    yhat = bias + inverse_difference(history, yhat, months_in_year)
    predictions.append(yhat)
    history.append(y[0])
    sendist.append(round(yhat[0],3))
    print('>Predicted=%.3f, ' % (yhat))
    # rolling forecasts
    for i in range(1, len(y)):
        # difference data
        months_in_year = 12
        diff = difference(history, months_in_year)
        # predict
        model = ARIMA(diff, order=(0, 0, 1))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        yhat = bias + inverse_difference(history, yhat, months_in_year)

        sendist.append(round(yhat[0],3))
        predictions.append(yhat)

        # observation
        obs = y[i]
        history.append(obs)
        # print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
        print('>Predicted=%.3f' % (yhat))
    # report performance
    rmse = sqrt(mean_squared_error(y, predictions))
    print('RMSE: %.3f' % rmse)
    print(sendist)
    return sendist
    # pyplot.plot(y)
    # pyplot.plot(predictions, color='red')
    # pyplot.show()

# In[ ]:
