# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:55:30 2021

@author: anann
"""


"""
In this dataset, we can take any column as the input column and output column
Since, for RNN, the output is trained on the previous data
Input column = Output column (except date)
Open Price is the best input to consider
"""

# Main aim is to predict the stock price

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv("Google_Stock_Price_train.csv")

# We need only Open price column
train_set = data_train.iloc[:,1:2].values

""" FEATURE SCALING """
# Min-Max scaler to scale all these values between 0 and 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
scaled_training = sc.fit_transform(train_set)

# Here, we don't have x_train, y_train as i/ps and o/ps
# So from the input column itself you will have to create such divisions

# For example: Taking 3 continuous data as input and the 4th data as the output
# For example: In a dataset of 12 rows: for i/p- 1,2,3, o/p- 4 and for i/p- 2,3,4, o/p-5..till ip- 11,12,1, op-1

""" CREATING DATA WITH TIMESTEPS 
- LSTMs expect our data to be in a specific format, usually a 3D array.  
- We start by creating data in 60 timesteps and converting it into an array using NumPy. 
- Next, we convert the data into a 3D dimension array with X_train samples, 60 timestamps, and one feature at each step.
"""

# For this dataset consider 2 months data as input
# 60 inputs and 61st as output. Thus, 0..59-ip and 60-op
# 60 columns for x_train and 1 column for y_train
x_train = []
y_train = []

# for iteration-1: x_train- [60-60:60,0] (0..59 rows taken and converted into 60 columns for the list) 
# and y_train- [60,0] (60th row converted into a list)
for i in range(60,1258):
    x_train.append(scaled_training[i-60:i,0])
    y_train.append(scaled_training[i,0])
    
# Convert the list into array
x_train, y_train = np.array(x_train), np.array(y_train)

x_train.shape
y_train.shape

# The shape of x(2D) and y(1D) are not according to LSTM
# LSTM needs 3D data for input
print(x_train.ndim, y_train.ndim)

# Reshaping
# Here, shape[0] = no of rows and shape[1] = no of columns
# (x_train,(1198,60,1))
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Initialize model
model = Sequential()

# Add Input layer

# num_units is the number of hidden units whose activations get sent forward to the next time step.
# return_sequence = True means you want to add another LSTM layer
# return_sequences=True which determines whether to return the last output in the output sequence, or the full sequence
# input_shape as the shape of our training set.
model.add(LSTM ( units = 60, return_sequences=True, input_shape=(60,1)))
model.add(Dropout(0.2))
model.add(LSTM ( units = 60, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM ( units = 60, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM ( units = 60))
model.add(Dropout(0.2))

# Add Output layer
model.add(Dense(units=1))

# Compile the model
model.compile("rmsprop", loss = "mean_squared_error")

# Train the model
model.fit(x_train, y_train, epochs = 10)

# Import Test data
data_test = pd.read_csv("Google_Stock_Price_Test.csv")

# Taking only the open price column from test data
real_stock_price = data_test.iloc[:,1:2] # = y_test

# Concatenating data from train set into test set
data_total = pd.concat((data_train["Open"], data_test["Open"]), axis=0)

# We have to increase the rows in test set, since there is only 1 col which has to be converted into 60 cols to make it like train shape
# (x_train and y_train shape should be same as x_test and y_test shape) 
# So, we will be using some data from train set and adding to test set

# inputs for test data: We have to make them on our own
# .values to convert into array
# 1278-20-60 to all rows - 1198 to all rows - 1198:1278 (80 rows)
# (input of 60 days, timestep of 60)
inputs = data_total[len(data_total) - len(data_test) - 60:].values
inputs = inputs.reshape(-1,1) # reshape into 80 rows and 1 columns

# Using Min-Max scaler to fit the inputs
inputs = sc.fit_transform(inputs)

x_test = []
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])

# Convert x_test into 3D
x_test = np.array(x_test)
x_test = np.reshape(x_test,(20,60,1))

# Prediction
ypred = model.predict(x_test)

# After making the predictions we use inverse_transform to get back the stock prices in normal readable format.
# scaling ypred to normal
ypred = sc.inverse_transform(ypred)

# Plotting a graph between actual and predicted stock price
plt.plot(real_stock_price, color="red", label = "actual stock price")
plt.plot(ypred, color = "blue", label = "predicted stock price")
plt.title("Stock Price Prediction")
plt.legend()
plt.show()

# Testing for individual values

# series of 60 values for testing
test =  [343,456,756,678,786,456,567,343,767,123,343,456,756,678,786,456,567,343,767,123,343,456,756,678,786,456,567,343,767,123,343,456,756,678,786,456,567,343,767,123,343,456,756,678,786,456,567,343,767,123,343,456,756,678,786,456,567,343,767,123]
# Performing all the previous functions
test = np.array(test) 
test = test.reshape(-1,1)
test = np.reshape(test, (1,60,1))
yp = model.predict (test)
yp = sc.inverse_transform(yp)
print(yp)