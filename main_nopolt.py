# Import the libraries
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
import matplotlib.pyplot as plt
import json


with open('setting.json') as f:
    setting = json.load(f)

path = "./dataset/{0}.csv".format(setting["file"])

df = pd.read_csv(path, encoding='UTF-8', low_memory=False)
# # Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])
# Converting the dataframe to a numpy array
dataset = data.values

# Get /Compute the number of rows to train the model on
training_rate = setting["training_rate"]
training_data_len = math.ceil(len(dataset) * training_rate)
front = len(dataset) - training_data_len

# Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
# Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
# Split the data into x_train and y_train data sets

for x in range(setting['runtime']):

    x_train = []
    y_train = []
    for i in range(front, len(train_data)):
        x_train.append(train_data[i-front:i, 0])
        y_train.append(train_data[i, 0])
    # Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    batch_size = 1
    epochs = setting["epochs"]

    # SLR model
    model_SLR = Sequential()
    model_SLR.add(Dense(units=1))
    # Compile the model
    model_SLR.compile(optimizer='adam', loss='mean_squared_error')
    model_SLR.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    # Reshape the data into the shape accepted by the RNN,LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # Build the LSTM network model

    model_LSTM = Sequential()
    model_LSTM.add(LSTM(units=100, return_sequences=True,
                        input_shape=(x_train.shape[1], 1)))
    model_LSTM.add(LSTM(units=100, return_sequences=False))
    model_LSTM.add(Dense(units=25))
    model_LSTM.add(Dense(units=1))
    # Compile the model
    model_LSTM.compile(optimizer='adam', loss='mean_squared_error')

    model_RNN = Sequential()
    model_RNN.add(SimpleRNN(units=100, return_sequences=True,
                            input_shape=(x_train.shape[1], 1)))
    model_RNN.add(SimpleRNN(units=100, return_sequences=False))
    model_RNN.add(Dense(units=25))
    model_RNN.add(Dense(units=1))
    # Compile the model
    model_RNN.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model_LSTM.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    model_RNN.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    # Test data set
    test_data = scaled_data[training_data_len - front:, :]
    # Create the x_test and y_test data sets
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(front, len(test_data)):
        x_test.append(test_data[i-front:i, 0])
    # Convert x_test to a numpy array
    x_test = np.array(x_test)

    predictions_SLR = model_SLR.predict(x_test)
    predictions_SLR = scaler.inverse_transform(predictions_SLR)
    # Reshape the data into the shape accepted by the RNN, LSTM
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # Getting the models predicted Close values
    predictions_LSTM = model_LSTM.predict(x_test)
    predictions_LSTM = scaler.inverse_transform(predictions_LSTM)

    predictions_RNN = model_RNN.predict(x_test)
    predictions_RNN = scaler.inverse_transform(predictions_RNN)

    # Plot/Create the data for the graph
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['LSTM'] = predictions_LSTM
    valid['RNN'] = predictions_RNN
    valid['SLR'] = predictions_SLR

    result_path = "./result/{0}/{1}-{2}-{3}.csv".format(
        setting["file"], setting["training_rate"], setting["epochs"], x)
    valid.to_csv(result_path)

    # Calculate/Get the value of RMSE
    rmse_LSTM = np.sqrt(np.mean(((predictions_LSTM - y_test)**2)))
    rmse_RNN = np.sqrt(np.mean(((predictions_RNN - y_test)**2)))
    rmse_SLR = np.sqrt(np.mean(((predictions_SLR - y_test)**2)))

    log_path = "./result/{0}/rmse_log.csv".format(setting["file"])

    with open(log_path, 'a') as log_file:
        log = "\n{0},{1:.4f},{2:.4f},{3:.4f}".format(
            x, rmse_LSTM, rmse_RNN, rmse_SLR)
        log_file.write(log)
