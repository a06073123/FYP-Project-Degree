# Import the libraries
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

df = pd.read_csv("./corn-futures-data/dataset4y.csv",
                 encoding='UTF-8', low_memory=False)
# # Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])
# Converting the dataframe to a numpy array
dataset = data.values

# Get /Compute the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .7)
# Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
# Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []
for i in range(373, len(train_data)):
    x_train.append(train_data[i-373:i, 0])
    y_train.append(train_data[i, 0])
# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

model = Sequential()
model.add(Dense(units=1))
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=10)
# Test data set
test_data = scaled_data[training_data_len - 373:, :]
# Create the x_test and y_test data sets
x_test = []
# Get all of the rows from index 137303 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 137303 = 400 rows of data
y_test = dataset[training_data_len:, :]
for i in range(373, len(test_data)):
    x_test.append(test_data[i-373:i, 0])
# Convert x_test to a numpy array
x_test = np.array(x_test)
# Getting the models predicted MinPrice values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Undo scaling
# Calculate/Get the value of RMSE
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
# Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
