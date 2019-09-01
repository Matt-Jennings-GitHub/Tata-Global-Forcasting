# Import Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM

# Input Data
df = pd.read_csv('NSE-TATAGLOBAL.csv')
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)), columns=['Date', 'Close']) #Initialise new dataframe
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

# Set Index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True) #Remove Labels

# Train and Test data
dataset = new_data.values #Numpy Arrays
train = dataset[0:1700,:]
test = dataset[1700:,:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)): # Take last 60 days as inputs
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

print(x_train.shape)
#x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_train = np.expand_dims(x_train, axis = 2)
print(x_train.shape)

# Define Network
retrain = False
if retrain :
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
    model.save("trained_model.h5")
else:
    model = load_model("trained_model.h5")

# Predicting values, using past 60 from the train data
inputs = new_data[len(new_data) - len(test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)
X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price) # Un-normalise the prediction

rms = np.sqrt(np.mean(np.power((test-closing_price),2)))

# Plot
train = new_data[:1700]
test = new_data[1700:]
test['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(test[['Close','Predictions']],linewidth=1)
plt.show()