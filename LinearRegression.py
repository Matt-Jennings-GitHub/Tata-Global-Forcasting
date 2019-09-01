# Import Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler(feature_range=(0, 1))

# Input Data
df = pd.read_csv('NSE-TATAGLOBAL.csv')
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date'] #Set Index as date

plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history')

# Convert to df with just date and closing price
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)), columns=['Date', 'Close']) #Initialise new dataframe

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]


# Create New Features to use a independents for linear regression
#from fastai.structured import  add_datepart
#add_datepart(new_data, 'Date')
#new_data.drop('Elapsed', axis=1, inplace=True)  # Elapsed will be the time stamp

for i in range(0,len(new_data)):
    new_data['Year'] = str(new_data['Date'][0])[0:5]
    new_data['Month'] = str(new_data['Date'][0])[6:8]
    new_data['mon_fri'] = 0
    if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
        new_data['mon_fri'][i] = 1
    else:
        new_data['mon_fri'][i] = 0

# Train test split
train_data = new_data[:1700]
test_data = new_data[1700:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_test = test.drop('Close', axis=1)
y_test = test['Close']

# Use Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

#plot
test['Predictions'] = 0
test['Predictions'] = preds

test.index = new_data[987:].index
train.index = new_data[:987].index

plt.plot(train['Close'])
plt.plot(test[['Close', 'Predictions']])