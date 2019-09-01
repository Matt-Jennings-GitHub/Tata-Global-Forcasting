# Import Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

# Input Data
df = pd.read_csv('NSE-TATAGLOBAL.csv')

# Convert to df with just date and closing price
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)), columns=['Date', 'Close']) #Initialise new dataframe

for i in range(0,len(data)):
    new_data['Date'][i] = i # Just have time steps
    new_data['Close'][i] = data['Close'][i]

new_data['Date'] = pd.to_datetime(new_data.Date,format='%Y-%m-%d')
new_data.index = new_data['Date']

new_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True) #Fb prophet takes these specific inputs

# Train and test
train = new_data[:1700]
test = new_data[1700:]

# Fit
model = Prophet()
model.fit(train)

# Predicitions
close_prices = model.make_future_dataframe(periods=len(test))
forecast = model.predict(close_prices)

forecast_valid = forecast['yhat'][1700:]
rms=np.sqrt(np.mean(np.power((np.array(valid['y'])-np.array(forecast_valid)),2)))

# Plot
valid['Predictions'] = 0
valid['Predictions'] = forecast_valid.values

plt.plot(train['y'])
plt.plot(valid[['y', 'Predictions']])
plt.show()