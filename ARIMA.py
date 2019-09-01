# Import Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyramid.arima import auto_arima

# Input Data
df = pd.read_csv('NSE-TATAGLOBAL.csv')

# Train test split
train = data[:1700]
test = data[1700:]

train = train['Close']
test = test['Close']

#Use Auto ARIMA
model = auto_arima(train, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model.fit(train)

forecast = model.predict(n_periods=400)
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-np.array(forecast['Prediction'])),2)))

plt.plot(train['Close'])
plt.plot(test['Close'])
plt.plot(forecast['Prediction'])

plt.show()