# Import Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler(feature_range=(0, 1))

# Input Data
df = pd.read_csv('NSE-TATAGLOBAL.csv')

# Convert to df with just date and closing price
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)), columns=['Date', 'Close']) #Initialise new dataframe

for i in range(0,len(data)):
    new_data['Date'][i] = i # Just have time steps
    new_data['Close'][i] = data['Close'][i]

# Train test split
train = new_data[:1700]
test = new_data[1700:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_test = test.drop('Close', axis=1)
y_test = test['Close']

# Use KNN
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale data
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)

# Find best params with gridsearch using cross-validation
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

# Fit and Predict
model.fit(x_train,y_train)
preds = model.predict(x_test)
rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))
print(rms)

# Plot
test['Predictions'] = 0
test['Predictions'] = preds
plt.plot(test[['Close', 'Predictions']])
plt.plot(train['Close'])
plt.show()