# Import modules for handling datasets 
import pandas as pd

# Library to handle numbers
import numpy as np

# Plot using Matlab's matplotlib library
import matplotlib.pyplot as plt

# Read data from a CSV file
df=pd.read_csv('alb_homeless.csv',index_col='DATE',parse_dates=True)

# Remove Not A Number Values
df=df.dropna()

# Define data
DATA_NAME = "Total Homeless"
ARIMA_ORDER = (5,1,1)

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

# Define testing and training data
train=df
test=df

# Import ARIMA
from statsmodels.tsa.arima.model import ARIMA

# Fit data to ARIMA
model=ARIMA(train[DATA_NAME],order=ARIMA_ORDER)
model=model.fit()
model.summary()

# Define starting and ending points for prediction
start=len(train)
end=len(train)+len(test)-1

# Predict data with model
pred=model.predict(start='2009-01-01',end='2021-01-01',typ='levels').rename('ARIMA Model')

# Plot predicted data and test data
pred.plot(legend=True)
test[DATA_NAME].plot(legend=True)

# Evaluating the model using mean squared error
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(pred,test[DATA_NAME]))
print("RMSE:" + str(rmse))

# Evaluating the model using AIC
aic = model.aic
print("AIC:", aic)

# Fit an ARIMA model to extrapolate
model2=ARIMA(df[DATA_NAME],order=ARIMA_ORDER)
model2=model2.fit()

# Determine the intervals to extrapolate by
index_future_dates=pd.date_range(start='2021-1-1',end='2074-01-01', freq="1Y")

# Predict using extrapolation model
pred=model2.predict(start=len(df),end=len(df)+(len(index_future_dates)-1),typ='levels').rename('ARIMA Forecasted Model')

# Format Future Dates
pred.index=index_future_dates

# Plot data
plt.title("Predictions of Albuquerque Homeless Population using ARIMA")
plt.xlabel('Year')
plt.ylabel(DATA_NAME)
pred.plot(figsize=(12,5),legend=True)
plt.show()

print(pred)