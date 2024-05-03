import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('seattle_homeless.csv',index_col='DATE',parse_dates=True)
df=df.dropna()
print('Shape of data',df.shape)

DATA_NAME = "Total Homeless"
#ARIMA_ORDER = (0,1,0)
ARIMA_ORDER = (5,1,3)
#df[DATA_NAME].plot(figsize=(12,5))

from statsmodels.tsa.stattools import adfuller

def adf_test(dataset):
  dftest = adfuller(dataset, autolag = 'AIC')
  print("1. ADF : ",dftest[0])
  print("2. P-Value : ", dftest[1])
  print("3. Num Of Lags : ", dftest[2])
  print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
  print("5. Critical Values :")
  for key, val in dftest[4].items():
      print("\t",key, ": ", val)

adf_test(df[DATA_NAME])

from pmdarima import auto_arima
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

stepwise_fit = auto_arima(df[DATA_NAME], suppress_warnings=True)           

print(stepwise_fit.summary())

print(df.shape)
train=df.iloc[:-10]
test=df.iloc[-10:]
print(train.shape,test.shape)
print(test.iloc[0],test.iloc[-1])

from statsmodels.tsa.arima.model import ARIMA

model=ARIMA(train[DATA_NAME],order=ARIMA_ORDER)
model=model.fit()
model.summary()

start=len(train)
end=len(train)+len(test)-1
#if the predicted values dont have date values as index, you will have to uncomment the following two commented lines to plot a graph
#index_future_dates=pd.date_range(start='2018-12-01',end='2018-12-30')
pred=model.predict(start=start,end=end,typ='levels').rename('ARIMA Model')
#pred.index=index_future_dates
pred.plot(legend=True)
test[DATA_NAME].plot(legend=True)

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(pred,test[DATA_NAME]))
print("RMSE:" + str(rmse))

aic = model.aic

print("AIC:", aic)

model2=ARIMA(df[DATA_NAME],order=ARIMA_ORDER)
model2=model2.fit()
df.tail()
#index_current_dates=pd.date_range(start='1990-01-01',end='2023-12-01')
index_future_dates=pd.date_range(start='2020-1-1',end='2074-01-01', freq="1Y")
#print(index_future_dates)
pred=model2.predict(start=len(df),end=len(df)+(len(index_future_dates)-1),typ='levels').rename('ARIMA Forecasted Model')
#print(comp_pred)
pred.index=index_future_dates
plt.title("Predictions of Seattle Homeless Population using ARIMA")
plt.xlabel('Year')
plt.ylabel(DATA_NAME)
print(pred)
pred.plot(figsize=(12,5),legend=True)
plt.show()