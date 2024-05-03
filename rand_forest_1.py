import pandas as pd #pip install pandas, then pip install Pyarrow
import matplotlib.pyplot as plt #pip install matplotlib
import sklearn #pip install scikit-learn

#df stands for dataframe
df = pd.read_csv("us_transport.csv")
#print(df)

# Assuming df is your DataFrame
X = df.iloc[:,0].values  #features
print(X)
y = df.iloc[:,2].values  # Target variable

#random model stuff
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
"""
#Check for and handle categorical variables
label_encoder = LabelEncoder()
x_categorical = df.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
x_numerical = df.select_dtypes(exclude=['object']).values
x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values
"""
# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

#not sure what this line does?
x = X.reshape(-1, 1)

# Fit the regressor with x and y data
regressor.fit(x, y)

#------------------------------------
# Evaluating the model
from sklearn.metrics import mean_squared_error, r2_score
 
# Access the OOB Score
oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')
 
# Making predictions on the same data or new data
predictions = regressor.predict(x)
 
# Evaluating the model
mse = mean_squared_error(y, predictions)
print(f'Mean Squared Error: {mse}')
 
r2 = r2_score(y, predictions)
print(f'R-squared: {r2}')
#-------------------------------------
import numpy as np
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)

max_year = 2028

X_grid_predict = np.arange(min(X),max_year,0.01)
X_grid_predict = X_grid_predict.reshape(len(X_grid_predict),1)
   
plt.scatter(X,y, color='blue') #plotting real points

plt.plot(X_grid, regressor.predict(X_grid),color='green') #plotting for predict points
plt.plot(X_grid_predict, regressor.predict(X_grid_predict),color='orange') #plotting for predict points

plt.title("Random Forest Regression Results")
plt.xlabel('Year')
plt.ylabel('Mass Transportation (Billions)')
plt.show()

#------------------------------------