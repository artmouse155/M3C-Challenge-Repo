# Import modules for handling datasets 
import pandas as pd

# Library to handle numbers
import numpy as np

# Plot using Matlab's matplotlib library
import matplotlib.pyplot as plt

# Machine learning imports
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Import warnings to can ignore harmless warnings when running code
import warnings
warnings.filterwarnings('ignore')

# Constants to guide our Random Forest Regression
CITIES = ["albuquerque", "seattle"]
CITY = CITIES[0]
FILE = CITY + ".csv"
PREDICTED_DATA = CITY.title() + ' Houses on Market'
RAND_NUM = 0

# Read data from CSV file format
df = pd.read_csv(FILE)

# Define our features and output variable
X = df.drop(columns=[PREDICTED_DATA])
y = df[PREDICTED_DATA]

# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators=10, random_state=RAND_NUM, oob_score=True, max_features=3)

# Fit the regressor with x and y data
regressor.fit(X, y)

# Making predictions on the same data or new data
predictions = regressor.predict(X)
 
# Evaluating the model using mean squared error
mse = mean_squared_error(y, predictions)
print(f'Mean Squared Error: {mse}')

# Evaluating the model using R Squared
r2 = r2_score(y, predictions)
print(f'R-squared: {r2}')

# Format data for prediction
X_grid = np.arange(min(X["Year"]),max(X["Year"]))
X_grid = X_grid.reshape(len(X_grid),1)
   
# Make our prediction
y_predict = regressor.predict(X[:-1])

# Plot our data
plt.plot(X_grid, y_predict,color='green')
plt.title(CITY.title() + " Houses On Market Using Random Forest Regression")
plt.xlabel('Year')
plt.ylabel(PREDICTED_DATA)
plt.show()