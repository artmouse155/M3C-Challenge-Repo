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

# Function to Generate a random color for the Monte Carlo simulation
def random_hex_color():
    # Generate three random integers for RGB values
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    
    # Convert RGB to hexadecimal format
    hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
    
    return hex_color

# Constants to guide Random Forest Regression
EXPORT_NAME = "economy recession"

# Seattle housing price adjustment
s_h_adj = (10 ** -1)

# Seattle population adjustment
s_p_adj = (10 ** -1)

# Seattle population adjustment
s_i_adj = (10 ** -1)

# Seattle housing price mean and standard deviation of normal probability curve
s_h_mean = 283.829 * s_h_adj
s_h_std = 84.5263

# Seattle population mean and standard deviation of normal probability curve
s_p_mean = 3.2727 * (10 ** 6) * s_p_adj
s_p_std = 3.9601 * (10 ** 5)  

# Seattle inflation mean and standard deviation of normal probability curve
s_i_mean = 2.4963 * s_i_adj
s_i_std = 1.5772

# Constants to guide Random Forest Regression and Monte Carlo Simulation
CITY = "seattle"
FILE = CITY + ".csv"
PREDICTED_DATA = CITY.title() + " Houses on Market"
RAND_NUM = 0
NUM_SIMULATIONS = 50

# Read data from CSV file
df_real = pd.read_csv(FILE)
y = df_real[PREDICTED_DATA]

# Set up data to export
exportedData = pd.DataFrame()

# Set up list to export data about accuracy of model
RMSELIST = []
r2LIST = []

# Run 50 different models
for carlo in range(NUM_SIMULATIONS):

    # Copy dataset so we can use varying parameters
    df = df_real.copy()

    # Apply new parameters
    for i in range(len(df["Seattle Population"])):
        df.at[i, "Seattle Wages"] = np.random.normal(loc=s_i_mean, scale=s_i_std)
        df.at[i, "Seattle House Price"] = np.random.normal(loc=s_h_mean, scale=s_h_std)
        df.at[i, "Seattle Population"] = np.random.normal(loc=s_p_mean, scale=s_p_std)
    df.update(cf)
    X = df.drop(columns=[PREDICTED_DATA])
    
    # Fitting Random Forest Regression to the dataset
    regressor = RandomForestRegressor(n_estimators=10, random_state=RAND_NUM, oob_score=True, max_features=3)
    x = X

    # Fit the regressor with x and y data
    regressor.fit(x, y)

    # Making predictions on the same data or new data
    predictions = regressor.predict(x)
 
    # Evaluating the model using mean squared error
    mse = mean_squared_error(y, predictions)
    RMSELIST.append(mse)

    # Evaluating the model using R Squared
    r2 = r2_score(y, predictions)
    r2LIST.append(r2)

    # Format data for prediction
    X_grid = np.arange(min(X["Year"]),max(X["Year"]))
    X_grid = X_grid.reshape(len(X_grid),1)

    # Make prediction
    y_predict = regressor.predict(X[:-1])

    # Plot prediction
    plt.plot(X_grid, y_predict,color=random_hex_color())

    # Save predicted data points
    exportedData[str(carlo)] = y_predict

# Save Monte Carlo Data for analysis
np.savetxt(EXPORT_NAME+'.csv', exportedData, delimiter=',')

# Plot final Monte Carlo data
plt.title(CITY.title() + " Houses On Market Using Random Forest Regression")
plt.xlabel('Year')
plt.ylabel(PREDICTED_DATA)
plt.show()

# Output Mean MSE and Mean R^2
print("MEAN MEAN SQUARED ERROR:" + str(pd.Series(RMSELIST).mean()))
print("MEAN R^2:" + str(pd.Series(r2LIST).mean()))
#------------------------------------