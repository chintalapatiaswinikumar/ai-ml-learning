import pandas as pd
import numpy as np  
from torch import tensor as t
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('housing.csv')
# data = t(df.values)
# print(data)
df.head()
df.fillna(0,inplace=True)
df.info()
df.describe()
df.isnull().sum()

# Lets predict median_house_value
# Step 1: Divide the data into traning data and testing data
    #from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 2: Train the model by chosing a readymade learning algo
#     from sklearn.ensemble import RandomForestRegressor
#     model = RandomForestRegressor()
#     model.fit(X_train, y_train)
# Step 3: Test the model on testing data and check the results
    # y_pred = model.predict(X_test)
    # from sklearn.metrics import r2_score, mean_squared_error
    # print("R² Score:", r2_score(y_test, y_pred))
    # print("MSE:", mean_squared_error(y_test, y_pred))

# Step 4: If they are not accurate tune the hyper parameters using some readymade packages to get optimal results
# - Try a different algorithm or tune hyperparameters (model settings).
# - Repeat training and testing until results improve.
# Example:
# Try Gradient Boosting, XGBoost, or adjust parameters like n_estimators, learning_rate, etc.


#Step 1:
#Inputs
x = df.drop("median_house_value",axis =1) # we are dropping this column so that model wont know the value it is going to predict before hand
x = pd.get_dummies(x, columns=['ocean_proximity'], drop_first=True) 
#Output
y = df['median_house_value'] # and training the model with its values on traning dataset

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=42)

print("train data",x_train.shape,y_train.shape)
print("test data",x_test.shape,y_test.shape)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(x_train, y_train)

new_data = pd.DataFrame([{
    'longitude': -122.23,
    'latitude': 37.88,
    'housing_median_age': 41,
    'total_rooms': 880,
    'total_bedrooms': 129,
    'population': 322,
    'households': 126,
    'median_income': 8.3252,
    'ocean_proximity_INLAND': 0,
    'ocean_proximity_ISLAND':0,
    'ocean_proximity_NEAR BAY': 0,
    'ocean_proximity_NEAR OCEAN': 1
    
}])

# Predict to entire test data
y_pred = rf.predict(x_test)

#Predict to my one single data point
# y_pred = rf.predict(new_data)

df1 = pd.DataFrame()
df1["actual"] = y_test
df1['predicted'] = y_pred 

# Evaluate
print("R² Score:", r2_score(y_test, y_pred)) # coefficient of determination which is always <1, if it is nearing to 1 then the fit is good
print("MSE:", mean_squared_error(y_test, y_pred))  #Shows the squared component of error