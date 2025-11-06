import pandas as pd
import numpy as np  
from torch import tensor as t
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('housing.csv')
# data = t(df.values)
# print(data)
df.head()
df.info()
df.fillna(0,inplace=True)   
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


 #creates a histogram
# for col in df.columns:
#     plt.figure(figsize=(10,8))
#     plt.hist(df[col],bins=20)
#     plt.xlabel(f"{col}")
#     plt.ylabel(f"Count")
#     plt.title(f"Histogram for {col}")
#     plt.savefig(f"casestudy california/{col}")

#creates scatter plot

# plt.figure(figsize=(10,8))
# df = df.groupby(["median_income"])['median_house_value'].mean()
# plt.plot(df.index,df.values,marker='o')
# plt.grid(True)
# plt.xlabel("median_income")
# plt.ylabel("median_house_value")
# plt.title("Scatter plot b/w Median house value and income")

# avg_house_value_by_age = df.groupby('housing_median_age')['median_house_value'].mean()

# plt.figure(figsize=(10, 6))
# plt.plot(avg_house_value_by_age.index, avg_house_value_by_age.values, marker='o')
# plt.title('Average Median House Value by Housing Median Age')
# plt.xlabel('Housing Median Age')
# plt.ylabel('Average Median House Value')
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(10,8))
# plt.scatter(df['population'],df['median_income'])
# plt.xlabel("population")
# plt.ylabel("median income")
# plt.show()

