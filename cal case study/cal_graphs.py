import pandas as pd
import numpy as np  
from torch import tensor as t
import seaborn as sns
import matplotlib.pyplot as plt

#creates a histogram
for col in df.columns:
    plt.figure(figsize=(10,8))
    plt.hist(df[col],bins=20)
    plt.xlabel(f"{col}")
    plt.ylabel(f"Count")
    plt.title(f"Histogram for {col}")
    plt.savefig(f"casestudy california/{col}")

#creates scatter plot

plt.figure(figsize=(10,8))
df = df.groupby(["median_income"])['median_house_value'].mean()
plt.plot(df.index,df.values,marker='o')
plt.grid(True)
plt.xlabel("median_income")
plt.ylabel("median_house_value")
plt.title("Scatter plot b/w Median house value and income")

avg_house_value_by_age = df.groupby('housing_median_age')['median_house_value'].mean()

plt.figure(figsize=(10, 6))
plt.plot(avg_house_value_by_age.index, avg_house_value_by_age.values, marker='o')
plt.title('Average Median House Value by Housing Median Age')
plt.xlabel('Housing Median Age')
plt.ylabel('Average Median House Value')
plt.grid(True)
plt.show()

plt.figure(figsize=(10,8))
plt.scatter(df['population'],df['median_income'])
plt.xlabel("population")
plt.ylabel("median income")
plt.show()