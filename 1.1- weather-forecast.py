import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df=pd.read_csv("../input/weather-prediction/seattle-weather.csv")

df.shape

df.head()

df.describe()

df.info()

df.isnull().sum()

# Data Pre-Processing:

from sklearn.preprocessing import LabelEncoder

df['weather']=LabelEncoder().fit_transform(df['weather'])

# Data Visualization:

fig, axes=plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), ax=axes)

#Machine Learning Model for Prediction of Weather...

from sklearn.model_selection import train_test_split
features=["precipitation", "temp_max", "temp_min", "wind"]
X=df[features]
y=df.weather
train_X, test_X, train_y, test_y = train_test_split(X, y,random_state = 0)

#Decision Tree Regressor:

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

model1=DecisionTreeRegressor(random_state=1)
model1.fit(train_X, train_y)
pred1=model1.predict(test_X)
print("Mean Absolute Error: %f" %(mean_absolute_error(test_y, pred1)))


#Random Forest Regressor:

from sklearn.ensemble import RandomForestRegressor

model2=RandomForestRegressor(random_state=1)
model2.fit(train_X, train_y)
pred2=model2.predict(test_X)
print("Mean Absolute Error: %f" %(mean_absolute_error(test_y, pred2)))
    

# Extreme Gradient Regressor:

from xgboost import XGBRegressor

model3= XGBRegressor(n_estimators=100, learning_rate=0.04)
model3.fit(train_X, train_y)
pred3=model3.predict(test_X)
print("Mean Absolute Error: %f" %(mean_absolute_error(test_y, pred3)))