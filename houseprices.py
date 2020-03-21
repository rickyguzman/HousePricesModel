import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv (r'train.csv')
x = df['YearBuilt'].values.reshape(-1,1)
#x = df[['YearBuilt','OverallQual']].values.reshape(-1,1)
y = df['SalePrice'].values.reshape(-1,1)
linear_regressor = LinearRegression()
linear_regressor.fit(x,y)
y_predict = linear_regressor.predict(x)

plt.scatter(x,y)
plt.plot(x,y_predict,color = 'blue')
plt.show()


