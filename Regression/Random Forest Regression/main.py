import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values

# Training the Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100, random_state=0)
rfr.fit(X, Y)

# Predict
prediction = rfr.predict([[6.5]])

# Visualising the Random Forest Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

prediction = rfr.predict(X_grid)

plt.scatter(X, Y, color="red")
plt.plot(X_grid, prediction, color="blue")
plt.title("Truth or Bluff (Decision Tree)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()