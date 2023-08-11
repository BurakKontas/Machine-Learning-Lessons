import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values


# Training the Decision Tree regression model
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X, Y)

# Visualising the Decision Tree Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

prediction = dtr.predict(X_grid)

plt.scatter(X, Y, color="red")
plt.plot(X_grid, prediction, color="blue")
plt.title("Truth or Bluff (Decision Tree)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()