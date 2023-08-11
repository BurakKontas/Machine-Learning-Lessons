import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values
Y = Y.reshape(len(Y),1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

sc_Y = StandardScaler()
Y = sc_Y.fit_transform(Y)

# Training SVR Model
from sklearn.svm import SVR

svr = SVR(kernel="rbf")
svr.fit(X, Y)

# Predicting a new result (and reverse scale the prediction)
transformed_value = sc_X.transform([[6.5]])
prediction = svr.predict(transformed_value)
prediction = sc_Y.inverse_transform(prediction.reshape(-1, 1))
# print(prediction)

# Visualising the SVR Results
inversed_X = sc_X.inverse_transform(X)
inversed_Y =  sc_Y.inverse_transform(Y)
prediction_X = svr.predict(X).reshape(-1, 1)
inversed_prediction = sc_Y.inverse_transform(prediction_X)

plt.scatter(inversed_X, inversed_Y, color="red")
plt.plot(sc_X.inverse_transform(X), inversed_prediction, color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising the SVR Results (higher resolution and smoother curve)
X_grid = np.arange(min(inversed_X), max(inversed_X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

transformed_X_grid = sc_X.transform(X_grid)
prediction_X = svr.predict(transformed_X_grid).reshape(-1, 1)
inversed_prediction = sc_Y.inverse_transform(prediction_X)

plt.scatter(inversed_X, inversed_Y, color="red")
plt.plot(X_grid, inversed_prediction, color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()