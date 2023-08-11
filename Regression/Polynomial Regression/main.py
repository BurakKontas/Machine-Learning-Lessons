import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import and split dataset
dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values


# # Split data into test and train sets
# from sklearn.model_selection import train_test_split

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training linear regression model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X, Y)
lr_predict = lr.predict(X)

# Training polynomial regression model
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=5)
X_poly = pf.fit_transform(X)

pr = LinearRegression() # Polynomial Regression Model
pr.fit(X_poly, Y)
pr_predict = pr.predict(X_poly)

# Visualising the results
plt.scatter(X, Y, color="red")
plt.plot(X, lr_predict, color="blue", label='Linear Regression')
plt.plot(X, pr_predict, color="green", label='Polynomial Regression')
plt.title("Polynomial vs Simple")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Predicting with both models
lr_result = lr.predict([[6.5]])
pr_result = pr.predict(pf.fit_transform([[6.5]]))

print("Simple Linear Result:", lr_result)
print("Polynomial Linear Result:", pr_result)