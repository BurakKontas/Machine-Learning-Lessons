import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Split dataset into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Simple Linear Regression model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,Y_train)

# Predict the test set results
y_pred = lr.predict(X_test)
y_train_pred = lr.predict(X_train)

# Visulising the training set results
plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, y_train_pred, color="blue")
plt.title('Salary vs Experience (Training set)')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()


# Visualising the test set results
plt.scatter(X_test, Y_test, color="red")
plt.plot(X_train, y_train_pred, color="blue")
plt.title('Salary vs Experience (Test set)')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# Predict one data
result = lr.predict([X_test[0]])
print(result)

# Getting the final linear regression equation with the values of the coefficients
print(lr.coef_) # 9312.57512673
print(lr.intercept_) # 26780.09915062818

# Salary = 9312.57 × YearsExperience + 26780.09