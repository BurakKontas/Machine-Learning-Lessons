import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv("50_Startups.csv")

# Splitting into X and Y
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

# X = np.delete(X, 0, 1)

# Splitting datasets into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training the model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Y_train)

# Predict the test set results
Y_pred = lr.predict(X_test)
np.set_printoptions(precision=2)

vercital_Y_pred = Y_pred.reshape(len(Y_pred), 1)
vertical_Y_test = Y_test.reshape(len(Y_test), 1)

concated = np.concatenate((vercital_Y_pred, vertical_Y_test), 1)
print(concated)

# Plotting the concated array
plt.plot(concated[:, 0], label='Predicted Values', color="red")
plt.plot(concated[:, 1], label='Actual Values', color="blue")
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.title('Predicted vs Actual Values')
plt.grid()
plt.legend()
plt.show()

# Single prediction
result = lr.predict([X_test[0]])
print(result)

# The final regression equation
print(lr.coef_) # 8.66e+01 -8.73e+02  7.86e+02  7.73e-01  3.29e-02  3.66e-02]
print(lr.intercept_) # 42467.52924853325

# Profit = 86.6 × Dummy State 1 − 873 × Dummy State 2 + 786 × Dummy State 3 + 0.773 × R&D Spend + 0.0329 × Administration + 0.0366 × Marketing Spend + 42467.53

