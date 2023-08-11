import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Social_Network_Ads.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Splitting dataset into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 0)
lr.fit(X_train, Y_train)

# Predict a new result
transformed_value = sc.transform([[20,60000]])
prediction = lr.predict(transformed_value)

# Predicting the Test set results
test_prediction = lr.predict(X_test)
comparison = np.concatenate((test_prediction.reshape(len(test_prediction),1), Y_test.reshape(len(Y_test), 1)), 1)

# Making the confusion matris
from sklearn.metrics import confusion_matrix, accuracy_score
matrix = confusion_matrix(Y_test, test_prediction)
score = accuracy_score(Y_test, test_prediction)
print(score)