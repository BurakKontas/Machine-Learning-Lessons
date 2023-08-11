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

# KNN
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p = 2)
KNN.fit(X_train, Y_train)

# Predict a new result
transformed_value = sc.transform([[50,90000]])
prediction = KNN.predict(transformed_value)

# Predicting the Test set results
test_prediction = KNN.predict(X_test)
comparison = np.concatenate((test_prediction.reshape(len(test_prediction),1), Y_test.reshape(len(Y_test), 1)), 1)

# Making the confusion matris
from sklearn.metrics import confusion_matrix, accuracy_score
matrix = confusion_matrix(Y_test, test_prediction)
score = accuracy_score(Y_test, test_prediction)
print(score)

# Extract the training model as file
import joblib

# Save the trained model to a file
model_filename = 'knn_model.joblib'
joblib.dump(KNN, model_filename)

# Import trained model from file
from joblib import load

# Load the trained model from the file
loaded_model = load('knn_model.joblib')

# Now you can use the loaded model to make predictions
transformed_value = sc.transform([[50, 90000]])  # Assuming you have already transformed the input
loaded_prediction = loaded_model.predict(transformed_value)
print("Loaded Model Prediction:", loaded_prediction)