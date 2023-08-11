import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading data from csv file
dataset = pd.read_csv('Data.csv')

# Creating matrix of features and dependent variable vector
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Encode dependent variable
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Y = le.fit_transform(Y)

# Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split

X_train , X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Future Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

