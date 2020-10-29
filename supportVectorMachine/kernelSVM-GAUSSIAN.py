# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=colnames)

# data pre-processing
X = irisdata.drop('Class', axis=1)
y = irisdata['Class']

# dividing the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Training the Algorithm
from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

# Making Predictions
y_pred = svclassifier.predict(X_test)

# Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix
print("********** Printing the Confusion Matrix **********")
print(confusion_matrix(y_test, y_pred))
print("********** Printing the Classification Report **********")
print(classification_report(y_test, y_pred))
