# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
bankdata = pd.read_csv("/home/rajini/Documents/GitHub/ML-algo-collection/supportVectorMachine/bill_authentication.csv")

# exploratory data analysis
print("********** Dimensions of Dataset **********")
print(bankdata.shape)
print("********** Preview of Dataset **********")
print(bankdata.head())

# data pre-processing

# dividing the data into attributes and labels
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

# dividing the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Training the Algorithm
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

# Making Predictions
y_pred = svclassifier.predict(X_test)

# Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix
print("********** Printing the Confusion Matrix **********")
print(confusion_matrix(y_test,y_pred))
print("********** Printing the Classification Report **********")
print(classification_report(y_test,y_pred))
