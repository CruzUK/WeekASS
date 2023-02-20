# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm


# Prepare the data
data = pd.read_csv('fruit_types.csv')
X = data.iloc[:,2:5]
Y = data.iloc[:,0]

# Split into training and test sets changed from test 0.3 to 02 random state from 99 to 32 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=32)

#Create a SVM Classifier
clfLinear = svm.SVC(kernel='linear') # Linear Kernel
clfsig = svm.SVC(kernel='sigmoid') # 
clfrbf = svm.SVC(kernel='rbf') #

#Train the model using the training sets
clfLinear.fit(X_train, y_train)
clfsig.fit(X_train, y_train)
clfrbf.fit(X_train, y_train)


#Predict the response for test dataset
y_pred = clfLinear.predict(X_test)
y_pred_sig = clfsig.predict(X_test)
y_pred_rbf = clfrbf.predict(X_test)

#Calculate the accuracy of our model
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_sig))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rbf))