from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import glob
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

samples = [cv2.imread(file,0) for file in glob.glob("C:/Users/nada/Desktop/Academy/YEAR 4 -ya mosahel ya rb/d) Pattern Recognition - 3/assignments/Project 2/*.jpg")]

# features:

Mean = []
Variance = []
Median = []
STD = []            
Min = []
Max = []

# Feature Extraction:
###################
for counter in samples:
    Mean.append(np.mean( counter))
    Median.append(np.median(counter))
    STD.append(np.std(counter))
    Variance.append(np.var(counter))
    Min.append(np.amin(counter))
    Max.append(np.amax(counter))
    
Class = ["D","D","D","D","D","F","F","F","F","F"]
Features = [Mean, Median, Variance, STD, Min, Max]
##############################################################################

# SVM Classifier:
################
X = np.array(Features)      # X= (6x10) - representation of features on X axis
X = X.transpose()             # X= (10x6) - for X and Y to have the same dimensions
Y = np.array(Class)         # Y= (10, ) - representation of Class on Y axis

print("Resulted 6x10 matrix: ")
print("-----------------------------------")
print("Features Resulted Matrix:")
print(Features)
print("Features Resulted size:")
print(X.shape)
print("Class Resulted Matrix: ")
print(Y)
print("Class Resulted Size:")
print(Y.shape)

print(" ")
print("--------------------------------------------------------------------------------------------")
print(" ")
print("SVM Classifier results")
print("---------------------------------")

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4)
# 0.60 is the test_size:  represent the proportion of the dataset to include in the test split
# range from 0.00 to 1.00

SVClassifier = SVC(kernel = 'linear')
SVClassifier.fit(x_train, y_train)
Class_predict = SVClassifier.predict(x_test)
#svm library, which contains built-in classes for different SVM algorithms.
#Since we are going to perform a classification task, we will use the support vector classifier class,

print("Result of Confusion/ Error matrix:")
print(confusion_matrix(y_test, Class_predict))
print("Result of Classification Report:")
print(classification_report(y_test, Class_predict))
print(" ")
print("--------------------------------------------------------------------------------------------")
print(" ")


###############################################################################

# KNN Classifier:
###############
print("KNN Classifier results")
print("---------------------------------")
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)

c_predict = classifier.predict(x_test)

print("Result of Confusion/ Error matrix:")
print(confusion_matrix(y_test, c_predict))
print("Result of Classification Report:")
print(classification_report(y_test, c_predict))
print(" ")
print("--------------------------------------------------------------------------------------------")
print(" ")

