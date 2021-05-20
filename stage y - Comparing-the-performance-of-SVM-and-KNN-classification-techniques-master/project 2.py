import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

# Data maticies:
Mat_1 = np.random.randint(0, 500, size = (10,10))
Mat_2 = np.random.randint(0, 500, size = (10,10))
Mat_3 = np.random.randint(0, 500, size = (10,10))
Mat_4 = np.random.randint(0, 500, size = (10,10))
Mat_5 = np.random.randint(0, 500, size = (10,10))
Mat_6 = np.random.randint(0, 500, size = (10,10))
Mat_7 = np.random.randint(0, 500, size = (10,10))
Mat_8 = np.random.randint(0, 500, size = (10,10))
Mat_9 = np.random.randint(0, 500, size = (10,10))
Mat_10 = np.random.randint(0, 500, size = (10,10))

Mat_array = [Mat_1, Mat_2, Mat_3, Mat_4, Mat_5, Mat_6, Mat_7, Mat_8, Mat_9, Mat_10]

# features:

Mean = []
Variance = []
Median = []
STD = []            # standard deviation 
Min = []
Max = []

print("10x10 matricies representation:")
print("__________________________")

for counter in Mat_array:
    print(counter)
    print("**********************************************************")
    Mean.append(np.mean( counter))
    Median.append(np.median(counter))
    STD.append(np.std(counter))
    Variance.append(np.var(counter))
    Min.append(np.amin(counter))
    Max.append(np.amax(counter))
    
Class = [0,0,0,0,0,1,1,1,1,1]
Features = [Mean, Median, Variance, STD, Min, Max]

# SVM Classifier:
################

X = np.array(Features)      # X= (6x10) - representation of features on X axis
X = X.transpose()             # X= (10x6) - for X and Y to have the same dimensions
Y = np.array(Class)         # Y= (10, ) - representation of Class on Y axis

print("Resulted 6x10 matrix: ")
print("Features Resulted Matrix:")
print(X)
print("Features Resulted size:")
print(X.shape)
print("Class Resulted Matrix: ")
print(Y)
print("Class Resulted Size:")
print(Y.shape)

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
###############################################################################

# KNN Classifier:
###############
#  it uses all of the data for training while classifying a new data point or instance.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)

c_predict = classifier.predict(x_test)

print(confusion_matrix(y_test, c_predict))
print(classification_report(y_test, c_predict))




            
