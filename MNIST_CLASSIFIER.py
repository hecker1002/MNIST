#!/usr/bin/env python
# coding: utf-8




# This ML model will only identify (classify) a handwritten DIGIT ( NOT a number)
# Instances = data input (f + l)
import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_openml
import matplotlib as mpl 
import matplotlib.pyplot as plt

# Each image in this dataset is actually a 2D array of 28 x 28 pixels , so in actual , each image has 784 features and a 
#single label and we use these 784 features to train the model to classify new data .

mnist = fetch_openml('mnist_784' , version = 1 ,  parser='auto') # a dictionary of {key : value }

# mnist here is actually a dict. 

X  = mnist["data"]  # features of each img and img itself
Y = mnist["target"]  # labels of each image 

# displaying image of data 2D array
# IMP- image and its 784 features, both should be  np nd arrays 
img = X.to_numpy()[0] 
img_show = img.reshape(28 , 28)

# typecasting labels (actual numbers) as integer .
Y = Y.astype(np.uint8)

# plt.imshow(img_show , cmap =  mpl.cm.binary , interpolation="nearest")
# plt.axis("off")
# plt.show()

# Dividng dataset into training and testing data 

X_train = X[ : 60000]
Y_train = Y[ : 60000]

X_test =  X [60000 : ]
Y_test =  Y [60000 : ]


# For now , just build a Binary Classifier that check wether the given image is of '5' or NOT .
# changing the labels temp. to T/ F 
Y_train_5 = (Y_train == 5) 
Y_test_5 = (Y_test == 5)

from sklearn.linear_model import SGDClassifier  # SGD - Stochastic Gradient Descent updates param. after learning from each train_data

sgd_clf = SGDClassifier()

sgd_clf.fit(X_train , Y_train_5) # this Bin. classifier actually classifies if the features of img match with True(belong to 5) or NOT
# (f , l) = (784 pixel info. , True/False) based on if it belongs to class of 5 or NOT 

# prediction if img is of 5 or not

# for prediction , give input in form of a list 
sgd_clf.predict([img]) # => if True , means (img) is of digit '5'


# Performance Measure for this SGD classifier  - Confusion matrix

# confusion matrix - a matrix showing how many times a classifier correctly predicted the class of a data .(TP/ FN) U (TN , FP)

# we use K - fold cross validation - where train data is divided into K subsets of equal size => cross_val_predict(cv=K )
# and model  is independently trained on each independent dataset . and from here , we got predicted class .

from sklearn.model_selection import cross_val_predict

Y_train_predict = cross_val_predict( sgd_clf , X_train , Y_train_5 , cv=3) # predicted classes of binary classifier 

from sklearn.metrics import confusion_matrix , precision_score , recall_score , f1_score 
confusion_matrix (Y_train_5 , Y_train_predict)

precision_score (Y_train_5 , Y_train_predict)
recall_score (Y_train_5 , Y_train_predict) # how many correct class it can detect for data

# f1 score = Harmonic Mean ( precision , recall) gives a very accurate preformance measure of a classifier

f1_Score = f1_score (Y_train_5 , Y_train_predict)
f1_Score 

# to calculat score of each of all data input  ( in form of list)
y_score = sgd_clf.decision_function([img]) 

y_score # shows how much data (in probability) went into right class 

# Now , setting a threshold to compare scores to .
threshold = 0 
y_score_predict = (y_score > threshold ) 

# to decide threshold , return scores (from decision function) fo all data inputs 
y_score = cross_val_predict(sgd_clf , X_train , Y_train_5 , cv = 3 , method = "decision_function")
y_score 

from sklearn.metrics import precision_recall_curve 

precision , recall , threshold = precision_recall_curve (Y_train_5 , y_score)

# ploting the prec , recall , threshold
plt.plot(threshold , precision[ : -1 ] , "b--" , label="precision")
plt.plot(threshold , recall[ : -1 ] , "g--" , label="recall")


plt.show()

# lowest threshold that gives 90% precision
threshold_90_precision = threshold [np.argmax(precision >= 0.9)]
threshold_90_precision # req. threshold

Y_train_90 = (y_score >= threshold_90_precision)

precision_score (Y_train_5 , Y_train_90)

# ROC Curve - (True positive rate during class. = recall) / (false positive rate class.) , ROC = recall(senstivity)/ (1- specificity)
from sklearn.metrics import roc_curve
fpr , tpr , thresholds = roc_curve(Y_train_5 , y_score)

def plot_roc (fpr , tpr , label = None) :
    plt.plot(fpr , tpr , linewidth = 2 , label =label)
    plt.plot( [0, 1] , [0 , 1] , 'k--') # for a purely random classifier , ROC curve
    [...]
    
plot_roc(fpr , tpr )
plt.show()

from sklearn.metrics import roc_auc_score

# AUC - Area Under Curve =1 ( good classifier)

AUC = roc_auc_score(Y_train_5 , y_score)


# ROC curve for a random Forest Classifier is better .


# Multiclass Classification 

# Using SGD_CLF but using One-v/s- All strategy - DEFAULT

sgd_clf.fit(X_train , Y_train) # it gives decision score to each class wrt given new input and gives the output as class with highest score 
sgd_clf.predict([img])


# Using SGD_CLF using One-v/s-One strategy  - here , (n C 2 ) binary classifier have been built. (n=10 here  from 0 to 9 digit)


from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(sgd_clf )
ovo_clf.fit(X_train , Y_train)
ovo_clf.predict([img])


# building a confusion matrix for multi class classification 

Y_Predict = cross_val_predict( sgd_clf ,X_train , Y_train  , cv=3)
Conf_Mat = confusion_matrix (Y_train , Y_Predict)
plt.matshow(Conf_Mat , cmap="binary")
plt.show()


#we can remove noise (blurrines) from a mnist image using KNN Classifier .







