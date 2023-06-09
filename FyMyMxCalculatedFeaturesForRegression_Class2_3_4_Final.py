# -*- coding: utf-8 -*-
"""
Created on Wed June 03 13:19:27 2020

@author: Bahare Samadi, PhD Thesis, Identification of the Cobb angle in AIS
"""




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn import svm

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import glob


from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Force Keras to use CPU instead of GPU (no gpu installed, result in error)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

clear = lambda: os.system('cls')  # On Windows System
clear()

print("Import finished")





# Arranging the data
# =============================================================================




directory = "C:/Users/basama/PycharmProjects/Scoliosis Belgium Database/Qact_data/Graphs of 6 cycle per class"
for current_file in glob.glob(directory+"/Class2Parameters.csv"):
    Class2parameters = np.genfromtxt(current_file,delimiter=",",skip_header=0,usecols=range(0,28))
    FymeanClass2 = Class2parameters [:, [0]]
    FymeanABSClass2 = Class2parameters [:, [1]]
    FySDClass2 = Class2parameters [:, [2]]
    FyMinClass2 = Class2parameters [:, [3]]
    FyMinClass2ABS = abs(FyMinClass2)
    FyMaxClass2 = Class2parameters [:, [4]]
    FyMaxClass2ABS = abs(FyMaxClass2)
    FyVarClass2 = Class2parameters [:, [5]]
    FyMaxMeanClass2 = Class2parameters [:, [6]]
    FyVarMeanClass2 = Class2parameters [:, [7]]
    
    MymeanClass2 = Class2parameters [:, [8]]
    MymeanABSClass2 = Class2parameters [:, [9]]
    MySDClass2 = Class2parameters [:, [10]]
    MyMinClass2 = Class2parameters [:, [11]]
    MyMinClass2ABS = abs(MyMinClass2)
    MyMaxClass2 = Class2parameters [:, [12]]
    MyMaxClass2ABS = abs(MyMaxClass2)
    MyVarClass2 = Class2parameters [:, [13]]
    MyMaxMeanClass2 = Class2parameters [:, [14]]
    MyVarMeanClass2 = Class2parameters [:, [15]]
    
    MxmeanClass2 = Class2parameters [:, [16]]
    MxmeanABSClass2 = Class2parameters [:, [17]]
    MxSDClass2 = Class2parameters [:, [18]]
    MxMinClass2 = Class2parameters [:, [19]]
    MxMinClass2ABS = abs(MxMinClass2)
    MxMaxClass2 = Class2parameters [:, [20]]
    MxMaxClass2ABS = abs(MxMaxClass2)
    MxVarClass2 = Class2parameters [:, [21]]
    MxMaxMeanClass2 = Class2parameters [:, [22]]
    MxVarMeanClass2 = Class2parameters [:, [23]]
    
    MxdevideMyClass2 = Class2parameters [:, [24]]
    MaxMxdevideMyClass2 = MxdevideMyClass2 [0:8]

    labelClass2 = Class2parameters [:, [25]]
    CAClass2 = Class2parameters [:, [27]]


print("Class 2 finished")

directory = "C:/Users/basama/PycharmProjects/Scoliosis Belgium Database/Qact_data/Graphs of 6 cycle per class"
for current_file in glob.glob(directory+"/Class3Parameters.csv"):
    Class3parameters = np.genfromtxt(current_file,delimiter=",",skip_header=0,usecols=range(0,28))
    FymeanClass3 = Class3parameters [:, [0]]
    FymeanABSClass3 = Class3parameters [:, [1]]
    FySDClass3 = Class3parameters [:, [2]]
    FyMinClass3 = Class3parameters [:, [3]]
    FyMinClass3ABS = abs(FyMinClass3)
    FyMaxClass3 = Class3parameters [:, [4]]
    FyMaxClass3ABS = abs(FyMaxClass3)
    FyVarClass3 = Class3parameters [:, [5]]
    FyMaxMeanClass3 = Class3parameters [:, [6]]
    FyVarMeanClass3 = Class3parameters [:, [7]]
    
    MymeanClass3 = Class3parameters [:, [8]]
    MymeanABSClass3 = Class3parameters [:, [9]]
    MySDClass3 = Class3parameters [:, [10]]
    MyMinClass3 = Class3parameters [:, [11]]
    MyMinClass3ABS = abs(MyMinClass3)
    MyMaxClass3 = Class3parameters [:, [12]]
    MyMaxClass3ABS = abs(MyMaxClass3)
    MyVarClass3 = Class3parameters [:, [13]]
    MyMaxMeanClass3 = Class3parameters [:, [14]]
    MyVarMeanClass3 = Class3parameters [:, [15]]
    
    MxmeanClass3 = Class3parameters [:, [16]]
    MxmeanABSClass3 = Class3parameters [:, [17]]
    MxSDClass3 = Class3parameters [:, [18]]
    MxMinClass3 = Class3parameters [:, [19]]
    MxMinClass3ABS = abs(MxMinClass3)
    MxMaxClass3 = Class3parameters [:, [20]]
    MxMaxClass3ABS = abs(MxMaxClass3)
    MxVarClass3 = Class3parameters [:, [21]]
    MxMaxMeanClass3 = Class3parameters [:, [22]]
    MxVarMeanClass3 = Class3parameters [:, [23]]
    
    MaxMxdevideMyClass3 = Class3parameters [:, [24]]

    labelClass3 = Class3parameters [:, [25]]
    CAClass3 = Class3parameters [:, [27]]



print("Class 3 finished")  


directory = "C:/Users/basama/PycharmProjects/Scoliosis Belgium Database/Qact_data/Graphs of 6 cycle per class"
for current_file in glob.glob(directory+"/Class4Parameters.csv"):
    Class4parameters = np.genfromtxt(current_file,delimiter=",",skip_header=0,usecols=range(0,28))
    FymeanClass4 = Class4parameters [:, [0]]
    FymeanABSClass4 = Class4parameters [:, [1]]
    FySDClass4 = Class4parameters [:, [2]]
    FyMinClass4 = Class4parameters [:, [3]]
    FyMinClass4ABS = abs(FyMinClass4)
    FyMaxClass4 = Class4parameters [:, [4]]
    FyMaxClass4ABS = abs(FyMaxClass4)
    FyVarClass4 = Class4parameters [:, [5]]
    FyMaxMeanClass4 = Class4parameters [:, [6]]
    FyVarMeanClass4 = Class4parameters [:, [7]]
    
    MymeanClass4 = Class4parameters [:, [8]]
    MymeanABSClass4 = Class4parameters [:, [9]]
    MySDClass4 = Class4parameters [:, [10]]
    MyMinClass4 = Class4parameters [:, [11]]
    MyMinClass4ABS = abs(MyMinClass4)
    MyMaxClass4 = Class4parameters [:, [12]]
    MyMaxClass4ABS = abs(MyMaxClass4)
    MyVarClass4 = Class4parameters [:, [13]]
    MyMaxMeanClass4 = Class4parameters [:, [14]]
    MyVarMeanClass4 = Class4parameters [:, [15]]
    
    MxmeanClass4 = Class4parameters [:, [16]]
    MxmeanABSClass4 = Class4parameters [:, [17]]
    MxSDClass4 = Class4parameters [:, [18]]
    MxMinClass4 = Class4parameters [:, [19]]
    MxMinClass4ABS = abs(MxMinClass4)
    MxMaxClass4 = Class4parameters [:, [20]]
    MxMaxClass4ABS = abs(MxMaxClass4)
    MxVarClass4 = Class4parameters [:, [21]]
    MxMaxMeanClass4 = Class4parameters [:, [22]]
    MxVarMeanClass4 = Class4parameters [:, [23]]
    
    MaxMxdevideMyClass4 = Class4parameters [:, [24]]

    labelClass4 = Class4parameters [:, [25]]
    CAClass4 = Class4parameters [:, [27]]



print("Class 4 finished")      

####
## Define X, y
###

XFyMaxMean = np.concatenate ((FyMaxMeanClass2 , FyMaxMeanClass3 , FyMaxMeanClass4) ,
                      axis = 0)

XMyMaxMean = np.concatenate ((MyMaxMeanClass2 , MyMaxMeanClass3 , MyMaxMeanClass4),
                      axis = 0)

XMxMaxMean = np.concatenate ((MxMaxMeanClass2 , MxMaxMeanClass3 , MxMaxMeanClass4),
                      axis = 0)

XFyVarMean = np.concatenate ((FyVarMeanClass2 , FyVarMeanClass3 , FyVarMeanClass4) ,
                      axis = 0)

XMyVarMean = np.concatenate ((MyVarMeanClass2 , MyVarMeanClass3 , MyVarMeanClass4),
                      axis = 0)

XMxVarMean = np.concatenate ((MxVarMeanClass2 , MxVarMeanClass3 , MxVarMeanClass4),
                      axis = 0)

XFyMaxABSClass2 = np.max((FyMinClass2ABS,FyMaxClass2ABS),axis = 0)
XMyMaxABSClass2 = np.max((MyMinClass2ABS,MyMaxClass2ABS),axis = 0)
XMxMaxABSClass2 = np.max((MxMinClass2ABS,MxMaxClass2ABS),axis = 0)

XFyMaxABSClass3 = np.max((FyMinClass3ABS,FyMaxClass3ABS),axis = 0)
XMyMaxABSClass3 = np.max((MyMinClass3ABS,MyMaxClass3ABS),axis = 0)
XMxMaxABSClass3 = np.max((MxMinClass3ABS,MxMaxClass3ABS),axis = 0)

XFyMaxABSClass4 = np.max((FyMinClass4ABS,FyMaxClass4ABS),axis = 0)
XMyMaxABSClass4 = np.max((MyMinClass4ABS,MyMaxClass4ABS),axis = 0)
XMxMaxABSClass4 = np.max((MxMinClass4ABS,MxMaxClass4ABS),axis = 0)




XFyDifABSMinMaxClass2 = np.subtract(FyMinClass2ABS,FyMaxClass2ABS)
XMyDifABSMinMaxClass2 = np.subtract(MyMinClass2ABS,MyMaxClass2ABS)
XMxDifABSMinMaxClass2 = np.subtract(MxMinClass2ABS,MxMaxClass2ABS)

XFyDifABSMinMaxClass3 = np.subtract(FyMinClass3ABS,FyMaxClass3ABS)
XMyDifABSMinMaxClass3 = np.subtract(MyMinClass3ABS,MyMaxClass3ABS)
XMxDifABSMinMaxClass3 = np.subtract(MxMinClass3ABS,MxMaxClass3ABS)

XFyDifABSMinMaxClass4 = np.subtract(FyMinClass4ABS,FyMaxClass4ABS)
XMyDifABSMinMaxClass4 = np.subtract(MyMinClass4ABS,MyMaxClass4ABS)
XMxDifABSMinMaxClass4 = np.subtract(MxMinClass4ABS,MxMaxClass4ABS)  

XFyDifABSMinMax = np.concatenate((XFyDifABSMinMaxClass2,XFyDifABSMinMaxClass3,XFyDifABSMinMaxClass4),axis = 0)
XMyDifABSMinMax = np.concatenate((XMyDifABSMinMaxClass2,XMyDifABSMinMaxClass3,XMyDifABSMinMaxClass4),axis = 0)
XMxDifABSMinMax = np.concatenate((XMxDifABSMinMaxClass2,XMxDifABSMinMaxClass3,XMxDifABSMinMaxClass4),axis = 0)                    

XFyMaxABS = np.concatenate((XFyMaxABSClass2,XFyMaxABSClass3,XFyMaxABSClass4),axis = 0)
XMyMaxABS = np.concatenate((XMyMaxABSClass2,XMyMaxABSClass3,XMyMaxABSClass4),axis = 0)
XMxMaxABS = np.concatenate((XMxMaxABSClass2,XMxMaxABSClass3,XMxMaxABSClass4),axis = 0)


XFyABSMean = np.concatenate ((FymeanABSClass2 , FymeanABSClass3 , FymeanABSClass4) ,
                      axis = 0)

XMyABSMean = np.concatenate ((MymeanABSClass2 , MymeanABSClass3 , MymeanABSClass4),
                      axis = 0)

XMxABSMean = np.concatenate ((MxmeanABSClass2 , MxmeanABSClass3 , MxmeanABSClass4),
                      axis = 0)

XFySD = np.concatenate ((FySDClass2 , FySDClass3 , FySDClass4) ,
                      axis = 0)

XMySD = np.concatenate ((MySDClass2 , MySDClass3 , MySDClass4),
                      axis = 0)

XMxSD = np.concatenate ((MxSDClass2 , MxSDClass3 , MxSDClass4),
                      axis = 0)



XFy = np.concatenate((XFyMaxMean,XFyVarMean,XFyMaxABS,XFyABSMean,XFySD,XFyDifABSMinMax), axis = 1)
XMy = np.concatenate((XMyMaxMean,XMyVarMean,XMyMaxABS,XMyABSMean,XMySD,XMyDifABSMinMax), axis = 1)
XMx = np.concatenate((XMxMaxMean,XMxVarMean,XMxMaxABS,XMxABSMean,XMxSD,XMxDifABSMinMax), axis = 1)

#XFy = np.concatenate((XFyMaxMean,XFyVarMean,XFyMaxABS,XFyABSMean), axis = 1)
#XMy = np.concatenate((XMyMaxMean,XMyVarMean,XMyMaxABS,XMyABSMean), axis = 1)
#XMx = np.concatenate((XMxMaxMean,XMxVarMean,XMxMaxABS,XMxABSMean), axis = 1)

# =============================================================================
# Scaling the data
# =============================================================================

X1 = np.concatenate((XFy,XMy,XMx), axis = 1)
scaler = StandardScaler()

X = scaler.fit_transform(X1)
#y = np.concatenate ((labelClass2,labelClass3,labelClass4), axis = 0)
y = np.concatenate ((CAClass2,CAClass3,CAClass4), axis = 0)

y = y.ravel()

print("Define X, y finished") 


from sklearn.model_selection import ShuffleSplit
from numpy import mean
from numpy import std

MAETest = []
MAETrain = []

j = 500
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)


knn = KNeighborsRegressor(n_neighbors=3)



# KNN with kFold = 10, cross Validation

print('Cross validation score for KNN (mean):', cross_val_score(knn, X, y, cv=cv, scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(knn, X, y, cv=cv, scoring='neg_mean_absolute_error').std())


maeKNNCross = cross_val_score(knn, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of KNN(kfold cross validation):',maeKNNCross)


scoresKNN = list()
yPredictCrossKNN = list()
kfold = KFold(n_splits=10, random_state=0, shuffle=True)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	knn.fit(train_X, train_y)
	# evaluate model
	yhatKNN = knn.predict(test_X)
	yPredictCrossKNN.append(yhatKNN)
	maeKNN = mean_absolute_error(test_y, yhatKNN)
	# store score
	scoresKNN.append(maeKNN)
    
	print('> ', maeKNN)
    
# summarize model performance
mean_s, std_s = mean(scoresKNN), std(scoresKNN)
print('Mean(KNN): %.3f, Standard Deviation(KNN): %.3f' % (mean_s, std_s))

### Radius Neighbors Regressor
MAETest = []
MAETrain = []

RadiusNeigh = RadiusNeighborsRegressor(
algorithm='auto', leaf_size=30, metric='minkowski',
                          metric_params=None, n_jobs=None,
                          p=2, radius=6.0, weights='distance')




# Radius Neighbors Regressor with kFold = 10, cross Validation

print('Cross validation score for radius neighbor (mean):', cross_val_score(RadiusNeigh, X, y, cv=cv, scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(RadiusNeigh, X, y, cv=cv, scoring='neg_mean_absolute_error').std())


RadNeigh = cross_val_score(RadiusNeigh, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of radius neighbor(kfold cross validation):',RadNeigh)




scoresRadNeigh = list()
yPredictCrossRadNeigh = list()
kfold = KFold(n_splits=10, shuffle=True, random_state = 0)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	RadiusNeigh.fit(train_X, train_y)
	# evaluate model
	yhatRadNeigh = RadiusNeigh.predict(test_X)
	yPredictCrossRadNeigh.append(yhatRadNeigh)
	maeRadNeigh = mean_absolute_error(test_y, yhatRadNeigh)
	# store score
	scoresRadNeigh.append(maeRadNeigh)
    
	print('> ', maeRadNeigh)
    
# summarize model performance
mean_s, std_s = mean(scoresRadNeigh), std(scoresRadNeigh)
print('Mean(radius neighbor): %.3f, Standard Deviation(radius neighbor): %.3f' % (mean_s, std_s))

## Support Vector Machine Regressor

# SVR hyperparameter tuning
#from sklearn.model_selection import GridSearchCV 
#param_grid = {'C': [0.1, 1, 10, 100, 1000],  
#              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
#              'kernel': ['rbf','linear','poly']}  
#  
#grid = GridSearchCV(svm.SVR(), param_grid, refit = True, verbose = 3) 
#  
## fitting the model for grid search 
#grid.fit(X ,y) 
#
## print best parameter after tuning 
#print(grid.best_params_) 
#  
## print how our model looks after hyper-parameter tuning 
#print(grid.best_estimator_) 
##{'C': 1, 'gamma': 1, 'kernel': 'linear'}

RegSVR = svm.SVR(kernel='linear', C=1, gamma = 1) #kernel = 'poly', degree = 3 , C = 0.9, gamma = 'scale', decision_function_shape= 'ovr')
#RegSVR = svm.SVR(kernel='poly', C=0.9, degree = 3, gamma = 1) #kernel = 'poly', degree = 3 , C = 0.9, gamma = 'scale', decision_function_shape= 'ovr')
#decision_function_shape='ovo', 'ovr'
#for i in range(j): 
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle = True)
#    
#
#    RegSVR.fit(X_train, y_train)
#    y_pred_Test = RegSVR.predict(X_test)
#    y_pred_Train = RegSVR.predict(X_train)
#
#    maeTest = mean_absolute_error(y_test, y_pred_Test)
#    maeTrain = mean_absolute_error(y_train, y_pred_Train)
#
#    
#    MAETest.append(maeTest)
#    MAETrain.append(maeTrain)
#    
#maeTrain_final = np.mean(MAETrain)
#maeTest_final = np.mean(MAETest)
#
#print('Mean absolute error of train dataset Final (SVR):', maeTrain_final,'SD:',np.std(MAETrain))
#print('Mean absolute error of test dataset Final (SVR):', maeTest_final,'SD:',np.std(MAETest))


# cross Validation

print('Cross validation score for SVR (mean):', cross_val_score(RegSVR, X, y, cv=cv, 
      scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(RegSVR, X, y, cv=cv, scoring='neg_mean_absolute_error').std())


RegSVRcrossMAE = cross_val_score(RegSVR, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of SVR(kfold cross validation):',RegSVRcrossMAE)

scoresRegSVR = list()
yPredictCrossRegSVR = list()
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	RegSVR.fit(train_X, train_y)
	# evaluate model
	yhatRegSVR = RegSVR.predict(test_X)
	yPredictCrossRegSVR.append(yhatRegSVR)
	maeRegSVR = mean_absolute_error(test_y, yhatRegSVR)
	# store score
	scoresRegSVR.append(maeRegSVR)
    
	print('> ', maeRegSVR)
    
# summarize model performance
mean_s, std_s = mean(scoresRegSVR), std(scoresRegSVR)
print('Mean(SVR): %.3f, Standard Deviation(SVR): %.3f' % (mean_s, std_s))

# Random Forest Regressor

## Random Forest hyperparameter tuning
#from sklearn.model_selection import GridSearchCV 
#param_grid = { 
#    'n_estimators': [1, 200, 10],
#    'max_features': ['auto', 'sqrt', 'log2'],
#    'max_depth' : [5,150,1],
#    'min_samples_split' : [2, 5, 10],
#    'min_samples_leaf' : [1, 2, 4],
#    'bootstrap' : [True, False]
#}
#  
#grid = GridSearchCV(RandomForestRegressor(), param_grid, refit = True, verbose = 3) 
#  
## fitting the model for grid search 
#grid.fit(X ,y) 
#
## print best parameter after tuning 
#print(grid.best_params_) 
#  
## print how our model looks after hyper-parameter tuning 
#print(grid.best_estimator_) 


#{'bootstrap': True, 'max_depth': 1, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 1}
RFR = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mae',
                      max_depth=15, max_features='log2', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=2,
                      min_samples_split=10, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=None, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)

#RFR = RandomForestRegressor(n_estimators=1000,random_state=None)

#for i in range(j): 
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle = True)
#    
#
#
#    RFR.fit(X_train, y_train)
#    y_pred_Test = RFR.predict(X_test)
#    y_pred_Train = RFR.predict(X_train)
#
#    maeTest = mean_absolute_error(y_test, y_pred_Test)
#    r2_scoreTest = r2_score(y_test, y_pred_Test)
#    maeTrain = mean_absolute_error(y_train, y_pred_Train)
#    r2_scoreTrain = r2_score(y_train, y_pred_Train)
#
#
#    
#    MAETest.append(maeTest)
#    MAETrain.append(maeTrain)
#    
#maeTrain_final = np.mean(MAETrain)
#maeTest_final = np.mean(MAETest)
#
#print('Mean absolute error of train dataset Final (RFR):', maeTrain_final,'SD:',np.std(MAETrain))
#print('Mean absolute error of test dataset Final (RFR):', maeTest_final,'SD:',np.std(MAETest))


# cross Validation

print('Cross validation score for RFR (mean):', cross_val_score(RFR, X, y, cv=cv, 
      scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(RFR, X, y, cv=cv, scoring='neg_mean_absolute_error').std())


RFRcrossMAE = cross_val_score(RFR, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of RFR(kfold cross validation):',RFRcrossMAE)

scoresRegRFR = list()
yPredictCrossRegRFR = list()
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	RFR.fit(train_X, train_y)
	# evaluate model
	yhatRegRFR =RFR.predict(test_X)
	yPredictCrossRegRFR.append(yhatRegRFR)
	maeRegRFR = mean_absolute_error(test_y, yhatRegRFR)
	# store score
	scoresRegRFR.append(maeRegRFR)
    
	print('> ', maeRegRFR)
    
# summarize model performance
mean_s, std_s = mean(scoresRegRFR), std(scoresRegRFR)
print('Mean(RFR): %.3f, Standard Deviation(RFR): %.3f' % (mean_s, std_s))

#####
## Linear Regression 
LogReg = LinearRegression()
#####

#for i in range(j): 
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle = True)
#    
#
### Linear Regression
#
#    LogReg.fit(X_train, y_train)
#    y_pred_Test = LogReg.predict(X_test)
#    y_pred_Train = LogReg.predict(X_train)
#
#    maeTest = mean_absolute_error(y_test, y_pred_Test)
#    maeTrain = mean_absolute_error(y_train, y_pred_Train)
#
#    
#    MAETest.append(maeTest)
#    MAETrain.append(maeTrain)
#    
#maeTrain_final = np.mean(MAETrain)
#maeTest_final = np.mean(MAETest)
#
#print('Mean absolute error of train dataset Final (Linear Regression):', maeTrain_final,'SD:',np.std(MAETrain))
#print('Mean absolute error of test dataset Final (Linear Regression):', maeTest_final,'SD:',np.std(MAETest))


# cross Validation

print('Cross validation score for Linear Regression (mean):', cross_val_score(LogReg, X, y, cv=cv, 
      scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(LogReg, X, y, cv=cv, scoring='neg_mean_absolute_error').std())


LogRegcrossMAE = cross_val_score(LogReg, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of Linear Regression(kfold cross validation):',LogRegcrossMAE)


scoresLogReg = list()
yPredictCrossLogReg = list()
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	LogReg.fit(train_X, train_y)
	# evaluate model
	yhatLogReg = LogReg.predict(test_X)
	yPredictCrossLogReg.append(yhatLogReg)
	maeLogReg = mean_absolute_error(test_y, yhatLogReg)
	# store score
	scoresLogReg.append(maeLogReg)
    
	print('> ', maeLogReg)
    
# summarize model performance
mean_s, std_s = mean(scoresLogReg), std(scoresLogReg)
print('Mean(LogReg): %.3f, Standard Deviation(LogReg): %.3f' % (mean_s, std_s))

#####
## Ridge Regression
LogRegRidge = linear_model.Ridge(alpha=0.1)
#####

#for i in range(j): 
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle = True)
#    
#
### Linear Regression
#
#    LogRegRidge.fit(X_train, y_train)
#    y_pred_Test = LogRegRidge.predict(X_test)
#    y_pred_Train = LogRegRidge.predict(X_train)
#
#    maeTest = mean_absolute_error(y_test, y_pred_Test)
#    maeTrain = mean_absolute_error(y_train, y_pred_Train)
#
#    
#    MAETest.append(maeTest)
#    MAETrain.append(maeTrain)
#    
#maeTrain_final = np.mean(MAETrain)
#maeTest_final = np.mean(MAETest)
#
#print('Mean absolute error of train dataset Final (Ridge Regression):', maeTrain_final,'SD:',np.std(MAETrain))
#print('Mean absolute error of test dataset Final (Ridge Regression):', maeTest_final,'SD:',np.std(MAETest))


# cross Validation

print('Cross validation score for Ridge Regression (mean):', cross_val_score(LogRegRidge, X, y, cv=cv, 
      scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(LogRegRidge, X, y, cv=cv, scoring='neg_mean_absolute_error').std())


LogRegRidgecrossMAE = cross_val_score(LogRegRidge, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of Ridge Regression(kfold cross validation):',LogRegRidgecrossMAE)


scoresLogRegRidge = list()
yPredictCrossLogRegRidge = list()
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	LogRegRidge.fit(train_X, train_y)
	# evaluate model
	yhatLogRegRidge = LogRegRidge.predict(test_X)
	yPredictCrossLogRegRidge.append(yhatLogRegRidge)
	maeLogRegRidge = mean_absolute_error(test_y, yhatLogRegRidge)
	# store score
	scoresLogRegRidge.append(maeLogRegRidge)
    
	print('> ', maeLogRegRidge)
    
# summarize model performance
mean_s, std_s = mean(scoresLogRegRidge), std(scoresLogRegRidge)
print('Mean(LogRegRidge): %.3f, Standard Deviation(LogRegRidge): %.3f' % (mean_s, std_s))




## Lasso Regression
LogRegLasso = linear_model.Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=10000,
      normalize=False, positive=False, precompute=False, random_state=None,
      selection='random', tol=0.0001, warm_start=False)
#####

#for i in range(j): 
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle = True)
#    
#
### Linear Regression
#
#    LogRegLasso.fit(X_train, y_train)
#    y_pred_Test = LogRegLasso.predict(X_test)
#    y_pred_Train = LogRegLasso.predict(X_train)
#
#    maeTest = mean_absolute_error(y_test, y_pred_Test)
#    maeTrain = mean_absolute_error(y_train, y_pred_Train)
#
#    
#    MAETest.append(maeTest)
#    MAETrain.append(maeTrain)
#    
#maeTrain_final = np.mean(MAETrain)
#maeTest_final = np.mean(MAETest)
#
#print('Mean absolute error of train dataset Final (Lasso Regression):', maeTrain_final,'SD:',np.std(MAETrain))
#print('Mean absolute error of test dataset Final (Lasso Regression):', maeTest_final,'SD:',np.std(MAETest))


# cross Validation

print('Cross validation score for Lasso Regression (mean):', cross_val_score(LogRegLasso, X, y, cv=cv, 
      scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(LogRegLasso, X, y, cv=cv, scoring='neg_mean_absolute_error').std())


LogRegLassocrossMAE = cross_val_score(LogRegLasso, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of Lasso Regression(kfold cross validation):',LogRegLassocrossMAE)


scoresLogRegLasso = list()
yPredictCrossLogRegLasso = list()
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	LogRegLasso.fit(train_X, train_y)
	# evaluate model
	yhatLogRegLasso = LogRegLasso.predict(test_X)
	yPredictCrossLogRegLasso.append(yhatLogRegLasso)
	maeLogRegLasso = mean_absolute_error(test_y, yhatLogRegLasso)
	# store score
	scoresLogRegLasso.append(maeLogRegLasso)
    
	print('> ', maeLogRegLasso)
    
# summarize model performance
mean_s, std_s = mean(scoresLogRegLasso), std(scoresLogRegLasso)
print('Mean(LogRegLasso): %.3f, Standard Deviation(LogRegLasso): %.3f' % (mean_s, std_s))

## Guassian process Classifier
#kernel = C(2.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
kernel = DotProduct() + WhiteKernel()
gpReg = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=50)


#for i in range(j): 
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle = True)
#    
#
#    gpReg .fit(X_train, y_train)
#    y_pred_Test = gpReg .predict(X_test)
#    y_pred_Train = gpReg .predict(X_train)
#
#    maeTest = mean_absolute_error(y_test, y_pred_Test)
#    maeTrain = mean_absolute_error(y_train, y_pred_Train)
#
#    
#    MAETest.append(maeTest)
#    MAETrain.append(maeTrain)
#    
#maeTrain_final = np.mean(MAETrain)
#maeTest_final = np.mean(MAETest)
#
#print('Mean absolute error of train dataset Final (Gaussian Process):', maeTrain_final,'SD:',np.std(MAETrain))
#print('Mean absolute error of test dataset Final (Gaussian Process):', maeTest_final,'SD:',np.std(MAETest))


# cross Validation

print('Cross validation score for Gaussian Process (mean):', cross_val_score(gpReg, X, y, cv=cv, 
      scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(gpReg, X, y, cv=cv, scoring='neg_mean_absolute_error').std())


gpRegcrossMAE = cross_val_score(gpReg, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of Gaussian Process(kfold cross validation):',gpRegcrossMAE)


scoresgpReg = list()
yPredictCrossgpReg = list()
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	gpReg.fit(train_X, train_y)
	# evaluate model
	yhatgpReg = gpReg.predict(test_X)
	yPredictCrossgpReg.append(yhatgpReg)
	maegpReg = mean_absolute_error(test_y, yhatgpReg)
	# store score
	scoresgpReg.append(maegpReg)
    
	print('> ', maegpReg)
    
# summarize model performance
mean_s, std_s = mean(scoresgpReg), std(scoresgpReg)
print('Mean(gpReg): %.3f, Standard Deviation(gpreg): %.3f' % (mean_s, std_s))
## Multi-layer Perceptron  Regressor

MLPReg = MLPRegressor(activation='logistic', alpha=0.001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(60,), learning_rate='constant',
              learning_rate_init=0.003, max_fun=15000, max_iter=1000,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=0, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)
#MLPReg = MLPRegressor (random_state=1, max_iter=500)
#for i in range(j): 
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle = True)
#    
#
#    MLPReg .fit(X_train, y_train)
#    y_pred_Test = MLPReg .predict(X_test)
#    y_pred_Train = MLPReg .predict(X_train)
#
#    maeTest = mean_absolute_error(y_test, y_pred_Test)
#    maeTrain = mean_absolute_error(y_train, y_pred_Train)
#
#    
#    MAETest.append(maeTest)
#    MAETrain.append(maeTrain)
#    
#maeTrain_final = np.mean(MAETrain)
#maeTest_final = np.mean(MAETest)
#
#print('Mean absolute error of train dataset Final (MLP Regression):', maeTrain_final,'SD:',np.std(MAETrain))
#print('Mean absolute error of test dataset Final (MLP Regression):', maeTest_final,'SD:',np.std(MAETest))


# cross Validation

print('Cross validation score for MLP Regression (mean):', cross_val_score(MLPReg, X, y, cv=cv, 
      scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(MLPReg, X, y, cv=cv, scoring='neg_mean_absolute_error').std())


MLPRegcrossMAE = cross_val_score(MLPReg, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of MLP Regression(kfold cross validation):',MLPRegcrossMAE)

scoresMLPReg = list()
yPredictCrossMLPReg = list()
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	MLPReg.fit(train_X, train_y)
	# evaluate model
	yhatMLPReg = MLPReg.predict(test_X)
	yPredictCrossMLPReg.append(yhatMLPReg)
	maeMLPReg = mean_absolute_error(test_y, yhatMLPReg)
	# store score
	scoresMLPReg.append(maeMLPReg)
    
	print('> ', maeMLPReg)
    
# summarize model performance
mean_s, std_s = mean(scoresMLPReg), std(scoresMLPReg)
print('Mean(MLPReg): %.3f, Standard Deviation(MLPReg): %.3f' % (mean_s, std_s))

####
#AdaBoostRegressor
####



AdaBoostReg = AdaBoostRegressor(base_estimator=None, learning_rate=1.1, loss='linear',
                  n_estimators=250, random_state=0)

#for i in range(j): 
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle = True)
#    
#
#    AdaBoostReg .fit(X_train, y_train)
#    y_pred_Test =AdaBoostReg .predict(X_test)
#    y_pred_Train =AdaBoostReg .predict(X_train)
#
#    maeTest = mean_absolute_error(y_test, y_pred_Test)
#    maeTrain = mean_absolute_error(y_train, y_pred_Train)
#
#    
#    MAETest.append(maeTest)
#    MAETrain.append(maeTrain)
#    
#maeTrain_final = np.mean(MAETrain)
#maeTest_final = np.mean(MAETest)
#
#print('Mean absolute error of train dataset Final (Ada Boost Regression):', maeTrain_final,'SD:',np.std(MAETrain))
#print('Mean absolute error of test dataset Final (Ada Boost Regression):', maeTest_final,'SD:',np.std(MAETest))
#

# cross Validation

print('Cross validation score for Ada Boost Regression (mean):', cross_val_score(AdaBoostReg, X, y, cv=cv, 
      scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(AdaBoostReg, X, y, cv=cv, scoring='neg_mean_absolute_error').std())


AdaBoostRegcrossMAE = cross_val_score(AdaBoostReg, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of Ada Boost Regression(kfold cross validation):',AdaBoostRegcrossMAE)

scoresAdaBoostReg = list()
yPredictCrossAdaBoostReg = list()
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	AdaBoostReg.fit(train_X, train_y)
	# evaluate model
	yhatAdaBoostReg = AdaBoostReg.predict(test_X)
	yPredictCrossAdaBoostReg.append(yhatAdaBoostReg)
	maeAdaBoostReg = mean_absolute_error(test_y, yhatAdaBoostReg)
	# store score
	scoresAdaBoostReg.append(maeAdaBoostReg)
    
	print('> ', maeAdaBoostReg)
    
# summarize model performance
mean_s, std_s = mean(scoresAdaBoostReg), std(scoresAdaBoostReg)
print('Mean(AdaBoostReg): %.3f, Standard Deviation(AdaBoostReg): %.3f' % (mean_s, std_s))

#######
#Decision tree Regression
######

DTReg = DecisionTreeRegressor(max_depth=4,random_state=0,criterion='mse' )

#for i in range(j): 
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle = True)
#    
#
#    DTReg .fit(X_train, y_train)
#    y_pred_Test =DTReg .predict(X_test)
#    y_pred_Train = DTReg .predict(X_train)
#
#    maeTest = mean_absolute_error(y_test, y_pred_Test)
#    maeTrain = mean_absolute_error(y_train, y_pred_Train)
#
#    
#    MAETest.append(maeTest)
#    MAETrain.append(maeTrain)
#    
#maeTrain_final = np.mean(MAETrain)
#maeTest_final = np.mean(MAETest)
#
#print('Mean absolute error of train dataset Final (Decision tree Regression):', maeTrain_final,'SD:',np.std(MAETrain))
#print('Mean absolute error of test dataset Final (Decision tree Regression):', maeTest_final,'SD:',np.std(MAETest))
#

# cross Validation

print('Cross validation score for Decision tree Regression (mean):', cross_val_score(DTReg, X, y, cv=cv, 
      scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(DTReg, X, y, cv=cv, scoring='neg_mean_absolute_error').std())
#print('r2:', cross_val_score(DTReg, X, y, cv=cv, scoring='r2').mean())


DTRegcrossMAE = cross_val_score(DTReg, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of Decision tree Regression(kfold cross validation):',DTRegcrossMAE)

scoresDTReg = list()
yPredictCrossDTReg = list()
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	DTReg.fit(train_X, train_y)
	# evaluate model
	yhatDTReg = DTReg.predict(test_X)
	yPredictCrossDTReg.append(yhatDTReg)
	maeDTReg = mean_absolute_error(test_y, yhatDTReg)
	# store score
	scoresDTReg.append(maeDTReg)
    
	print('> ', maeDTReg)
    
# summarize model performance
mean_s, std_s = mean(scoresDTReg), std(scoresDTReg)
print('Mean(DTReg): %.3f, Standard Deviation(DTReg): %.3f' % (mean_s, std_s))

#######
#Bayesian Ridge Regression
######

BRidgeReg = linear_model.BayesianRidge()


#for i in range(j): 
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle = True)
#    
#
#    BRidgeReg .fit(X_train, y_train)
#    y_pred_Test =BRidgeReg .predict(X_test)
#    y_pred_Train =BRidgeReg .predict(X_train)
#
#    maeTest = mean_absolute_error(y_test, y_pred_Test)
#    maeTrain = mean_absolute_error(y_train, y_pred_Train)
#
#    
#    MAETest.append(maeTest)
#    MAETrain.append(maeTrain)
#    
#maeTrain_final = np.mean(MAETrain)
#maeTest_final = np.mean(MAETest)
#
#print('Mean absolute error of train dataset Final (Bayesian Regression):', maeTrain_final,'SD:',np.std(MAETrain))
#print('Mean absolute error of test dataset Final (Bayesian Regression):', maeTest_final,'SD:',np.std(MAETest))


# cross Validation

print('Cross validation score for Bayesian Regression (mean):', cross_val_score(BRidgeReg, X, y, cv=cv, 
      scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(BRidgeReg, X, y, cv=cv, scoring='neg_mean_absolute_error').std())


BRidgeRegcrossMAE = cross_val_score(BRidgeReg, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of Bayesian Regression(kfold cross validation):',BRidgeRegcrossMAE)

scoresBRidgeReg = list()
yPredictCrossBRidgeReg = list()
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	BRidgeReg.fit(train_X, train_y)
	# evaluate model
	yhatBRidgeReg = BRidgeReg.predict(test_X)
	yPredictCrossBRidgeReg.append(yhatBRidgeReg)
	maeBRidgeReg = mean_absolute_error(test_y, yhatBRidgeReg)
	# store score
	scoresBRidgeReg.append(maeBRidgeReg)
    
	print('> ', maeBRidgeReg)
    
# summarize model performance
mean_s, std_s = mean(scoresBRidgeReg), std(scoresBRidgeReg)
print('Mean(BRidgeReg): %.3f, Standard Deviation(BRidgeReg): %.3f' % (mean_s, std_s))
#######
###BTweedieRegressor
#######
#from sklearn.linear_model import TweedieRegressor
#regTweedie = TweedieRegressor(power=1, alpha=0.5, link='log')
#
#for i in range(j): 
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle = True)
#    
#
#    regTweedie .fit(X_train, y_train)
#    y_pred_Test =regTweedie .predict(X_test)
#    y_pred_Train =regTweedie .predict(X_train)
#
#    maeTest = mean_absolute_error(y_test, y_pred_Test)
#    maeTrain = mean_absolute_error(y_train, y_pred_Train)
#
#    
#    MAETest.append(maeTest)
#    MAETrain.append(maeTrain)
#    
#maeTrain_final = np.mean(MAETrain)
#maeTest_final = np.mean(MAETest)
#
#print('Mean absolute error of train dataset Final (Tweedie Regression):', maeTrain_final,'SD:',np.std(MAETrain))
#print('Mean absolute error of test dataset Final (Tweedie Regression):', maeTest_final,'SD:',np.std(MAETest))
#
#
## cross Validation
#
#print('Cross validation score for Tweedie Regression (mean):', cross_val_score(regTweedie, X, y, cv=cv, 
#      scoring='neg_mean_absolute_error').mean()*(-1))
#print('SD:', cross_val_score(regTweedie, X, y, cv=cv, scoring='neg_mean_absolute_error').std())
#
#
#regTweediecrossMAE = cross_val_score(regTweedie, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
#print('Mean absoulte errors of Tweedie Regression(kfold cross validation):',regTweediecrossMAE)
#
#
#
#

######
##Bagging Regressor
######

BaggingReg = BaggingRegressor(base_estimator=DTReg,
                         n_estimators=20, random_state=0)

#for i in range(j): 
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle = True)
#    
#
#    BaggingReg .fit(X_train, y_train)
#    y_pred_Test =BaggingReg .predict(X_test)
#    y_pred_Train = BaggingReg .predict(X_train)
#
#    maeTest = mean_absolute_error(y_test, y_pred_Test)
#    maeTrain = mean_absolute_error(y_train, y_pred_Train)
#
#    
#    MAETest.append(maeTest)
#    MAETrain.append(maeTrain)
#    
#maeTrain_final = np.mean(MAETrain)
#maeTest_final = np.mean(MAETest)
#
#print('Mean absolute error of train dataset Final (BaggingReg):', maeTrain_final,'SD:',np.std(MAETrain))
#print('Mean absolute error of test dataset Final (BaggingReg):', maeTest_final,'SD:',np.std(MAETest))

scoresBaggingReg = list()
yPredictCrossBaggingReg = list()
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	BaggingReg.fit(train_X, train_y)
	# evaluate model
	yhatBaggingReg = BaggingReg.predict(test_X)
	yPredictCrossBaggingReg.append(yhatBaggingReg)
	maeBaggingReg = mean_absolute_error(test_y, yhatBaggingReg)
	# store score
	scoresBaggingReg.append(maeBaggingReg)
    
	print('> ', maeBaggingReg)
    
# summarize model performance
mean_s, std_s = mean(scoresBaggingReg), std(scoresBaggingReg)
print('Mean(BaggingReg): %.3f, Standard Deviation(BaggingReg): %.3f' % (mean_s, std_s))
# cross Validation

print('Cross validation score for Bagging Regression (mean):', cross_val_score(BaggingReg, X, y, cv=cv, 
      scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(BaggingReg, X, y, cv=cv, scoring='neg_mean_absolute_error').std())


BaggingRegcrossMAE = cross_val_score(BaggingReg, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of Bagging Regression(kfold cross validation):',BaggingRegcrossMAE)




########
## Extra Trees Regression
########

ExtraTreesReg = ExtraTreesRegressor(bootstrap=False, n_estimators=200, random_state=0,
                                    max_features=None,criterion='mae')


#for i in range(j): 
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle = True)
#    
#
#    ExtraTreesReg .fit(X_train, y_train)
#    y_pred_Test =ExtraTreesReg .predict(X_test)
#    y_pred_Train = ExtraTreesReg .predict(X_train)
#
#    maeTest = mean_absolute_error(y_test, y_pred_Test)
#    maeTrain = mean_absolute_error(y_train, y_pred_Train)
#
#    
#    MAETest.append(maeTest)
#    MAETrain.append(maeTrain)
#    
#maeTrain_final = np.mean(MAETrain)
#maeTest_final = np.mean(MAETest)
#
#print('Mean absolute error of train dataset Final (Extra Trees Regression):', maeTrain_final,'SD:',np.std(MAETrain))
#print('Mean absolute error of test dataset Final (Extra Trees regression):', maeTest_final,'SD:',np.std(MAETest))


# cross Validation

print('Cross validation score for Extra Trees Regression (mean):', cross_val_score(ExtraTreesReg, X, y, cv=cv, 
      scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(ExtraTreesReg, X, y, cv=cv, scoring='neg_mean_absolute_error').std())


ExtraTreesRegcrossMAE = cross_val_score(ExtraTreesReg, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of Extra Tree Regression(kfold cross validation):',ExtraTreesRegcrossMAE)


scoresExtraTreesReg = list()
yPredictCrossExtraTreesReg = list()
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	ExtraTreesReg.fit(train_X, train_y)
	# evaluate model
	yhatExtraTreesReg = ExtraTreesReg.predict(test_X)
	yPredictCrossExtraTreesReg.append(yhatExtraTreesReg)
	maeExtraTreesReg = mean_absolute_error(test_y, yhatExtraTreesReg)
	# store score
	scoresExtraTreesReg.append(maeExtraTreesReg)
    
	print('> ', maeExtraTreesReg)
    
# summarize model performance
mean_s, std_s = mean(scoresExtraTreesReg), std(scoresExtraTreesReg)
print('Mean(ExtraTreesReg): %.3f, Standard Deviation(ExtraTreesReg): %.3f' % (mean_s, std_s))
########
## GradientBoostingRegressor
########

GBReg = GradientBoostingRegressor(alpha=0.85, ccp_alpha=0.0, criterion='mae',
                          init=None, learning_rate=1 , loss='huber', max_depth=3)
#                          ,max_features = None,n_estimators=100)


#for i in range(j): 
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle = True)
#    
#
#    GBReg .fit(X_train, y_train)
#    y_pred_Test =GBReg .predict(X_test)
#    y_pred_Train = GBReg .predict(X_train)
#
#    maeTest = mean_absolute_error(y_test, y_pred_Test)
#    maeTrain = mean_absolute_error(y_train, y_pred_Train)
#
#    
#    MAETest.append(maeTest)
#    MAETrain.append(maeTrain)
#    
#maeTrain_final = np.mean(MAETrain)
#maeTest_final = np.mean(MAETest)
#
#print('Mean absolute error of train dataset Final (Gradient Boosting Regression):', maeTrain_final,'SD:',np.std(MAETrain))
#print('Mean absolute error of test dataset Final (Gradient Boosting regression):', maeTest_final,'SD:',np.std(MAETest))

scoresGBReg = list()
yPredictCrossGBReg = list()
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	GBReg.fit(train_X, train_y)
	# evaluate model
	yhatGBReg = GBReg.predict(test_X)
	yPredictCrossGBReg.append(yhatGBReg)
	maeGBReg = mean_absolute_error(test_y, yhatGBReg)
	# store score
	scoresGBReg.append(maeGBReg)
    
	print('> ', maeGBReg)
    
# summarize model performance
mean_s, std_s = mean(scoresGBReg), std(scoresGBReg)
print('Mean(GBReg): %.3f, Standard Deviation(GBReg): %.3f' % (mean_s, std_s))

# cross Validation

print('Cross validation score for Gradient Boosting Regression (mean):', cross_val_score(GBReg, X, y, cv=cv, 
      scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(GBReg, X, y, cv=cv, scoring='neg_mean_absolute_error').std())


GBRegcrossMAE = cross_val_score(GBReg, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of Gradient Boosting Regression(kfold cross validation):',GBRegcrossMAE)




########
## Ensemble methods
########
from sklearn.ensemble import VotingRegressor
#r1 = LinearRegression()
#r1 = LogReg
MLP = MLPReg
MLPn = MLPRegressor (random_state=1, max_iter=500)
BaggingReg = BaggingReg
#r1 = ExtraTreesReg
#r1 = RegSVR
r1 = AdaBoostReg
r2 = RandomForestRegressor(n_estimators=10, random_state=0)
#r3 = DecisionTreeRegressor()
r3 = DTReg
r4 = LogRegLasso


#EnReg = VotingRegressor([('lr', r1), ('rf', r2), ('dt', r3),('LogRess',r4)])
EnReg = VotingRegressor([('dt', r3), ('AdaBoostReg', r1)])
#EnReg = VotingRegressor([('dt', r3), ('ExtraTrees', ExtraTreesReg)])
#EnReg = VotingRegressor([('BaggingReg', BaggingReg), ('RFReg', r2)])
#EnReg = VotingRegressor([('dt', r3), ('RFReg', r2)])
#EnReg = VotingRegressor([('dt', r3)])
#EnReg = VotingRegressor([('BaggingReg', BaggingReg), ('AdaBoostReg', r1)]) 
#EnReg = VotingRegressor([('MLP', MLP), ('AdaBoostReg', r2)]) 
#EnReg = VotingRegressor([('dt', r3), ('BaggingReg', BaggingReg)])
#EnReg = VotingRegressor([('BaggingReg', BaggingReg), ('DT', DTReg),('SVR', RegSVR)])
#EnReg = VotingRegressor(estimators=[
#       ('DT', DecisionTreeRegressor()),('SVR', SVR()), ('MLP',MLPRegressor())])
#    
#('DT', DTReg), ('GB', GBReg), ('ExtraTrees', ExtraTreesReg), ('MLP',MLPReg)    

#for i in range(j): 
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle = True)
#    
#
#    EnReg .fit(X_train, y_train)
#    y_pred_Test =EnReg .predict(X_test)
#    y_pred_Train = EnReg .predict(X_train)
#
#    maeTest = mean_absolute_error(y_test, y_pred_Test)
#    maeTrain = mean_absolute_error(y_train, y_pred_Train)
#
#    
#    MAETest.append(maeTest)
#    MAETrain.append(maeTrain)
#    
#maeTrain_final = np.mean(MAETrain)
#maeTest_final = np.mean(MAETest)
#
#print('Mean absolute error of train dataset Final (Ensemble voting Regression):', maeTrain_final,'SD:',np.std(MAETrain))
#print('Mean absolute error of test dataset Final (Ensemble voting regression):', maeTest_final,'SD:',np.std(MAETest))
#

# cross Validation

print('Cross validation score for Ensemble voting Regression (mean):', cross_val_score(EnReg, X, y, cv=cv, 
      scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(EnReg, X, y, cv=cv, scoring='neg_mean_absolute_error').std())

EnRegcrossMAE = cross_val_score(EnReg, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of Ensemble voting Regression(kfold cross validation):',EnRegcrossMAE)

y_predEnReg = cross_val_predict(EnReg, X, y, cv=10)



scoresEnReg = list()
yPredictCrossEnReg = list()
kfold = KFold(n_splits=10, shuffle=True, random_state=None)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	EnReg.fit(train_X, train_y)
	# evaluate model
	yhatEnReg = EnReg.predict(test_X)
	yPredictCrossEnReg.append(yhatEnReg)
	maeEnReg = mean_absolute_error(test_y, yhatEnReg)
	# store score
	scoresEnReg.append(maeEnReg)
    
	print('> ', maeEnReg)
    
# summarize model performance
mean_s, std_s = mean(scoresEnReg), std(scoresEnReg)
print('Mean(EnReg): %.3f, Standard Deviation(EnReg): %.3f' % (mean_s, std_s))








from sklearn.ensemble import StackingRegressor
estimators = [('dt', r3), ('ExtraTrees', ExtraTreesReg)]
#estimators = [('dt', r3), ('AdaBoostReg', r1)]

StackReg = StackingRegressor(estimators=estimators, final_estimator=r3)


print('Cross validation score for Stacking regressor (mean):', cross_val_score(StackReg, X, y, cv=cv, 
      scoring='neg_mean_absolute_error').mean()*(-1))
print('SD:', cross_val_score(StackReg, X, y, cv=cv, scoring='neg_mean_absolute_error').std())


StackRegcrossMAE = cross_val_score(StackReg, X, y, cv=cv, scoring='neg_mean_absolute_error')*(-1)
print('Mean absoulte errors of Stacking regressor(kfold cross validation):',StackRegcrossMAE)

y_predEnReg = cross_val_predict(StackReg, X, y, cv=10)



scoresStackReg = list()
yPredictCrossStackReg = list()
kfold = KFold(n_splits=10, shuffle=True, random_state=None)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	
	StackReg.fit(train_X, train_y)
	# evaluate model
	yhatStackReg = StackReg.predict(test_X)
	yPredictCrossStackReg.append(yhatStackReg)
	maeStackReg = mean_absolute_error(test_y, yhatStackReg)
	# store score
	scoresStackReg.append(maeStackReg)
    
	print('> ', maeStackReg)
    
# summarize model performance
mean_s, std_s = mean(scoresStackReg), std(scoresStackReg)
print('Mean(StackReg): %.3f, Standard Deviation(StackReg): %.3f' % (mean_s, std_s))


##RESULTS
resultsCrosscv = [maeKNNCross, RadNeigh,RegSVRcrossMAE,RFRcrossMAE,LogRegcrossMAE,LogRegRidgecrossMAE,LogRegLassocrossMAE,
           gpRegcrossMAE,MLPRegcrossMAE,AdaBoostRegcrossMAE,DTRegcrossMAE,BRidgeRegcrossMAE,BaggingRegcrossMAE,
           ExtraTreesRegcrossMAE,GBRegcrossMAE,EnRegcrossMAE,StackRegcrossMAE]



ResultsCV = np.reshape(resultsCrosscv, (17, 10))



resultsCrosskfold = [scoresKNN, scoresRadNeigh,scoresRegSVR,scoresRegRFR,scoresLogReg,scoresLogRegRidge,
                     scoresLogRegLasso,scoresgpReg,scoresMLPReg,scoresAdaBoostReg,scoresDTReg,
                     scoresBRidgeReg,scoresBaggingReg,scoresExtraTreesReg,scoresGBReg,scoresEnReg,
                     scoresStackReg]



ResultsKFOLD = np.reshape(resultsCrosskfold, (17, 10))
