import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
#import nslkdd
#import unsw

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from  sklearn.tree  import  DecisionTreeClassifier
from  sklearn.ensemble  import  RandomForestClassifier , VotingClassifier
from  sklearn.linear_model  import  LogisticRegression
from  sklearn.metrics  import  accuracy_score , roc_curve , auc , f1_score, confusion_matrix, classification_report
from  sklearn.preprocessing  import  LabelEncoder , MinMaxScaler
from  sklearn  import svm #SVC , LinearSVC
from  sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors
#draw roc cuver
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.metrics import (geometric_mean_score, make_index_balanced_accuracy)
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report

from scipy import interp

from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing

def svm(X_tr, Y_tr, X_te, Y_te):
    if Y_tr.shape[1] > 1:
        Y_tr = np.argmax(Y_tr, axis=1)
        Y_te = np.argmax(Y_te, axis=1)
    parameters =  [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-2, 1e-1, 1],
                     'C': [1]}]
                    #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters,cv= 5)
   
    clf.fit(X_tr, Y_tr)
    y_pred = clf.predict(X_te)
    acc= accuracy_score(Y_te, y_pred)
    fpr_vot , tpr_vot , _ = roc_curve(Y_te , y_pred , pos_label =1,  drop_intermediate=False)
    roc_auc_vot = auc(fpr_vot , tpr_vot)
    cmat = classification_report_imbalanced(Y_te, y_pred)
    print ("SVM")
   
    print (cmat)
   
    cnf_matrix = confusion_matrix(Y_te, y_pred)
    print (cnf_matrix)
    geo = geometric_mean_score(Y_te,y_pred)
    f1 = f1_score(Y_te, y_pred, average='micro')
    print('The geometric mean is {}'.format(geo))
    print('The auc is {}'.format(roc_auc_vot))
    print('The f1 is {}'.format(f1))
  
    return acc

def randomforest(X_tr, Y_tr, X_te, Y_te):
    if Y_tr.shape[1] > 1:
        Y_tr = np.argmax(Y_tr, axis=1)
        Y_te = np.argmax(Y_te, axis=1)
    rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=40, oob_score = True)
    clf = RandomForestClassifier(n_estimators=100, max_depth=80,random_state=0)

    param_grid = {'n_estimators': [5,10,20,40,80,150]}
    clf = GridSearchCV(estimator=rfc, param_grid=param_grid)
    clf.fit(X_tr, Y_tr)
    y_pred = clf.predict(X_te)
    acc= accuracy_score(Y_te, y_pred)
    fpr_vot , tpr_vot , _ = roc_curve(Y_te , y_pred , pos_label =1,  drop_intermediate=False)
    roc_auc_vot = auc(fpr_vot , tpr_vot)
    cmat = classification_report(Y_te, y_pred)
    print (cmat)
    geo = geometric_mean_score(Y_te,y_pred)
    f1 = f1_score(Y_te, y_pred, average='micro')
    print('The geometric mean is {}'.format(geo))
    cnf_matrix = confusion_matrix(Y_te, y_pred)
    print (cnf_matrix)
    print('The auc is {}'.format(roc_auc_vot))
    print('The f1 is {}'.format(f1))
   
    return acc
def decisiontree(X_tr, Y_tr, X_te, Y_te):
     if Y_tr.shape[1] > 1:
        Y_tr = np.argmax(Y_tr, axis=1)
        Y_te = np.argmax(Y_te, axis=1)
     param_grid = {'max_depth': [5,6,7,8,9,10,50,100]}
     tree = GridSearchCV(DecisionTreeClassifier(), param_grid)

     tree.fit(X_tr, Y_tr)
     y_pred = tree.predict(X_te)
     acc= accuracy_score(Y_te, y_pred)
     fpr_vot , tpr_vot , _ = roc_curve(Y_te , y_pred , pos_label =1,  drop_intermediate=False)
     roc_auc_vot = auc(fpr_vot , tpr_vot)
     cmat = classification_report_imbalanced(Y_te, y_pred)
     print ("Decision tree")
     print (cmat)
     cnf_matrix = confusion_matrix(Y_te, y_pred)
     print (cnf_matrix)
     geo = geometric_mean_score(Y_te,y_pred)
     f1 = f1_score(Y_te, y_pred, average='micro')
     print('The geometric mean is {}'.format(geo))
     print('The auc is {}'.format(roc_auc_vot))
     print('The f1 is {}'.format(f1))
    
     return  acc
    
  