#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# In[4]:


class classification_models:
    #Random Forest Classifier:
    def RF_model(self,X,Y,X1,Y1):
        self.X=X
        self.Y=Y
        self.X1=X1
        self.Y1=Y1
        RF_model=RandomForestClassifier(n_estimators=100,class_weight='balanced')
        RF_model.fit(X,Y)
        Y_predict_RF=RF_model.predict(X1)
        print("The RF model accuracy is given as : ",metrics.accuracy_score(Y1,Y_predict_RF))
        print(classification_report(Y1,Y_predict_RF))
    
    #Decision Tree Classifier:
    def DF_model(self,X,Y,X1,Y1):
        self.X=X
        self.Y=Y
        self.X1=X1
        self.Y1=Y1
        DF_model=DecisionTreeClassifier(max_depth=7,random_state=42,class_weight='balanced')
        DF_model.fit(X,Y)
        Y_predict_DF=DF_model.predict(X1)
        print("The DF model accuracy is given as : ",metrics.accuracy_score(Y1,Y_predict_DF))
        print(classification_report(Y1,Y_predict_DF))
        
    #Logistic Regression Classifier:
    def LR_model(self,X,Y,X1,Y1):
        self.X=X       
        self.Y=Y
        self.X1=X1
        self.Y1=Y1
        LR_model=LogisticRegression(class_weight='balanced')
        LR_model.fit(X,Y)
        Y_predict_LR=LR_model.predict(X1)
        print("The LR_model accuracy is given as : ",metrics.accuracy_score(Y1,Y_predict_LR))
        print(classification_report(Y1,Y_predict_LR))

    #Support Vector Machine:
    def SVM_model(self,X,Y,C,g,X1,Y1):
        self.X=X
        self.Y=Y
        self.C=C
        self.g=g
        self.X1=X1
        self.Y1=Y1
        SVM_model = SVC(kernel='rbf',C=C, gamma=g, class_weight='balanced')
        SVM_model.fit(X,Y)
        Y_predict_SVM=SVM_model.predict(X1)
        print("The SVM_accuracy is given as : ",metrics.accuracy_score(Y1,Y_predict_SVM))
        print(classification_report(Y1,Y_predict_SVM))

