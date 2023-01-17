# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 14:26:12 2022

@author: Mohd Ariz Khan
"""
# Importing the data
import pandas as pd
df = pd.read_csv("bank-full.csv")
df.shape
df.head()
df.info()
df["y"].value_counts()

# One-Hot Encoding of categrical variables
df = pd.get_dummies(df,columns=['job','marital','education','contact','poutcome'])
df

# To see all columns
pd.set_option("display.max.columns", None)
df

df.info()

# Custom Binary Encoding of Binary o/p variables 
import numpy as np
df['default'] = np.where(df['default'].str.contains("yes"), 1, 0)
df['housing'] = np.where(df['housing'].str.contains("yes"), 1, 0)
df['loan'] = np.where(df['loan'].str.contains("yes"), 1, 0)
df['y'] = np.where(df['y'].str.contains("yes"), 1, 0)
df

# Find and Replace Encoding for month categorical varaible
df['month'].value_counts()

order={'month':{'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}}

df = df.replace(order)
df

df.info()

# Dividing our data into input and output variables
X = pd.concat([df.iloc[:,0:11],df.iloc[:,12:]],axis=1)
Y = df.iloc[:,11]

# step5: Data Transformation
from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()
MM_X = MM.fit_transform(X)

# Data Partition
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test = train_test_split(MM_X,Y , test_size=0.3)

df.shape
X_train.shape
X_test.shape

# Model fitting
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)

Y_pred_train = logreg.predict(X_train)
Y_pred_test = logreg.predict(X_test)

# Model Predictions (Predict for x dataset)
Y_pred=logreg.predict(X)
Y_pred

Y_pred_df=pd.DataFrame({'actual_Y':Y,'Y_pred_prob':Y_pred})
Y_pred_df

#=================================================================
# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,Y_pred)
confusion_matrix

from sklearn.metrics import accuracy_score
ac1 =  accuracy_score(Y_train,Y_pred_train)
ac2 =  accuracy_score(Y_test,Y_pred_test)
print("Training score:", ac1.round(3))
print("Test score:", ac2.round(3))

from sklearn.metrics import recall_score,precision_score,f1_score

rs =  recall_score(Y,Y_pred)
print("Sensitivity/Recall score:", rs.round(3))

ps =  precision_score(Y,Y_pred)
print("precision score:", ps.round(3))

f1s =  f1_score(Y,Y_pred)
print("F1 score:", f1s.round(3))

TN = confusion_matrix[0,0]
FP = confusion_matrix[1,0]
TNR = TN/(TN + FP)
print("Specificity score:", TNR.round(3))

#=================================================================
from sklearn.metrics import roc_curve,roc_auc_score
probs =logreg.predict_proba(X)[:,1]
fpr, tpr,_ =  roc_curve(Y,probs)

plt.plot(fpr,tpr,color='red')

import matplotlib.pyplot as plt
plt.plot(fpr,tpr,color = 'red',label='logit model(area  = %0.2f)'%auc)
plt.plot([0,1],[0,1],'k--')
plt.ylabel('TPR - True Positive Rate')
plt.xlabel('FPR - False Positive Rate')
plt.show()

auc = roc_auc_score(Y,probs)
print("Area under curve score:", auc.round(3))
#=================================================================

'''
    