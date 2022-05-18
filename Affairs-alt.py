import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

#Importing Data
Affairs = pd.read_csv("C:/Users/personal/Desktop/Affairs.csv",sep=",")


#removing Index
Affairs1 = Affairs.drop('Index', axis = 1)
Affairs1.head()
Affairs1.describe()
Affairs1.isna().sum() # # AS there are no NA values ,no imputation is required
Affairs1.columns

# creating a column
Affairs1['affairs'] = np.where(Affairs1.naffairs > 0,1,0)
Affairs1.drop(["naffairs"], axis = 1 , inplace = True )
Affairs1

# Rearraning of columns
Affairs = Affairs1.iloc[:, [17, 0,1,2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
Affairs.columns

# Model building 
# import statsmodels.formula.api as sm

logit_model = sm.logit('affairs ~ kids+vryunhap+unhap+avgmarr+hapavg+vryhap+antirel+notrel+slghtrel+smerel+vryrel+yrsmarr1+yrsmarr2+yrsmarr3+yrsmarr4+yrsmarr5+yrsmarr6', data = Affairs).fit()
#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(Affairs.iloc[:,1:])

fpr,tpr,thersholds = roc_curve(Affairs.affairs, pred)
optimal_idx = np.argmax(tpr-fpr)
optimal_threshold = thersholds[optimal_idx]
optimal_threshold

import pylab as p1

roc_auc = auc(fpr,tpr)
print("Area under the ROC curve :%f" % roc_auc)

# Filling the cells with zero
Affairs["pred"] = np.zeros(600)

# Taking Thershold value nad above the prob value will be treated as correct value
Affairs.loc[pred > optimal_trersholds, "pred"] = 1

# Classification Report
classification = classification_report(Affairs["pred"],Affairs[affairs])
classification

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(Affairs, test_size = 0.3) # 30% test data

model = sm.logit('affairs ~ kids+vryunhap+unhap+avgmarr+hapavg+vryhap+antirel+notrel+slghtrel+smerel+vryrel+yrsmarr1+yrsmarr2+yrsmarr3+yrsmarr4+yrsmarr5+yrsmarr6', data = train_data).fit()
#summary
model.summary2() # for AIC
model.summary()

# Prediction on test data
test_pred = logit_model.predict(test_data)

# Creating new columns for sorting predicted class of attorney
#filling the cells with zero
test_data["test_pred"] = np.zeros()

# Taking Thershold value nad above the prob value will be treated as correct value
test_data.loc[test_pred > optimal_trersholds ,"test_pred"] = 1

# Confusion Matrix
Confusion_matrix = pd.crosstab(test_data.test_pred, test_data['affairs'])
Confusion_matrix

accuracy_test = ()/()
accuracy_test

# Classification Report
classification_test = classification_report(test_data["test_pred"],test_data["affairs"])
classification_test

#Roc Curve and AUC
fpr,tpr,thershold = metric.roc_curve(test_data["affairs"],test_pred)

# Area under the curve
roc_auc_test = metrics.auc(fpr,tpr)
roc_auc_test()

# Prediction on test data
train_pred = model.predict(train_data.iloc[:,1:])

# Creating new columns for sorting predicted class of attorney
#filling the cells with zero
train_data["train_pred"] = np.zeros()

# Taking Thershold value nad above the prob value will be treated as correct value
train_data.loc[train_pred > optimal_trersholds ,"train_pred"] = 1

# Confusion Matrix
Confusion_matrix = pd.crosstab(train_data.train_pred, train_data['affairs'])
Confusion_matrix

accuracy_train = ()/()
accuracy_train

                      

