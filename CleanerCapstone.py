import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from boruta import BorutaPy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from imblearn.over_sampling import SMOTE



pd.set_option('display.max_columns', None)


import warnings
warnings.simplefilter('ignore', DeprecationWarning)
                      
                      
df=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",sep=" ",header=None)
headers=["Status of existing checking account","Duration in month","Credit history",\
         "Purpose","Credit amount","Savings account/bonds","Present employment since",\
         "Installment rate in percentage of disposable income","Personal status and sex",\
         "Other debtors / guarantors","Present residence since","Property","Age in years",\
        "Other installment plans","Housing","Number of existing credits at this bank",\
        "Job","Number of people being liable to provide maintenance for","Telephone","foreign worker","Cost Matrix(Risk)"]
df.columns=headers
df.to_csv("german_data_credit_cat.csv",index=False) #save as csv file

#for structuring only
Status_of_existing_checking_account={'A14':"no checking account",'A11':"<0 DM", 'A12': "0 <= <200 DM",'A13':">= 200 DM "}
df["Status of existing checking account"]=df["Status of existing checking account"].map(Status_of_existing_checking_account)

Credit_history={"A34":"critical account","A33":"delay in paying off","A32":"existing credits paid back duly till now","A31":"all credits at this bank paid back duly","A30":"no credits taken"}
df["Credit history"]=df["Credit history"].map(Credit_history)

Purpose={"A40" : "car (new)", "A41" : "car (used)", "A42" : "furniture/equipment", "A43" :"radio/television" , "A44" : "domestic appliances", "A45" : "repairs", "A46" : "education", 'A47' : 'vacation','A48' : 'retraining','A49' : 'business','A410' : 'others'}
df["Purpose"]=df["Purpose"].map(Purpose)

Saving_account={"A65" : "no savings account","A61" :"<100 DM","A62" : "100 <= <500 DM","A63" :"500 <= < 1000 DM", "A64" :">= 1000 DM"}
df["Savings account/bonds"]=df["Savings account/bonds"].map(Saving_account)

Present_employment={'A75':">=7 years", 'A74':"4<= <7 years",  'A73':"1<= < 4 years", 'A72':"<1 years",'A71':"unemployed"}
df["Present employment since"]=df["Present employment since"].map(Present_employment)

Personal_status_and_sex={ 'A95':"female:single",'A94':"male:married/widowed",'A93':"male:single", 'A92':"female:divorced/separated/married", 'A91':"male:divorced/separated"}
df["Personal status and sex"]=df["Personal status and sex"].map(Personal_status_and_sex)

Other_debtors_guarantors={'A101':"none", 'A102':"co-applicant", 'A103':"guarantor"}
df["Other debtors / guarantors"]=df["Other debtors / guarantors"].map(Other_debtors_guarantors)

Property={'A121':"real estate", 'A122':"savings agreement/life insurance", 'A123':"car or other", 'A124':"unknown / no property"}
df["Property"]=df["Property"].map(Property)

Other_installment_plans={'A143':"none", 'A142':"store", 'A141':"bank"}
df["Other installment plans"]=df["Other installment plans"].map(Other_installment_plans)

Housing={'A153':"for free", 'A152':"own", 'A151':"rent"}
df["Housing"]=df["Housing"].map(Housing)


Job={'A174':"management/ highly qualified employee", 'A173':"skilled employee / official", 'A172':"unskilled - resident", 'A171':"unemployed/ unskilled  - non-resident"}
df["Job"]=df["Job"].map(Job)

Telephone={'A192':"yes", 'A191':"none"}
df["Telephone"]=df["Telephone"].map(Telephone)

foreign_worker={'A201':"yes", 'A202':"no"}
df["foreign worker"]=df["foreign worker"].map(foreign_worker)

risk={1:"Good Risk", 2:"Bad Risk"}
df["Cost Matrix(Risk)"]=df["Cost Matrix(Risk)"].map(risk)

#df = df.sample(frac=1).reset_index(drop=True)

#OneHotEncoding
df_2 = pd.get_dummies(df,drop_first=False)

## Data Sampling
rows = df_2.sample(frac =.010) 


## Feature Selection Boruta

sonar_x = df_2.iloc[:,0:61].values.astype(int)
sonar_y = df_2.iloc[:,62:].values.ravel().astype(int)


rf = RandomForestClassifier(n_jobs=-1, max_depth=10)

boruta_feature_selector = BorutaPy(rf,n_estimators='auto',perc = 90,random_state=0)

boruta_feature_selector.fit(sonar_x,sonar_y)

boruta_feature_selector.support_
boruta_feature_selector.ranking_

#Selected Features by Boruta
sonar_x_selected = sonar_x[:,[0,1,2,3,4,7,8,10,12,28,48,51]]


#Train/Test Split
x_train,x_test,y_train,y_test=train_test_split(sonar_x_selected,sonar_y,test_size=0.2,random_state=0)
smt = SMOTE()
x_train, y_train = smt.fit_sample(x_train, y_train)
y_train = y_train.astype(int)

x_test, y_test = smt.fit_sample(x_test, y_test)
y_test = y_test.astype(int)



sc=StandardScaler()
X_train= sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

# Algorithm comparison
seed = 0

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('XGB', XGBClassifier()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

for name, model in models:
        kfold = KFold(n_splits=3, random_state=seed,shuffle=False)
        cv_results = cross_val_score(model, X_test, y_test, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
fig = plt.figure(figsize=(11,6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


#Chi2 Feature Selection

X_new = SelectKBest(chi2, k=15).fit(sonar_x ,sonar_y)

X_new.scores_
X_new.get_support()


#Selected Features by Chi2
sonar_x_chiSelected = sonar_x[:,[0,1,4,7,8,10,11,12,15,18,28,30,33,44,46]]

#Train/Test Split

xchi_train,xchi_test,ychi_train,ychi_test=train_test_split(sonar_x_chiSelected,sonar_y,test_size=0.33,random_state=0)

smt = SMOTE()
xchi_train, ychi_train = smt.fit_sample(xchi_train, ychi_train)
ychi_train = ychi_train.astype(int)

xchi_test, ychi_test = smt.fit_sample(xchi_test, ychi_test)
ychi_test = ychi_test.astype(int)



sc=StandardScaler()
Xchi_train= sc.fit_transform(xchi_train)
Xchi_test=sc.fit_transform(xchi_test)


for name, model in models:
        kfold = KFold(n_splits=3, random_state=seed,shuffle=False)
        cv_results = cross_val_score(model, Xchi_test, ychi_test, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
fig = plt.figure(figsize=(11,6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()



#Neural Network

import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers.core import Dropout

classifier = Sequential()
classifier.add(Dense(7 ,kernel_initializer='uniform',activation='relu',input_dim=12))



classifier.add(Dense(7 ,kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(1 ,kernel_initializer='uniform',activation='sigmoid'))


classifier.compile(optimizer='adam',loss= 'binary_crossentropy', metrics= ['accuracy'])

history=classifier.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=110)
nn_pred=classifier.predict(X_test)

nn_pred=(nn_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,nn_pred)
print(cm)

import collections

collections.Counter(y_train)