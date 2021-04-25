import numpy as np
import pandas as pd
import seaborn as sns
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

df = df.sample(frac=1).reset_index(drop=True)

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
x_train,x_test,y_train,y_test=train_test_split(sonar_x_selected,sonar_y,test_size=0.33,random_state=0)
smt = SMOTE()
x_train, y_train = smt.fit_sample(x_train, y_train)
y_train = y_train.astype(int)

x_test, y_test = smt.fit_sample(x_test, y_test)
y_test = y_test.astype(int)

folds = KFold(n_splits = 3, shuffle = False, random_state = 0)
scores = []

for n_fold, (train_index, valid_index) in enumerate(folds.split(sonar_x_selected,sonar_y)):
    # print('\n Fold '+ str(n_fold+1 ) + 
    #       ' \n\n train ids :' +  str(train_index) +
    #       ' \n\n validation ids :' +  str(valid_index))
    
    x_train, x_valid = sonar_x_selected[train_index], sonar_x_selected[valid_index]
    y_train, y_valid = sonar_y[train_index], sonar_y[valid_index]
    
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_valid)
    
    
    acc_score = accuracy_score(y_pred, y_valid)
    scores.append(acc_score)
    print('\n Accuracy score for Fold ' +str(n_fold+1) + ' --> ' + str(acc_score)+'\n')

    
print(scores)
print('Avg. accuracy score :' + str(np.mean(scores)))




scores = cross_val_score(rf, sonar_x_selected, sonar_y, cv=3)

print("Avg. cross val score: "+str(scores.mean()))

x_train,x_test,y_train,y_test=train_test_split(sonar_x_selected,sonar_y,test_size=0.33,random_state=0)

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
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
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

X_new = SelectKBest(chi2, k=12).fit(sonar_x ,sonar_y)

X_new.scores_
X_new.get_support()


#Selected Features by Chi2
sonar_x_chiSelected = sonar_x[:,[0,1,4,7,8,10,11,12,15,28,30,46]]

#Train/Test Split

xchi_train,xchi_test,ychi_train,ychi_test=train_test_split(sonar_x_chiSelected,sonar_y,test_size=0.33,random_state=0)



for name, model in models:
        kfold = KFold(n_splits=3, random_state=seed,shuffle=False)
        cv_results = cross_val_score(model, xchi_train, ychi_train, cv=kfold, scoring=scoring)
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

classifier = Sequential()
classifier.add(Dense(7 ,kernel_initializer='uniform',activation='relu',input_dim=12))

classifier.add(Dense(7 ,kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(1 ,kernel_initializer='uniform',activation='sigmoid'))


classifier.compile(optimizer='adam',loss= 'binary_crossentropy', metrics= ['accuracy'])

history=classifier.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=50)
nn_pred=classifier.predict(X_test)

nn_pred=(nn_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,nn_pred)
print(cm)


from matplotlib import pyplot as plt

plt.figure(figsize=(14,3))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()