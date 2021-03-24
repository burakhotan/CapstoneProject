import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

## Data Visualization


print(df.isnull().sum())
print(df.info())
print(df.head())
print(df.describe())

#################################################################################################

credit_history = df["Credit history"]
get_credit = df["Cost Matrix(Risk)"]

cat_vars = df.iloc[:,[0,2,3,5,6,8,9,11,13,14,16,18,19]]

for col in cat_vars:
    df.groupby([col, 'Cost Matrix(Risk)']).size().unstack().plot(kind='bar', stacked = 'False', figsize=(20, 20))


fig, ax = plt.subplots(1, 3, figsize=(70, 35))
df[df['Cost Matrix(Risk)'] == "Good Risk"][cat_vars[0:-1]].hist(bins = 25, 
  color = "red", ax=ax)
df[df['Cost Matrix(Risk)'] == "Bad Risk"][cat_vars[0:-1]].hist(bins = 25, 
  color = "blue", ax=ax)


num_vars = df.iloc[:,[1,4,7,10,12,15,17]].values



##Label Encoding
import pandas as pd
a = [0,2,3,5,6,8,9,11,13,14,16,18,19,20]
X = df.iloc[:,0:21].values
from sklearn.preprocessing import LabelEncoder 
labelencoder_X = LabelEncoder()


for item in a:
    X[:,item] =labelencoder_X.fit_transform(X[:,item])
    Y = pd.DataFrame(X)
    
#OneHotEncoding
df_2 = pd.get_dummies(df,drop_first=False)

## Data Sampling
rows = Y.sample(frac =.010) 


## Feature Selection Boruta

sonar_x = Y.iloc[:,0:20].values.astype(int)
sonar_y = Y.iloc[:,-1].values.ravel().astype(int)



from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

boruta_feature_selector = BorutaPy(rf,n_estimators='auto',perc = 90,random_state=111)

boruta_feature_selector.fit(sonar_x,sonar_y)

boruta_feature_selector.support_
boruta_feature_selector.ranking_











