import sys

from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from sonHal10 import Ui_MainWindow

class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_win)
        self.ui.stackedWidget.setCurrentWidget(self.ui.boruta_widget)

        self.ui.radioButton_4.setChecked(True)
        self.ui.radioButton_5.setChecked(True)
        self.ui.radioButton_3.clicked.connect(self.show_page)
        self.ui.radioButton_4.clicked.connect(self.show_page)
        self.ui.radioButton_2.clicked.connect(self.show_page)
        self.ui.radioButton.clicked.connect(self.show_page)
        self.ui.pushButton.clicked.connect(self.calculate)

        self.ui.lineEdit.setValidator(QIntValidator())
        self.ui.lineEdit_2.setValidator(QIntValidator())
        self.ui.lineEdit_3.setValidator(QIntValidator())
        self.ui.lineEdit_4.setValidator(QIntValidator())
        self.ui.lineEdit_5.setValidator(QIntValidator())
        


    def show(self):
        self.main_win.show()

    def show_page(self):
        if self.ui.radioButton.isChecked():
            self.ui.stackedWidget.setCurrentWidget(self.ui.logistic_widget)
        if self.ui.radioButton_2.isChecked() :
            self.ui.stackedWidget.setCurrentWidget(self.ui.relief_widget)
        if self.ui.radioButton_3.isChecked() :
            self.ui.stackedWidget.setCurrentWidget(self.ui.chi_widget)
        if self.ui.radioButton_4.isChecked():
            self.ui.stackedWidget.setCurrentWidget(self.ui.boruta_widget)

    def calculate(self):
        
        self.getValues()
    
        self.ui.lineEdit.setText("")
        self.ui.lineEdit_2.setText("")
        self.ui.lineEdit_3.setText("")
        self.ui.lineEdit_4.setText("")
        self.ui.lineEdit_5.setText("")
        self.ui.checkBox_6.setChecked(False)
        self.ui.checkBox_7.setChecked(False)
        self.ui.checkBox_8.setChecked(False)
      
        
    def machineCalculate(self,val_duration,val_credit,val_dispIncome,val_residence,val_age,valrad_0to200,valrad_smaller0,valrad_nocheck,valcheck_critical,valcheck_smaller100,valcheck_other,valcheck_housingown):
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from imblearn.over_sampling import SMOTE
        import keras
        from keras.models import Sequential
        from keras.layers import Dense 
        from keras.layers.core import Dropout
        
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
        
        
        ## Feature Selection Boruta
        
        sonar_x = df_2.iloc[:,0:61].values.astype(int)
        sonar_y = df_2.iloc[:,62:].values.ravel().astype(int)
        
        
        #Selected Features by Boruta
        sonar_x_selected = sonar_x[:,[0,1,2,3,4,7,8,10,12,28,48,51]]
        
        
        #Train/Test Split
        x_train,x_test,y_train,y_test=train_test_split(sonar_x_selected,sonar_y,test_size=0.2,random_state=0)
        
        #Dataset Balancing
        smt = SMOTE()
        x_train, y_train = smt.fit_resample(x_train, y_train)
        y_train = y_train.astype(int)
        
        x_test, y_test = smt.fit_resample(x_test, y_test)
        y_test = y_test.astype(int)
        
        #Standard Scaling
        sc=StandardScaler()
        X_train= sc.fit_transform(x_train)
        X_test=sc.fit_transform(x_test)
        
        
        
        #Neural Network
        
        classifier = Sequential()
        classifier.add(Dense(7 ,kernel_initializer='uniform',activation='tanh',input_dim=X_train.shape[1]))
        classifier.add(Dropout(0.5))
        classifier.add(Dense(7 ,kernel_initializer='uniform',activation='relu'))

        classifier.add(Dense(1 ,kernel_initializer='uniform',activation='sigmoid'))

        classifier.compile(optimizer='adam',loss= 'binary_crossentropy', metrics= ['accuracy'])
        
        classifier.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=45)
        nn_pred=classifier.predict(X_test)
        
        nn_pred=(nn_pred > 0.5)
        
        
        tahmin = np.array([val_duration,val_credit,val_dispIncome,val_residence,val_age,valrad_0to200,valrad_smaller0,valrad_nocheck,valcheck_critical,valcheck_smaller100,valcheck_other,valcheck_housingown]).reshape(1,12)
        print(tahmin)
        tahmin=sc.transform(tahmin)
        print(tahmin)
        classifier.predict_classes(tahmin)
        print(classifier.predict_classes(tahmin)[0][0])
        
        if classifier.predict_classes(tahmin)[0][0] ==0:
            self.ui.frame.setStyleSheet("background-color:qlineargradient(spread:pad, x1:0.523, y1:0, x2:0.534, y2:1, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(183, 0, 0, 255))")
            self.ui.label_6.setText("bad risk")
        else:
            self.ui.frame.setStyleSheet("background-color:qlineargradient(spread:pad, x1:1, y1:0.511545, x2:1, y2:0.046, stop:0 rgba(0, 58, 11, 255), stop:1 rgba(20, 30, 29, 255))")
            self.ui.label_6.setText("good risk")
        
        
        
    def getValues(self):
        val_age= int(self.ui.lineEdit.text())
        val_credit=int(self.ui.lineEdit_3.text())
        val_duration=int(self.ui.lineEdit_2.text())
        val_dispIncome=int(self.ui.lineEdit_4.text())
        val_residence=int(self.ui.lineEdit_5.text())
        
        if(self.ui.radioButton_7.isChecked()):
            valrad_0to200=1
        else:
            valrad_0to200=0
        
        if(self.ui.radioButton_9.isChecked()):
            valrad_smaller0=1
        else:
            valrad_smaller0=0
        
        if(self.ui.radioButton_8.isChecked()):
            valrad_nocheck=1
        else:
            valrad_nocheck=0
        
        ###################################################################
        if(self.ui.checkBox_6.isChecked()):
            valcheck_smaller100=1
        else:
            valcheck_smaller100=0
        
        
        if(self.ui.checkBox_7.isChecked()):
            valcheck_housingown=1
        else:
            valcheck_housingown=0
        
        
        if(self.ui.checkBox_8.isChecked()):
            valcheck_other=1
        else:
            valcheck_other=0
        
        
        if(self.ui.checkBox_9.isChecked()):
            valcheck_critical=1
        else:
            valcheck_critical=0
        
        self.machineCalculate(val_duration,val_credit,val_dispIncome,val_residence,val_age,valrad_0to200,valrad_smaller0,valrad_nocheck,valcheck_critical,valcheck_smaller100,valcheck_other,valcheck_housingown)
        

if __name__ == '__main__':
    app=QApplication(sys.argv)
    main_win=MainWindow()
    main_win.show()
    sys.exit(app.exec_())
    
    
    

    
    