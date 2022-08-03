# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:33:09 2022

@author: KTong
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import missingno as msno
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
import Visuals as vs
import NN_modules as nn
import CramersV as cv


#%% STATICS
log_dir=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
LOG_PATH = os.path.join(os.getcwd(),'logs',log_dir)
DATA_FILE_PATH=os.path.join(os.getcwd(),'dataset','train.csv')
JOB_PICKLE_PATH=os.path.join(os.getcwd(),'model','job.pkl')
MARITAL_PICKLE_PATH=os.path.join(os.getcwd(),'model','marital.pkl')
EDU_PICKLE_PATH=os.path.join(os.getcwd(),'model','edu.pkl')
DEF_PICKLE_PATH=os.path.join(os.getcwd(),'model','def.pkl')
HSE_LOAN_PICKLE_PATH=os.path.join(os.getcwd(),'model','hseloan.pkl')
IDV_LOAN_PICKLE_PATH=os.path.join(os.getcwd(),'model','idvloan.pkl')
COMM_PICKLE_PATH=os.path.join(os.getcwd(),'model','communication.pkl')
MONTH_PICKLE_PATH=os.path.join(os.getcwd(),'model','month.pkl')
PREV_PICKLE_PATH=os.path.join(os.getcwd(),'model','previouscamp.pkl')
II_PICKLE_PATH=os.path.join(os.getcwd(),'model','ii.pkl')
MMS_PICKLE_PATH=os.path.join(os.getcwd(),'model','mms.pkl')
OHE_PICKLE_PATH=os.path.join(os.getcwd(),'model','ohe.pkl')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'model','model.h5')

#%% DATA LOADING
df=pd.read_csv(DATA_FILE_PATH)
# df_backup=df.copy()
# df=df_backup

#%% DATA INSPECTION
column_names=df.columns
num_data=['customer_age','balance','day_of_month','last_contact_duration',
          'num_contacts_in_campaign','days_since_prev_campaign_contact',
          'num_contacts_prev_campaign']
cat_data=np.array(['job_type','marital','education','default',
                   'housing_loan','personal_loan','communication_type',
                   'month','prev_campaign_outcome','term_deposit_subscribed'])

# Check for null values
df.info()

# No numeric data and objects in observed in categorical and numeric 
# columns respectively

df.isna().sum()
msno.matrix(df)
msno.bar(df)

# 'days_since_prev_campaign_contact' has highest number of null values, 
# followed by 'customer_age','balance','last_contact_duration',
# 'marital','personal_loan','num_contacts_in_campaign'
# NaNs account for 82% in 'days_since_prev_campaign_contact' column hence 
# will be dropped.

# Check for duplicate observations
df.duplicated().sum()

# No duplicated rows present in dataset

# Check for errorneous observations
df[['customer_age','balance','day_of_month','last_contact_duration']].describe()
df[['num_contacts_in_campaign','days_since_prev_campaign_contact']].describe()
df['num_contacts_prev_campaign'].describe()

df['balance'][df['balance']<0]

# 'balance' contain 16 negative values however it is acceptable as cases of 
# overdraft accounts

#%% DATA VISUALISATION
# df.boxplot(rot=90)
visuals=vs.Visualisation()
visuals.cont_plot(df,num_data)
visuals.cat_plot(df,cat_data)
visuals.cat_group_plot(df,cat_data,target='term_deposit_subscribed')

# Preliminary assessment of data:
#     - Participants mostly in age range 30-40.
#     - Most participants has balance of less than 10000.
#     - Highest success rate appears to be in May-Aug.
#     - Most participants who subsribed term deposit has secondary/tertiary 
#       education.
#     - Participants who are on housing/personal loan not likely to have 
#       term deposit.
#     - Promotional strategy via cellular has highest success rate to 
#       subscribe term deposit. 

#%% DATA CLEANING
# Numerize categorical columns
LE=LabelEncoder()
pickle_path=[JOB_PICKLE_PATH,MARITAL_PICKLE_PATH,EDU_PICKLE_PATH,
             DEF_PICKLE_PATH,HSE_LOAN_PICKLE_PATH,IDV_LOAN_PICKLE_PATH,
             COMM_PICKLE_PATH,MONTH_PICKLE_PATH,PREV_PICKLE_PATH]

for index,i in enumerate(np.delete(cat_data,-1)):
    temp=df[i]
    temp[temp.notnull()]=LE.fit_transform(temp[temp.notnull()])
    df[i]=pd.to_numeric(temp,errors='coerce')
    
    with open(pickle_path[index],'wb') as file:
        pickle.dump(LE,file)

# Drop ID and 'days_since_prev_campaign_contact' columns 
df=df.drop(labels=['id','days_since_prev_campaign_contact'],axis=1) 

# Impute NaNs using MICE method
ii=IterativeImputer()
df_clean=ii.fit_transform(df)
df_clean=pd.DataFrame(df_clean)
df_clean.columns=column_names.drop(labels=['id','days_since_prev_campaign_contact'])

with open(II_PICKLE_PATH,'wb') as file:
    pickle.dump(ii,file)

# Return imputed categorical/numeric data to int format
na_columns=['customer_age','balance','last_contact_duration',
            'marital','personal_loan','num_contacts_in_campaign']
df_clean.loc[:,na_columns] = np.floor(df_clean.loc[:,na_columns]).astype('int')

# Recheck imputed dataset
df_clean.isna().sum()
df_clean.info()

#%% FEATURE SELECTION
# Check correlation of numeric features to target
lr=LogisticRegression()

for i in np.delete(num_data,-2):
    lr.fit(np.expand_dims(df_clean[i],axis=-1),df_clean['term_deposit_subscribed'])
    print('{}: R^2 is {}'.format(i,lr.score(np.expand_dims(df_clean[i],axis=-1),df_clean['term_deposit_subscribed'])))

# All numeric features listed in 'num_data' displayed at least89% accuracy, 
# inferring the numeric features have good correlation to target feature

# Check correlation of categorical features to target
func=cv.CramersV()
for i in cat_data:
    con_mat=pd.crosstab(df_clean[i], df_clean['term_deposit_subscribed']).to_numpy()
    print('{}: accuracy is {}'.format(i,func.cramers_corrected_stat(con_mat)))

# Since all categorical features display negligible to weak correlation to 
# output they will be discarded

# Finalised feature
X=df_clean.loc[:,np.delete(num_data,-2)]
Y=df_clean['term_deposit_subscribed']

#%% PREPROCESSING
# Normalize input features
mms=MinMaxScaler()
X=mms.fit_transform(X)

with open(MMS_PICKLE_PATH,'wb') as file:
    pickle.dump(mms,file)

# Apply OHE on target column
ohe=OneHotEncoder(sparse=False)
Y=ohe.fit_transform(np.expand_dims(Y,axis=-1))

with open(OHE_PICKLE_PATH,'wb') as file:
    pickle.dump(ohe,file)

# Create train and test data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,
                                               random_state=123)
                                               
#%% MODEL BUILDING
nmod=nn.NeuralNetworkModel()
model=nmod.two_layer_model(x_train,y_train,64,32,0.2,activ='softmax')
plot_model(model,show_shapes=(True))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

tensorboard=TensorBoard(log_dir=LOG_PATH)
earlystop=EarlyStopping(monitor='val_loss',patience=10)

hist=model.fit(x_train,y_train,batch_size=128,epochs=100,verbose=2,
               validation_data=(x_test,y_test),
               callbacks=[tensorboard,earlystop])

#%% MODEL EVALUATION
nmod.eval_plot(hist)
nmod.model_eval(model,x_test,y_test,label=['Success','Failure'])

# loss,acc [0.23, 0.90]

#%% MODEL EXPORT
model.save(MODEL_SAVE_PATH)
