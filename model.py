# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 21:48:12 2021

@author: Mithilesh
"""

import numpy as np
import pandas as pd
import pickle as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
#load Billboard data set
df1=pd.read_csv('BillBoard_Features.csv')
#remove SpotifyID column
df1.drop('SpotifyID',axis=1,inplace=True)
#since null values are very less 0.01 & 0.02% so we remove those observations
df1=df1.dropna()
#so we have -999 value in Mode and key column 
#mode column should contain values 0,1 so we have to replace -999 as null
df1['mode'].replace(-999,np.nan,inplace=True)
#so we got 10 observations having null values
#since mode is target variable
#so we remove null values from dataset instead of replacing with mean or median
df1=df1.dropna()
#load MSD_Features data set
df2=pd.read_csv('MSD_Features.csv')
#rename columns
df2=df2.rename({'0':'Artist','1':'Album','2':'Track','3':'Year'},axis=1)
#remove ID columns
df2.drop(['4','5'],axis=1,inplace=True)
df=pd.concat([df1,df2])
#removing duplicates
df.sort_values('Track')
df.drop_duplicates(subset='Track',keep=False,inplace=True)
#we have more than 50% of values are null in column Album and Year
#so we remove those columns
df.drop(['Album','Year'],axis=1,inplace=True)
#change order of columns
df=df[['Track','Artist','key','danceability','energy','speechiness','acousticness','instrumentalness','liveness',
'valence','tempo','duration_ms','loudness','mode']]
df['mode']=df['mode'].astype('int64')
df_new=df.drop(['Track','Artist','key'],axis=1)
X=df_new.drop('mode',axis=1)#input variables
Y=df_new['mode']#target variable
#split data into two parts
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
#it is very important to perform feature scaling here
#because feature values lie in different ranges
obj=StandardScaler()
x_train=obj.fit_transform(x_train)
x_test=obj.transform(x_test)
#training model with Logistic regression
#create object of class Logistic Regression
lr=LogisticRegression()
#fit model
lr.fit(x_train,y_train)
lr.predict(x_test)
p.dump(lr,open('dp.pkl','wb'))
model=p.load(open('dp.pkl','rb'))
model.predict(x_test)