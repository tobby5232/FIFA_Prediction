# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:20:00 2022

@author: ch406
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#%%
#Dataframe information

data_full = pd.read_csv('cleanData.csv', encoding='utf-8')
data_full=data_full[['date','home_team','away_team','result',
                     'home_score','away_score','goal_dif',
                      'home_team_odd','away_team_odd','Draw_odd','odd_dif',
                      'ovr_home','att_home','mid_home','defe_home','ovr_away','att_away','mid_away','defe_away',
                      'ovr_dif','att_dif','mid_dif','defe_dif',
                      'Previous_Rank_home','Total_Points_home','rank_dif',
                      'Previous_Rank_away','Total_Points_away','points_dif']]

#Result 編碼 {'win': 0, 'lose': 1, 'draw': 2}
label_map = {'win': 0, 'lose': 1, 'draw': 2}
data_full['result'] = data_full['result'].map(label_map)

# print(data_full['result'])

#%%
#Seperate traning & testing data
data_simple=data_full[['result',
                       'odd_dif',
                       'ovr_dif',
                       'rank_dif']]

from sklearn.model_selection import train_test_split
X=data_simple.drop(labels=['result'],axis=1)
y=data_simple['result']
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=.3 , random_state=42)

print(X)
print(y)

print('Training data shape:',X_train.shape)
print('Testing data shape:',X_test.shape)

#%%
#Fitting random forest model

rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)

#Score
y_predict=rfc.predict(X_test)
y_predict
print('Score is : ',rfc.score(X_test,y_test))

