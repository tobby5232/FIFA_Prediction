# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:50:42 2022

@author: ch406
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
#Dataframe information

data_full = pd.read_csv('cleanData.csv', encoding='utf-8')
data_full=data_full[['result',
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
#Defined X & y data
data_simple=data_full[['result',
                       'home_team_odd','away_team_odd','Draw_odd','odd_dif',
                       'ovr_home','ovr_away','ovr_dif',
                       'Previous_Rank_home','Previous_Rank_away','rank_dif',
                       'Total_Points_home','Total_Points_away','points_dif']]

from sklearn.model_selection import train_test_split
X=data_simple.drop(labels=['result'],axis=1)
y=data_simple['result']

#%%
#Seperate traning & testing data

X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=.3 , random_state=42)

print('Training data shape:',X_train.shape)
print('Testing data shape:',X_test.shape)

#%%

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

print('訓練集: ',gbc.score(X_train,y_train))
print('測試集: ',gbc.score(X_test,y_test))