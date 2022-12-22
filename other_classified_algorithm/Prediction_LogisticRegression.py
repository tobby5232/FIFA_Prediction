# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 20:20:08 2022

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

print(data_full['result'])

#%%
#Seperate traning & testing data
data_simple=data_full[['result',
                       'odd_dif',
                       'ovr_dif',
                       'att_dif',
                       'mid_dif',
                       'defe_dif',
                       'rank_dif',
                       'points_dif']]

from sklearn.model_selection import train_test_split
X=data_simple.drop(labels=['result'],axis=1)
y=data_simple['result']
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=.3 , random_state=42)

print('Training data shape:',X_train.shape)
print('Testing data shape:',X_test.shape)

#%%
#Fitting Logistic regression model

from sklearn.linear_model import LogisticRegression

# 建立Logistic模型
logisticModel = LogisticRegression(random_state=0)
# 使用訓練資料訓練模型
logisticModel.fit(X_train, y_train)
# 使用訓練資料預測分類
predicted = logisticModel.predict(X_train)

# 預測成功的比例
print('訓練集: ',logisticModel.score(X_train,y_train))
print('測試集: ',logisticModel.score(X_test,y_test))

