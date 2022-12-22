# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 20:25:38 2022

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
#PCA dimention reduction

from sklearn.decomposition import PCA

pca = PCA(n_components=2, iterated_power=1)
train_reduced = pca.fit_transform(X_train)
test_reduced = pca.transform(X_test)

#%%
#Fitting decision tree model

from sklearn.tree import DecisionTreeClassifier

# 建立Logistic模型
decisionTreeModel = DecisionTreeClassifier(criterion = 'entropy', max_depth=6, random_state=42)
# 使用訓練資料訓練模型
decisionTreeModel.fit(train_reduced, y_train)
# 使用訓練資料預測分類
predicted = decisionTreeModel.predict(train_reduced)

# 訓練集預測
print('train set accurancy: ',decisionTreeModel.score(train_reduced, y_train))
print('train set accurancy: ',decisionTreeModel.score(test_reduced, y_test))