# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 11:20:07 2022

@author: ch406
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xG_prediction_linearRegression import xG_predicted_method as xg

#%%
#Dataframe information

data_full = pd.read_csv('cleanData_V3.csv', encoding='utf-8')
# data_full=data_full[['result',
#                      'home_score','away_score','goal_dif',
#                       'home_team_odd','away_team_odd','Draw_odd','odd_dif',
#                       'ovr_home','att_home','mid_home','defe_home','ovr_away','att_away','mid_away','defe_away',
#                       'ovr_dif','att_dif','mid_dif','defe_dif',
#                       'Previous_Rank_home','Total_Points_home','rank_dif',
#                       'Previous_Rank_away','Total_Points_away','points_dif']]

#Result 編碼 {'win': 0, 'lose': 1, 'draw': 2}
label_map = {'win': 0, 'lose': 1, 'draw': 2}
data_full['result'] = data_full['result'].map(label_map)

print(data_full.info())

#%%
#Simplifized dataframe
data_simple=data_full[['result',
                       'Draw_odd',
                       'odd_dif',
                       'ovr_dif',
                       'att_dif',
                       'mid_dif',
                       'defe_dif',
                       'rank_dif',
                       'points_dif',
                       'Elo_ranking_dif',
                       'Elo_rating_dif',
                       'goal_balance_dif']]

#%%
#Import xG_prediction
xG_predicted=xg(data_simple[['Elo_rating_dif']])
# print(xG_predicted)

data_simple.insert(len(data_simple.columns),'xG_predicted',xG_predicted)
# print(data_simple)

#%%
#Seperate X & y data
from sklearn.model_selection import train_test_split
X=data_simple.drop(labels=['result'],axis=1)
y=data_simple['result']

# print(data_simple.info())


#%%
#Standardization 平均&變異數標準化

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler().fit(X)
# X_scaled = scaler.transform(X)

# # scaled之後的資料零均值，單位方差  
# print('資料集 X 的平均值 : ', X.mean(axis=0))
# print('資料集 X 的標準差 : ', X.std(axis=0))

# print('\nStandardScaler 縮放過後資料集 X 的平均值 : ', X_scaled.mean(axis=0))
# print('StandardScaler 縮放過後資料集 X 的標準差 : ', X_scaled.std(axis=0))

#%%
#Seperate traning & testing data

X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=.3 , random_state=42)

print('Training data shape:',X_train.shape)
print('Testing data shape:',X_test.shape)

#%%
# #PCA dimention reduction

# from sklearn.decomposition import PCA

# pca = PCA(n_components=4, iterated_power=1)
# train_reduced = pca.fit_transform(X_train)
# test_reduced = pca.transform(X_test)



#%%
#Fitting random forest model

from sklearn.ensemble import RandomForestClassifier

# 建立RandomForest模型
randomForestModel = RandomForestClassifier(n_estimators=100, criterion = 'gini')
# 使用訓練資料訓練模型
randomForestModel.fit(X_train, y_train)
# 使用訓練資料預測分類
predicted = randomForestModel.predict(X_train)

# 預測成功的比例
print('訓練集: ',randomForestModel.score(X_train,y_train))
print('測試集: ',randomForestModel.score(X_test,y_test))

#%%
#Output data
label_map_rollBack = {0: 'win', 1: 'lose', 2: 'draw'}
data_simple['result'] = data_simple['result'].map(label_map_rollBack)

data_simple.to_csv("cleanData_xG_dif.csv", index = None)
print(data_simple.info())

#%%
#Feature importance
data_simple = data_simple.set_index('result')

imp=randomForestModel.feature_importances_
Featur_names=list(data_simple.columns)

dic_imp={'Featur_names':Featur_names,'imp':imp}
df_imp=pd.DataFrame(dic_imp)
df_imp=df_imp.sort_values(by=['imp'],ascending=False)

print(df_imp)

