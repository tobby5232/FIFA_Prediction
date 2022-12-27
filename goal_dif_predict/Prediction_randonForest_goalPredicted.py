# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 23:30:42 2022

@author: ch406
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colour import Color
import seaborn as sns

#%%
#Dataframe information

data_full = pd.read_csv('cleanData_V3.csv', encoding='utf-8')

data_simp=data_full[['result',
                     'goal_dif',
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

label_map = {'win': 0, 'lose': 1, 'draw': 2}
data_simp['result'] = data_simp['result'].map(label_map)


print(data_simp.info())

#%%
#Linear_regression ,predict goal_dif

from sklearn.linear_model import LinearRegression

# 建立LinearRegression模型
linearMmodel = LinearRegression(fit_intercept=True)
# 使用訓練資料訓練模型
linearMmodel.fit(data_simp[['Elo_ranking_dif']], data_simp['goal_dif'])
# 使用訓練資料預測
predicted = linearMmodel.predict(data_simp[['Elo_ranking_dif']])

data_simp.insert(len(data_simp.columns),'goal_dif_predicted',predicted)

from sklearn import metrics
print('R2 score: ', linearMmodel.score(data_simp[['Elo_ranking_dif']], data_simp['goal_dif']))
mse = metrics.mean_squared_error(data_simp['goal_dif'], predicted)
print('MSE score: ', mse)

print(data_simp.info())

#%%
#Random forest ,predict result

#Split X,y data
X=data_simp.drop(labels=['result','goal_dif'],axis=1)
y=data_simp['result']

#Split training & testing data
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=.3 , random_state=42)

print('Training data shape:',X_train.shape)
print('Testing data shape:',X_test.shape)

#%%
#Fitting random forest model

from sklearn.ensemble import RandomForestClassifier

# 建立RandomForest模型
randomForestModel = RandomForestClassifier(n_estimators=200, criterion = 'gini')
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
data_simp['result'] = data_simp['result'].map(label_map_rollBack)

data_simp=data_simp.drop(['goal_dif'],axis=1)
print(data_simp.info())

data_simp.to_csv("cleanData_goal_dif.csv", index = None)

#%%
#Feature importance
# data_simp=data_simp.drop(labels=['goal_dif'],axis=1)
data_simp = data_simp.set_index('result')

imp=randomForestModel.feature_importances_
Featur_names=list(data_simp.columns)

dic_imp={'Featur_names':Featur_names,'imp':imp}
df_imp=pd.DataFrame(dic_imp)
df_imp=df_imp.sort_values(by=['imp'],ascending=False)

print(df_imp)

