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

data_full = pd.read_csv(r'D:\chuanght\097202\Coding\專題\TobbyProject\Word cup prediction\Tobby\xG_dif_prediction\cleanData_V3.csv', encoding='utf-8')

#Result 編碼 {'win': 0, 'lose': 1, 'draw': 2}
label_map = {'win': 0, 'lose': 1, 'draw': 2}
data_full['result'] = data_full['result'].map(label_map)

print(data_full.info())

#%%
#Simplifized dataframe
data_simple=data_full[['result',
                       'ovr_dif',
                       'att_dif',
                       'mid_dif',
                       'defe_dif',
                       'rank_dif',
                       'points_dif',
                       'Elo_ranking_dif',
                       'Elo_rating_dif']]

#%%
#Import xG_prediction
xG_predicted=xg(data_simple[['Elo_rating_dif']])
# print(xG_predicted)

data_simple.insert(len(data_simple.columns),'xG_predicted',xG_predicted)
print(data_simple.info())

#%%
#Seperate X & y data
X=data_simple.drop(labels=['result'],axis=1)
y=data_simple['result']


#%%
#Fitting random forest model
from sklearn.ensemble import RandomForestClassifier

def result_prediction(X_df):
    randomForestModel = RandomForestClassifier(n_estimators=100, criterion = 'gini')
    randomForestModel.fit(X, y)
    predicted = pd.Series(randomForestModel.predict(X_df))
    
    label_map_rollBack = {0: 'win', 1: 'lose', 2: 'draw'}
    predicted = predicted.map(label_map_rollBack)
    
    return predicted
    
