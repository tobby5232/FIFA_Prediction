# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 16:11:32 2022

@author: ch406
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
#Dataframe information
def pridicted_accuracy (file):
    data_full = pd.read_csv(file, encoding='utf-8')
    
    
    label_map = {'win': 0, 'lose': 1, 'draw': 2} #Result 編碼 {'win': 0, 'lose': 1, 'draw': 2}
    data_full['result'] = data_full['result'].map(label_map)
    data_full=data_full.drop(labels=['Draw_odd','odd_dif'],axis=1)
    
    print(data_full.info())
    
    from sklearn.model_selection import train_test_split
    X=data_full.drop(labels=['result'],axis=1) #Seperate X & y data
    y=data_full['result']
    
    X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=.3 , random_state=42) #Seperate traning & testing data
    
    from sklearn.ensemble import RandomForestClassifier #Fitting random forest model
    
    # 建立RandomForest模型
    randomForestModel = RandomForestClassifier(n_estimators=100, criterion = 'gini')
    # 使用訓練資料訓練模型
    randomForestModel.fit(X_train, y_train)
    # 使用訓練資料預測分類
    predicted = randomForestModel.predict(X_train)
    
    return randomForestModel.score(X_test,y_test)
    
#%%
# Multi-caculate model score

data_1=[]
data_2=[]
data_3=[]
dtat_goal=[]
data_xG=[]

for _ in range(100):
    d1=pridicted_accuracy('cleanData.csv')
    data_1.append(d1)
    d2=pridicted_accuracy('cleanData_V2.csv')
    data_2.append(d2)
    d3=pridicted_accuracy('cleanData_V3.csv')
    data_3.append(d3)
    dg=pridicted_accuracy('cleanData_goal_dif.csv')
    dtat_goal.append(dg)
    dxg=pridicted_accuracy('cleanData_xG_dif.csv')
    data_xG.append(dxg)
    
x_axis=[i for i in range(len(data_1))]

#%%
#Plot scatter for testing data accuracy
    
fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True,dpi=200)
# plt.scatter(x_axis, data_1,s=10, label='data_1',color='red')
plt.scatter(x_axis, data_2,s=10, label='origin_data',color='blue')
plt.scatter(x_axis, data_3,s=10, label='data_3',color='green')
plt.scatter(x_axis, dtat_goal,s=10, label='dtat_goal',color='yellow')
# plt.scatter(x_axis, data_xG,s=10, label='data_xG',color='gray')

plt.ylabel('Model score')
plt.legend()
plt.show()




