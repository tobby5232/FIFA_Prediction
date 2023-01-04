# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 20:18:37 2022

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

data_simp=data_full[['date','home_team','away_team','Country_Code_home','Country_Code_away','goal_dif',
                     'ovr_dif','Previous_Rank_home','Previous_Rank_away',
                     'ovr_home','ovr_away',
                     'Elo_ranking_dif','Elo_rating_dif']]

insert_titles=['goal_dif','ovr_dif','Elo_ranking_dif','Elo_rating_dif']

print(data_simp.info())

#%%
#Groupby country_code
data_group_home=data_simp.groupby('Country_Code_home').mean()
data_group_home=data_group_home[['ovr_home','goal_dif','ovr_dif','Elo_ranking_dif','Elo_rating_dif']]


data_group_away=data_simp.groupby('Country_Code_away').mean()
data_group_away=data_group_away[['ovr_away','goal_dif','ovr_dif','Elo_ranking_dif','Elo_rating_dif']]
for i in range(len(insert_titles)):
    data_group_away[insert_titles[i]]=data_group_away[insert_titles[i]]*(-1)

data_group=data_group_home.add(data_group_away, fill_value=0)
data_group['ovr_home']=(data_group['ovr_home']+data_group['ovr_away'])/2
data_group=data_group.drop(['ISL','PAR','AUS','AUT','SCO','KSA','ECU','URU'], axis=0, inplace=False)

print(data_group)


#%%
#Plot highest goal_dif country

data_group_dif=data_group.sort_values(by=['goal_dif'],ascending=False).head(25)
# print(data_group_dif.info())

#Simplified nation name
country_short2 =  data_group_dif.index
# print(country_short)

#Plot highest winning rate country
xticks = range(len(data_group_dif))
labels=list(country_short2)#Xticket label to nation name

#Color 漸層色
red = Color("green")
colors = list(red.range_to(Color("blue",'red'),len(data_group_dif)))
colors = [color.rgb for color in colors]

fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True,dpi=200)
ax2 = ax.twinx()

plt.xticks(xticks, labels,fontsize=10)
p2=ax.bar(xticks,data_group_dif["goal_dif"],width=0.5, color=colors)
ax.bar_label(p2,fmt='%.1f',padding=0,label_type='edge',size=10,color='blue')
ax.axis([-1,len(data_group_dif),-0.6,2.6])
ax.set_ylabel("goal_dif",size=10,color='blue')

ax2.plot(xticks, data_group_dif['ovr_home'], color='red')
ax2.scatter(xticks, data_group_dif['ovr_home'],s=30)
#Plot scattter text labels
for i, rank in enumerate (list(data_group_dif['ovr_home'])):
    plt.text(xticks[i]+0.2, data_group_dif['ovr_home'][i]+0.2, '%d' %rank,color='red')
    
ax2.axis([-1,len(data_group_dif),60,90])
ax2.set_ylabel("Overall rating",size=10,color='red')

ax.set_xlabel("Countary",size=10)
plt.title("Goal difference per game",size=15)

plt.show()
#%%
#Plot scatter for Elo_rating_dif & goal_dif(predict_goal_dif)


from sklearn.linear_model import LinearRegression

# 建立LinearRegression模型
linearMmodel = LinearRegression(fit_intercept=True)
# 使用訓練資料訓練模型
linearMmodel.fit(data_simp[['Elo_rating_dif']], data_simp['goal_dif'])
# 使用訓練資料預測
predicted = linearMmodel.predict(data_simp[['Elo_rating_dif']])

from sklearn import metrics
print('R2 score: ', linearMmodel.score(data_simp[['Elo_rating_dif']], data_simp['goal_dif']))
mse = metrics.mean_squared_error(data_simp['goal_dif'], predicted)
print('MSE score: ', mse)

fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True,dpi=200)
plt.scatter(data_simp['Elo_rating_dif'], data_simp['goal_dif'],s=10, label='origin')
plt.plot(data_simp['Elo_rating_dif'], predicted,label='predicted',color='black')
plt.xlabel('Elo_rating_dif')
plt.ylabel('goal_dif')
plt.title('Linearity between goal_dif & Elo_rating_dif')
plt.legend()
plt.show()
