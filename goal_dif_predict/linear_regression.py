# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:44:35 2022

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
                     'Previous_Rank_home','Previous_Rank_away',
                     'Elo_rating_home','Elo_rating_away',
                     'goal_balance_home','goal_balance_away',
                     'win_rate_home','win_rate_away']]

print(data_simp.info())

#%%
#Plot highest winning rate country

#Group dataframe by nation
data_group=data_simp.groupby('Country_Code_home').mean()
print(data_group.info())

data_group_winRate=data_group.sort_values(by=['win_rate_home'],ascending=False).head(25)
print(data_group_winRate.info())

#Simplified nation name
country_short =  data_group_winRate.index
# print(country_short)

#Plot highest winning rate country
xticks = range(len(data_group_winRate))
labels=list(country_short)#Xticket label to nation name

#Color 漸層色
red = Color("green")
colors = list(red.range_to(Color("pink",'blue'),len(data_group_winRate)))
colors = [color.rgb for color in colors]

fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True,dpi=200)
ax2 = ax.twinx()

plt.xticks(xticks, labels,fontsize=10)
p1=ax.bar(xticks,data_group_winRate["win_rate_home"],width=0.5, color=colors)
ax.bar_label(p1,fmt='%.1f',padding=0,label_type='edge',size=10,color='blue')
ax.axis([-1,len(data_group_winRate),40,65])
ax.set_ylabel("Winning rate(%) ",size=10,color='blue')

ax2.plot(xticks, data_group_winRate['Previous_Rank_home'], color='red')
ax2.scatter(xticks, data_group_winRate['Previous_Rank_home'],s=30)
#Plot scattter text labels
for i, rank in enumerate (list(data_group_winRate['Previous_Rank_home'])):
    plt.text(xticks[i]+0.2, data_group_winRate['Previous_Rank_home'][i]+0.2, '%d' %rank,color='red')
    
ax2.axis([-1,len(data_group_winRate),100,0])
ax2.set_ylabel("FIFA ranking",size=10,color='red')

ax.set_xlabel("Countary",size=10)
plt.title("Winning rate per team",size=15)

plt.show()

#%%
#Plot highest goal balance country

data_group_goalBalance=data_group.sort_values(by=['goal_balance_home'],ascending=False).head(25)
print(data_group_goalBalance.info())

#Simplified nation name
country_short2 =  data_group_goalBalance.index
# print(country_short)

#Plot highest winning rate country
xticks = range(len(data_group_goalBalance))
labels=list(country_short2)#Xticket label to nation name

#Color 漸層色
red = Color("green")
colors = list(red.range_to(Color("pink",'blue'),len(data_group_goalBalance)))
colors = [color.rgb for color in colors]

fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True,dpi=200)
ax2 = ax.twinx()

plt.xticks(xticks, labels,fontsize=10)
p2=ax.bar(xticks,data_group_goalBalance["goal_balance_home"],width=0.5, color=colors)
ax.bar_label(p2,fmt='%.1f',padding=0,label_type='edge',size=10,color='blue')
ax.axis([-1,len(data_group_goalBalance),-0,1.5])
ax.set_ylabel("goal_balance ",size=10,color='blue')

ax2.plot(xticks, data_group_goalBalance['Previous_Rank_home'], color='red')
ax2.scatter(xticks, data_group_goalBalance['Previous_Rank_home'],s=30)
#Plot scattter text labels
for i, rank in enumerate (list(data_group_goalBalance['Previous_Rank_home'])):
    plt.text(xticks[i]+0.2, data_group_goalBalance['Previous_Rank_home'][i]+0.2, '%d' %rank,color='red')
    
ax2.axis([-1,len(data_group_goalBalance),100,0])
ax2.set_ylabel("FIFA ranking",size=10,color='red')

ax.set_xlabel("Countary",size=10)
plt.title("Goal balance per team",size=15)

plt.show()